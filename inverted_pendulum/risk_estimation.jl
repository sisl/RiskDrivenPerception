using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")

s0 = [0.0, 0, 0.0]
# Load the environmetn and policy
env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π,
    θ0 = Uniform(s0[2], s0[2] + 1e-6),
    ω0 = Uniform(s0[3], s0[3] + 1e-6))
policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]

# heatmap(θs, ωs, (θ, ω) -> action(policy, [θ, ω])[1], title = "Pendulum Control Policy", xlabel = "θ", ylabel = "ω")


# Convert into a cost function
tmax = 20 * env.dt
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
rmdp = RMDP(env, policy, costfn, true, env.dt, tmax, :noise)

# Set the nominal distribution of noise
noises_1_half = 10 .^ (collect(range(-1, stop = log10(1), length = 5)))
noises_1 = [reverse(-noises_1_half); 0.0; noises_1_half]
scatter(noises_1, zeros(length(noises_1)))
probs_1 = [pdf(Normal(0, 0.5), n) for n in noises_1]
probs_1 ./= sum(probs_1)
noises_2_half = 10 .^ (collect(range(-1, stop = log10(1), length = 5)))
noises_2 = [reverse(-noises_2_half); 0.0; noises_2_half]
scatter(noises_2, zeros(length(noises_2)))
probs_2 = [pdf(Normal(0, 0.5), n) for n in noises_2]
probs_2 ./= sum(probs_2)
noises = [[n1, n2] for n1 in noises_1 for n2 in noises_2]
probs = [p1 * p2 for p1 in probs_1 for p2 in probs_2]
# probs = [1.0, 0, 0, 0, 0, 0, 0]
px = DistributionPolicy(ObjectCategorical(noises, probs))

# Get the distribution of returns
N = 10000
D = episodes!(Sampler(rmdp, px), Neps = N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, label = "", title = "Pendulum Costs", xlabel = "|θ|", bins = range(0, π, step = 0.1))

## Solve for the risk function

# Define the grid for interpolation
# θs_half = 2 .^ (collect(range(-8, stop = log2(π), length = 25)))
# θs = [reverse(-θs_half); θs_half]
θs_half = π .- sqrt.(collect(range(0, stop = π^2, length = 25)))
θs = [-θs_half[1:end-1]; reverse(θs_half)]
scatter(θs, zeros(length(θs)))
# ωs_half = 10 .^ (collect(range(-2, stop = log10(8), length = 25)))
# ωs = [reverse(-ωs_half); ωs_half]
ωs_half = 8 .- sqrt.(collect(range(0, stop = 8^2, length = 25)))
ωs = [-ωs_half[1:end-1]; reverse(ωs_half)]
scatter(ωs, zeros(length(ωs)))
ts = 0:env.dt:tmax
grid = RectangleGrid(θs, ωs, ts)

# Define the state space and mapping to the grid
𝒮 = [[tmax - t, θ, ω] for θ in θs, ω in ωs, t in ts]
s2pt(s) = [s[2:end]..., tmax - s[1]]

length(𝒮)

# Define the update functions
rcondition(r) = r

# solve the bellman update
# ρ = solve_conditional_bellman(rmdp, px.distribution, rcondition, grid, 𝒮, s2pt)
# Qp, Qw = solve_cvar_particle(rmdp, px.distribution, grid, 𝒮, s2pt)

# Grab an initial state
s0 = rand(initialstate(rmdp))
si, wi = GridInterpolations.interpolants(grid, s2pt(s0))
si = si[argmax(wi)]
si
# histogram(Qp[1][si], weights = Qw[1][si], normalize = true, bins = range(0, 0.25, step = 0.01))

#cost_points = collect(range(0, 0.25, length = 100))
cost_points = 10 .^ (collect(range(-2, stop = log10(π), length = 100)))
scatter(cost_points, zeros(length(cost_points)))
@time Qw_fixed = solve_cvar_fixed_particle(rmdp, px.distribution, grid, 𝒮, s2pt, cost_points);
Qw_fixed[1][si]
p2 = histogram(cost_points, weights = Qw_fixed[1][si], normalize = true, bins = range(0, π, step = 0.1))
plot(p1, p2)

ϵ_grid = RectangleGrid(noises_1, noises_2)
ρ2(s, ϵ, α) = ρ(s2pt([0.0, s...]), ϵ, grid, ϵ_grid, Qw_fixed, cost_points, α = α)[1]

heatmap(θs, ωs, (x, y) -> ρ2([x, y], [0, 0], -0.5))
ρ2([0, 0], [0, 0], 0)

anim = @animate for α in range(-1, 1, length = 20)
    heatmap(θs, ωs, (x, y) -> ρ2([x, y], [0, 0], α), title = "α = $α")
end

Plots.gif(anim, fps = 2)

s = [0, -2, 0]
si, wi = GridInterpolations.interpolants(grid, s2pt(s))
si = si[argmax(wi)]
si
histogram(cost_points, weights = Qw_fixed[61][si], normalize = true)

[ρ2([x, y], [0, 0], -0.5) for x in θs, y in ωs]

noises[1]
s2pt(s0)
cvar, var = ρ(s2pt(s0), noises[1], grid, ϵ_grid, Qw_fixed, cost_points, α = 0.95)

histogram(cost_points, weights = Qw_fixed[1][si], normalize = true, bins = range(0, π, step = 0.1))
vline!([var])

cvar


###### New Policy ########
struct FunPolicy <: Policy
    f
end

function POMDPs.action(p::FunPolicy, s)
    return f(s)
end

function f(s)
    return (-8 / π) * s[1] < s[2] ? -1.0 : 1.0
end

simple_policy = FunPolicy(f)

heatmap(θs, ωs, (θ, ω) -> action(simple_policy, [θ, ω])[1], title = "Pendulum Control Policy", xlabel = "θ", ylabel = "ω")

# Specify cost function
function costfn(m, s, sp)
    if isterminal(m, sp)
        if abs(s[2]) ≤ deg2rad(15) && abs(s[3]) ≤ deg2rad(5)
            return 0.0
        else
            return abs(s[2]) + s[3]^2
        end
    else
        return 0.0
    end
end

function failure_costfn(m, s, sp)
    if isterminal(m, sp)
        if abs(s[2]) ≤ deg2rad(15) && abs(s[3]) ≤ deg2rad(5)
            return 0.0
        else
            return 1.0
        end
    end
end

# Create mdp
simple_rmdp = RMDP(env, simple_policy, costfn, true, env.dt, tmax, :noise)
Qw_simple = solve_cvar_fixed_particle(simple_rmdp, px.distribution, grid, 𝒮, s2pt, cost_points)

ρ3(s, ϵ, α) = ρ(s2pt([0.0, s...]), ϵ, grid, ϵ_grid, Qw_simple, cost_points, α = α)[1]
heatmap(θs, ωs, (x, y) -> ρ3([x, y], [0, 0], -0.5), clims=(0, π))

for α in range(-1, 1, length = 11)
    p = heatmap(θs, ωs, (x, y) -> ρ3([x, y], [0, 0], α), title = "α = $α")
    display(p)
end

anim = @animate for α in range(-0.95, 0.95, length = 20)
    heatmap(θs, ωs, (x, y) -> ρ3([x, y], [0, 0], α), title = "α = $α", clims = (0, π))
end

Plots.gif(anim, fps = 2)