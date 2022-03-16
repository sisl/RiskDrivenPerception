using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

s0 = [0.0, 0, 0.0]
# Load the environmetn and policy
env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π,
    θ0 = Uniform(s0[2], s0[2] + 1e-6),
    ω0 = Uniform(s0[3], s0[3] + 1e-6))
nn_policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]
simple_policy = FunPolicy(f)

# heatmap(θs, ωs, (θ, ω) -> action(policy, [θ, ω])[1], title = "Pendulum Control Policy", xlabel = "θ", ylabel = "ω")
rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)

# Plot the grids
scatter(θs, zeros(length(θs)))
scatter(ωs, zeros(length(ωs)))
scatter(ϵ1s, zeros(length(ϵ1s)))
scatter(ϵ2s, zeros(length(ϵ2s)))
scatter(cost_points, zeros(length(cost_points)))

# Get the distribution of returns and plot
N = 10000
D = episodes!(Sampler(rmdp, px), Neps = N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, label = "", title = "Pendulum Costs", xlabel = "|θ|", bins = range(0, π, step = 0.1))

# Grab an initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt(s0))
si = si[argmax(wi)]

# Solve for distribution over costs
@time Qw = solve_cvar_fixed_particle(rmdp, px.distribution, s_grid, 𝒮, s2pt, cost_points);

# Plot distribution at state s0
p2 = histogram(cost_points, weights=Qw[1][si], normalize=true, bins=range(0, π, step = 0.1))
plot(p1, p2)

# Create CVaR convenience functions
normalized_CVaR(s, ϵ, α) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α)
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# Plot one sample
heatmap(θs, ωs, (x, y) -> CVaR([x, y], [0, 0], 0), title = "α = 0", clims = (0, π))

# Sweep through α and create a gif
anim = @animate for α in range(-1.0, 1.0, length = 51)
    heatmap(θs, ωs, (x, y) -> CVaR([x, y], [0, 0], α), title = "CVaR (α = $α)", clims = (0, π), xlabel="θ (rad)", ylabel="ω (rad/s)")
end
Plots.gif(anim, fps = 6)


