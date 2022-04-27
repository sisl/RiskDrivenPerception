using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")

# Load the environment and policy
s0 = [0.0, 0.0, 0.0, 40.0]
env = DetectAndAvoidMDP(h0_dist=Uniform(s0[1] - 1e-16, s0[1] + 1e-16),
    dh0_dist=Uniform(s0[2] - 1e-16, s0[2] + 1e-16), ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
    actions=[-8.0, 0.0, 8.0])

hmax = 200
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
scatter(hs, zeros(length(hs)))
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

policy = OptimalDetectAndAvoidPolicy(env, hs, dhs, τs)

# Plot a slice of the policy
heatmap(τs, hs, (τ, h) -> action(policy, [h, 0.0, 0.0, τ]), xlabel="τ (s)", ylabel="h (m)", title="CAS Policy")

# Set up the cost function and risk mdp
costfn(m, s, sp) = isterminal(m, sp) ? 200 - abs(s[1]) : 0.0
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

# Start with just detect noise
p_detect(s) = sigmoid(-0.3232s[4] + 3.2294)
function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    return ObjectCategorical(noises, [1 - pd, pd])
end

#detect_prob = 0.9
noises_detect = [0, 1]
#probs_detect = [1.0 - detect_prob, detect_prob]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0, 0.0, 0.0] for ϵ in ϵ_grid]
#probs = probs_detect # NOTE: will need to change once also have additive noise

#px = DistributionPolicy(ObjectCategorical(noises, probs))
px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

# Get the distribution of returns and plot
N = 1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, title="CAS Costs", bins=range(0, 200, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 200, 50))
s_grid = RectangleGrid(hs, dhs, env.actions, τs)
𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs];
s2pt(s) = s

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);

# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt(s0))
si = si[argmax(wi)]

p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 200, 50), normalize=true, alpha=0.4, label="DP")

# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# Plot one sample
heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0], 0), title="α = 0")

anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0], α), title="CVaR (α = $α)", clims=(200, 700), xlabel="τ (s)", ylabel="h (m)")
end
Plots.gif(anim, "collision_avoidance/figures/daa_CVaR_v2.gif", fps=6)

# Most important states
riskmin(x; α) = minimum([CVaR(x, [noise], α) for noise in noises_detect])
riskmax(x; α) = maximum([CVaR(x, [noise], α) for noise in noises_detect])
risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)

heatmap(0:0.5:40, -200:5:200, (x, y) -> risk_weight([y, 0.0, 0.0, x], α=0.0), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors")#, clims=(0, 20))

anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(0:0.5:40, -200:5:200, (x, y) -> risk_weight([y, 0.0, 0.0, x], α=α), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors: α = $(α)", clims=(0, 20))
end
Plots.gif(anim, "collision_avoidance/figures/daa_risk_weights.gif", fps=6)

# Sampling distribution for risk-driven data generation
