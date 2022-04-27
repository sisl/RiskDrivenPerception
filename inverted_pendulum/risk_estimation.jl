using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

s0 = [0.0, 0.19691466527847248, 0.0]
# Load the environmetn and policy
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π/4,
    θ0=Uniform(s0[2], s0[2] + 1e-16),
    ω0=Uniform(s0[3], s0[3] + 1e-16))

# nn_policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]
simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))

# heatmap(θs, ωs, (θ, ω) -> action(simple_policy, [θ, ω])[1], title = "Pendulum Control Policy", xlabel = "θ", ylabel = "ω")
rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)

# Plot the grids
scatter(θs, zeros(length(θs)))
scatter(ωs, zeros(length(ωs)))
scatter(ϵ1s, zeros(length(ϵ1s)))
scatter(ϵ2s, zeros(length(ϵ2s)))
scatter(cost_points, zeros(length(cost_points)))

# Get the distribution of returns and plot
N = 1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, title="Pendulum Costs", label="MC", xlabel="|θ|", bins=range(0, π/4, step=0.1), alpha=0.3, normalize=true)

# # Grab an initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt(s0))
si = si[argmax(wi)]

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt, cost_points);

# Plot distribution at state s0
# Uw_replace = mean([px.distribution.p[ai] * Qw[ai][si] for ai in 1:length(px.distribution.objs)])
p2 = histogram(cost_points, weights=Uw[si], normalize=true, bins=range(0, π/4, step=0.1), alpha=0.4, label="Dynamic Programming")
# # plot(p1, p2)
# vline!([1.3], label="VaR")

# savefig("inverted_pendulum/figures/cost_distribution.png")

# Create CVaR convenience functions
normalized_CVaR(s, ϵ, α) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α)
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# Plot one sample
heatmap(-π/4:0.1:π/4, -2:0.1:2, (x, y) -> CVaR([x, y], [0, 0], 0), title="α = 0")
heatmap(θs, ωs, (x, y) -> CVaR([x, y], [0, 0], 0), title="α = 0", clims=(0, π))
heatmap(-1:0.1:1, -1:0.1:1, (x, y) -> CVaR([0.2, 0], [x, y], 0), title="α = 0")

# Sweep through α and create a gif
anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(-π/4:0.05:π/4, -2:0.05:2, (x, y) -> CVaR([x, y], [0, 0], α), title="CVaR (α = $α)", clims=(0, π/4), xlabel="θ (rad)", ylabel="ω (rad/s)")
end
Plots.gif(anim, "inverted_pendulum/figures/CVaR.gif", fps=6)

# For a given theta, pick the omega that minimizes CVaR
# function get_min_ω(Qw, θ, ωs, ϵ2s, cost_points; α)
#     for ω in ωs
#         ω̂s = ω .+ ϵ2s

#     rms = [mean([CVaR([θ; ω], [0.0, ϵ2], α) for ω in ωs]) for ϵ2 in ϵs]
#     ϵ = 
#     return ωs[argmin(rms[i])]
# end


## Control with noise
θmax = π/4
ωmax = 2.0

function riskmin_perception(x; λ=0f0, α=0f0)
    ind = argmin([CVaR(x, [ϵ1, ϵ2], α)[1] + λ * 0.5 * (ϵ1^2 + ϵ2^2) for ϵ1 in ϵ1s, ϵ2 in ϵ2s])
    return [x[1] + ϵ1s[ind.I[1]], x[2] + ϵ2s[ind.I[2]]]
end

α = 0.0f0
λ = 10f0
policy = ContinuousNetwork(Chain((x) -> riskmin_perception(x; α, λ), (x) -> [action(simple_policy, x)]), 1)

env = InvertedPendulumMDP(λcost=0.f0, failure_thresh=π/4)
# undiscounted_return(Sampler(env, policy), Neps=100)

anim = @animate for λ = 10 .^ (2:-0.1:-3)
    println("λ: $λ")
    policy = ContinuousNetwork(Chain((x) -> riskmin_perception(x; α, λ), (x) -> [action(simple_policy, x)]), 1)
    heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> clamp.(action(policy, [θ, ω])[1], -2, 2), clims=(-2,2), title="Risk Sensitive Control Policy (λ=$λ)", xlabel="θ", ylabel="ω")
end

Plots.gif(anim, fps=4)

Crux.gif(env, policy, "out.gif", max_steps=10, Neps =10)


heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) ->  riskmin_perception([θ, ω]; α, λ)[2], xlabel="θ", ylabel="ω", title="ω̂")
heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) ->  riskmin_perception([θ, ω]; α, λ)[1], xlabel="θ", ylabel="ω", title="θ̂",)


s = [0.205646928405678, 0]
α = 0
p1 = heatmap(-.4:0.05:0.4, -1:0.05:1, (θ̂, ω̂) -> CVaR(s, [θ̂, ω̂], α))
p2 = heatmap(-.4:0.05:0.4, -1:0.05:1, (θ, ω) -> clamp.(action(simple_policy, s .+ [θ, ω]), -2, 2))
plot(p1, p2, size= (1200, 400))




