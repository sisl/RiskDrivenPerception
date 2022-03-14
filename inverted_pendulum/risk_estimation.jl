using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
using ImportanceWeightedRiskMetrics
include("../src/risk_solvers.jl")

# Load the environmetn and policy
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π)
policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]

# Convert into a cost function
tmax = 5*env.dt
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
rmdp = RMDP(env, policy, costfn, true, env.dt, tmax, :noise)

# Set the nominal distribution of noise
noises = [[0,0], [-1,-1], [1,1]]
probs = [0.8, 0.1, 0.1]
# probs = [1.0, 0, 0, 0, 0, 0, 0]
px = DistributionPolicy(ObjectCategorical(noises, probs))

# Get the distribution of returns
N=1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]] 

histogram(samples, label="", title="Pendulum Costs", xlabel="|θ|")

## Solve for the risk function

# Define the grid for interpolation
θs = -0.3:0.01:.3
ωs = -1:0.1:1
ts = 0:env.dt:tmax
grid = RectangleGrid(θs, ωs, ts)

# Define the state space and mapping to the grid
𝒮 = [[tmax-t, θ, ω] for θ in θs, ω in ωs, t in ts]
s2pt(s) = [s[2:end]..., tmax-s[1]]

length(𝒮)

# Define the update functions
rcondition(r) = r

# solve the bellman update
# ρ = solve_conditional_bellman(rmdp, px.distribution, rcondition, grid, 𝒮, s2pt)
Qp, Qw = solve_cvar_particle(rmdp, px.distribution, grid, 𝒮, s2pt)

# Grab an initial state
s0 = rand(initialstate(rmdp))
si, wi = GridInterpolations.interpolants(grid, s2pt(s0))
si = si[argmax(wi)]

histogram(Qp[1][si], weights=Qw[1][si], normalize=true)

rm = IWRiskMetrics(Qp[1][si], length(Qw[1][si]) * Qw[1][si], 0.05)
vline!([rm.var], label="α=0.05")

rm = IWRiskMetrics(Qp[1][si], length(Qw[1][si]) * Qw[1][si], 0.01)
vline!([rm.var], label="α=0.01")

plot(rm.est.Xs, rm.est.partial_Ws ./ length(Qw[1][si]))
hline!([0.95])
hline!([0.99])


# Plot some of the results
t = .1
ω = 0
p = plot(xlabel="θ", ylabel="Expected Return", title="Expected return at t=$t, ω=$ω")
for i in 1:length(ρ)
	plot!(θs, (θ) -> interpolate(grid, ρ[i], s2pt([t, θ, 0])), label="Noise: $(noises[i])")
end
p

ps = [heatmap(θs, ωs, (θ, ω) -> interpolate(grid, ρ[i], s2pt([t, θ, ω])), title="Noise: $(noises[i])") for i in 1:length(ρ)]
plot(ps...,)

