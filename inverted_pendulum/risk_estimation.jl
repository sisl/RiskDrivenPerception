using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")

# Load the environmetn and policy
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π)
policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]

# Convert into a cost function
tmax = 20*env.dt
costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
rmdp = RMDP(env, policy, costfn, true, env.dt, tmax, :noise)

# Set the nominal distribution of noise
noises = [[0,0], [-1,0], [0,-1], [1,0], [0,1], [-1,-1], [1,1]]
probs = [0.958, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001]
# probs = [1.0, 0, 0, 0, 0, 0, 0]
px = DistributionPolicy(ObjectCategorical(noises, probs))

# Get the distribution of returns
N=1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

histogram(samples, label="", title="Pendulum Costs", xlabel="|θ|")

## Solve for the risk function

# Define the grid for interpolation
θs = -1:0.05:1
ωs = -2:0.1:2
ts = 0:env.dt:tmax
grid = RectangleGrid(θs, ωs, ts)

# Define the state space and mapping to the grid
𝒮 = [[tmax-t, θ, ω] for θ in θs, ω in ωs, t in ts]
s2pt(s) = [s[2:end]..., tmax-s[1]]

# Define the update functions
rcondition(r) = r>0.4

# solve the bellman update
ρ = solve_conditional_bellman(rmdp, px.distribution, rcondition, grid, 𝒮, s2pt)

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

