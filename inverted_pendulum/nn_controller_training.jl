using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, BSON, Printf
using Zygote
include("../inverted_pendulum/controllers/rule_based.jl")

# Generate the environment
env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π)

# Define the networks we will use
A() = DiscreteNetwork(Chain(Dense(2, 12, relu), Dense(12, 2)), env.actions)
C() = ContinuousNetwork(Chain(Dense(2, 12, relu), Dense(12, 1)))

# Solve for a policy
solver = PPO(π=ActorCritic(A(), C()), S=state_space(env), N=5000, ΔN=100)
policy = solve(solver, env)
BSON.@save "inverted_pendulum/controllers/policy.bson" policy
# π_ppo = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]


# Create some visualizations
Crux.gif(env, policy, "inverted_pendulum/figures/pendulum_control.gif")
heatmap(-0.4:0.05:0.4, -1:0.05:1, (θ, ω) -> action(policy, [θ, ω])[1], title="Pendulum Control Policy", xlabel="θ", ylabel="ω")
savefig("inverted_pendulum/figures/controller_policy.png")