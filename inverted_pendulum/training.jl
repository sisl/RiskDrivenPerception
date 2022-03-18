using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, BSON, Printf
include("../inverted_pendulum/controllers/rule_based.jl")
## Train state-based controller

# Generate the environment
env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π)
simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))

# # Define the networks we will use
# A() = DiscreteNetwork(Chain(Dense(2, 12, relu), Dense(12, 2)), env.actions)
# C() = ContinuousNetwork(Chain(Dense(2, 12, relu), Dense(12, 1)))

# # Solve for a policy
# solver = PPO(π=ActorCritic(A(), C()), S=state_space(env), N=5000, ΔN=100)
# policy = solve(solver, env)
# BSON.@save "inverted_pendulum/controllers/policy.bson" policy
# # π_ppo = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]


# # Create some visualizations
# Crux.gif(env, policy, "inverted_pendulum/figures/pendulum_control.gif")
# heatmap(-0.4:0.05:0.4, -1:0.05:1, (θ, ω) -> action(policy, [θ, ω])[1], title="Pendulum Control Policy", xlabel="θ", ylabel="ω")
# savefig("inverted_pendulum/figures/controller_policy.png")


## Train the perception system
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, show_prev = false)

# Generate training images
N = 1000
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-0.4, 0.4)), rand(Uniform(-1.26, 1.26))]
    X[:, :, i] = obsfn(y[:, i])
end

# Visualize some fo the training data
ps = [plot(obsfn(y[:, i]), xaxis = false, xticks = false, yticks = false, titlefont = font(pointsize = 8), title = @sprintf("θ=%0.2f, ω=%0.2f", y[1, i], y[2, i])) for i = 1:10]
plot(ps..., layout = (2, 5))
# savefig("inverted_pendulum/figures/training_data.png")

# Prepare training data and model
data = Flux.DataLoader((X, y), batchsize = 128)
model_reg = Chain(flatten, Dense(180, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [0.4, 1.26])
model = Chain(flatten, Dense(180, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [0.4, 1.26])
opt = ADAM(1e-3)
loss(x, y) = Flux.Losses.mse(model_reg(x), y)

# Load in the risk network
risk_function = BSON.load("inverted_pendulum/risk_networks/risk_net_k1_0_unnorm.bson")[:model]

risk_function([0.5, 0, 0, 0])[1]

function ρloss(x, y)
    ŷ = model(x)
    ϵ = y .- ŷ
    ϵ_risk = risk_function([y; ϵ])
    weighted_errors = ϵ .* ϵ_risk
    return mean(weighted_errors .^ 2)
end

function ρ2loss(x, y)
    ŷ = model(x)
    ϵ = y .- ŷ
    ϵ_risk = risk_function([y; ϵ])
    return mean(ϵ_risk)
end

function ρ3loss(x, y)
    ŷ = model(x)
    ϵ = y .- ŷ
    ϵ_risk = risk_function([y; ϵ])
    return Flux.mse(ŷ, y) + mean(ϵ_risk)
end

# Train the model
evalcb() = println("train loss: ", ρ2loss(X, y))
throttlecb = Flux.throttle(evalcb, 0.1)

Flux.@epochs 1000 Flux.train!(loss, Flux.params(model_reg), data, opt, cb = throttlecb)
opt = ADAM(1e-3)
Flux.@epochs 1000 Flux.train!(ρ2loss, Flux.params(model), data, opt, cb = throttlecb)

# Show the errors
ŷ = model(X)

scatter(y[2, :], ŷ[2, :], label = "ω", alpha = 0.2, xlabel = "Ground Truth", ylabel = "Predicted", title = "Perception Model Accuracy")
scatter!(y[1, :], ŷ[1, :], label = "θ", alpha = 0.2)
# savefig("inverted_pendulum/figures/perception_model_accuracy_nom.png")

ŷ = model_reg(X)
scatter(y[2, :], ŷ[2, :], label = "ω", alpha = 0.2, xlabel = "Ground Truth", ylabel = "Predicted", title = "Perception Model Accuracy")
scatter!(y[1, :], ŷ[1, :], label = "θ", alpha = 0.2)


## Evaluate the combined system

# Construct the image-based pendulum environment
img_env = ImageInvertedPendulum(λcost = 0.0f0, observation_fn = obsfn)

# Construct the full policy by chaining the perception and state-based
π_img = ContinuousNetwork(Chain((x) -> model(reshape(x, 180, 1)), (x) -> [action(simple_policy, x)]), 1)
π_img_reg = ContinuousNetwork(Chain((x) -> model_reg(reshape(x, 180, 1)), (x) -> [action(simple_policy, x)]), 1)

simple_policy

# Plot the resulting policy
heatmap(-0.4:0.05:0.4, -1:0.05:1, (θ, ω) -> action(simple_policy, model(reshape(Float32.(obsfn([θ, ω])), 180, 1))[:]), title = "Image Control Policy", xlabel = "θ", ylabel = "ω")
heatmap(-0.4:0.05:0.4, -1:0.05:1, (θ, ω) -> action(simple_policy, model_reg(reshape(Float32.(obsfn([θ, ω])), 180, 1))[:]), title = "Image Control Policy", xlabel = "θ", ylabel = "ω")
# savefig("inverted_pendulum/figures/image_control_policy_risk.png")


s= rand(initialstate(img_env))
o = observation(img_env,s)


for i=1:10
    s,o,r = gen(img_env, s, action(π_img, o)[1])
    println(s)
end 


# Get the return and plot an episode
undiscounted_return(Sampler(img_env, π_img), Neps = 10)
undiscounted_return(Sampler(img_env, π_img_reg), Neps = 10)
Crux.gif(img_env, π_img, "inverted_pendulum/figures/img_pendulum.gif")
Crux.gif(img_env, π_img_reg, "inverted_pendulum/figures/img_pendulum.gif")

