using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, BSON, Printf
include("../inverted_pendulum/controllers/rule_based.jl")
## Train state-based controller

# Generate the environment
# env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π)
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

θmax = 1.2
ωmax = 3.1

# Generate training images
N = 10000
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    X[:, :, i] = obsfn(y[:, i])
end

# Visualize some fo the training data
ps = [plot(obsfn(y[:, i]), xaxis = false, xticks = false, yticks = false, titlefont = font(pointsize = 8), title = @sprintf("θ=%0.2f, ω=%0.2f", y[1, i], y[2, i])) for i = 1:10]
plot(ps..., layout = (2, 5))
# savefig("inverted_pendulum/figures/training_data.png")

# Prepare training data and model
data = Flux.DataLoader((X, y), batchsize = 1024)

loss(model) = (x, y) -> Flux.Losses.mse(model(x), y)

# Load in the risk network
risk_function = BSON.load("inverted_pendulum/risk_networks/rn_0.0.bson")[:model]

heatmap(-1:0.1:1, -1:0.1:1, (ϵ1, ϵ2) -> risk_function([0.3, 0, ϵ1, ϵ2])[1], clims=(0,π))
heatmap(-θmax:0.1:θmax, -ωmax:0.1:ωmax, (θ, ω) -> risk_function([θ, ω, 0, 0])[1])

function ρloss(model)
    (x, y) -> begin
        ŷ = model(x)
        ϵ = ŷ .- y 
        ϵ_risk = risk_function([y; ϵ])
        weighted_errors = ϵ .* ϵ_risk
        return mean(weighted_errors .^ 2)
    end
end

function ρ2loss(model)
    (x, y) -> begin
        ŷ = model(x)
        ϵ = ŷ .- y 
        ϵ_risk = risk_function([y; clamp.(ϵ, -0.6, 0.6)])
        return mean(ϵ_risk)
    end
end

function ρ3loss(model) 
    (x, y) -> begin
        ŷ = model(x)
        ϵ = ŷ .- y   
        ϵ_risk = risk_function([y; ϵ])
        return 0.1f0*Flux.mse(ŷ, y) + mean(ϵ_risk)
    end
end

r(ϵ) = risk_function([0.5, -1.0, 0, ϵ])[1]
ϵs = collect(-1:0.1:1)
plot(-6:0.1:6, r.(-6:0.1:6))
heatmap(-ωmax:0.1:ωmax, -ωmax:0.1:ωmax, (ω, ω̂) -> risk_function([0.5, ω, 0, clamp(ω̂ - ω, -0.6, 0.6)])[1])

function fplot(ω, ω̂)
    if abs(ω - ω̂) < 0.6
        return risk_function([0.5, ω, 0, clamp(ω̂ - ω, -0.6, 0.6)])[1]
    else
        return 0.0
    end
end

heatmap(-ωmax:0.1:ωmax, -ωmax:0.1:ωmax, fplot)

# Train the model
evalcb(model) = () -> println("train loss: ", ρ2loss(model)(X, y))
throttlecb2(model) = Flux.throttle(evalcb(model), 0.1)

model_reg = Chain(flatten, Dense(180, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [θmax, ωmax])
opt = ADAM(1e-3)
Flux.@epochs 100 Flux.train!(loss(model_reg), Flux.params(model_reg), data, opt, cb = throttlecb2(model_reg))

model = Chain(flatten, Dense(180, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [θmax, ωmax])
opt = ADAM(1e-3)
Flux.@epochs 100 Flux.train!(ρ3loss(model), Flux.params(model), data, opt, cb = throttlecb(model))

# Show the errors
ŷ = model(X)
# scatter(y[2, :], ŷ[2, :], label = "ω", alpha = 0.2, xlabel = "Ground Truth", ylabel = "Predicted", title = "Perception Model Accuracy")
scatter(y[1, :], ŷ[1, :], label = "θ", alpha = 0.2, xlabel="θ")
scatter!(y[1, :], ŷ[2, :], label = "ω", alpha = 0.2)
# savefig("inverted_pendulum/figures/perception_model_accuracy_nom.png")

ŷ = model_reg(X)
scatter(y[1, :], ŷ[1, :], label = "θ", alpha = 0.2, xlabel="θ")
scatter!(y[1, :], ŷ[2, :], label = "ω", alpha = 0.2)


## Evaluate the combined system

# Construct the image-based pendulum environment
img_env = ImageInvertedPendulum(λcost = 0.0f0, observation_fn = obsfn, θ0 = Uniform(-0.1,0.1), ω0 = Uniform(-0.1,0.1))

# Construct the full policy by chaining the perception and state-based
π_img = ContinuousNetwork(Chain((x) -> model(reshape(x, 180, 1)), (x) -> [action(simple_policy, x)]), 1)
π_img_reg = ContinuousNetwork(Chain((x) -> model_reg(reshape(x, 180, 1)), (x) -> [action(simple_policy, x)]), 1)

# Plot the resulting policy
heatmap(-1.5:0.05:1.5, -2:0.05:2, (θ, ω) -> action(simple_policy, model(reshape(Float32.(obsfn([θ, ω])), 180, 1))[:]), title = "Image Control Policy", xlabel = "θ", ylabel = "ω")
heatmap(-1.5:0.05:1.5, -2:0.05:2, (θ, ω) -> action(simple_policy, model_reg(reshape(Float32.(obsfn([θ, ω])), 180, 1))[:]), title = "Image Control Policy", xlabel = "θ", ylabel = "ω")
# savefig("inverted_pendulum/figures/image_control_policy_risk.png")


# s= rand(initialstate(img_env))
# o = observation(img_env,s)
# 
# for i=1:10
#     s,o,r = gen(img_env, s, action(π_img, o)[1])
#     println(s)
# end 


# Get the return and plot an episode
undiscounted_return(Sampler(img_env, π_img), Neps = 100)
undiscounted_return(Sampler(img_env, π_img_reg), Neps = 100)
Crux.gif(img_env, π_img, "inverted_pendulum/figures/img_pendulum_risk.gif")
Crux.gif(img_env, π_img_reg, "inverted_pendulum/figures/img_pendulum_reg.gif")

rand(initialstate(img_env))

