using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, BSON, Printf

## Train state-based controller

# Generate the environment
env = InvertedPendulumMDP(λcost=0.1f0)

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


## Train the perception system
obsfn = (s)->POMDPGym.simple_render_pendulum(s, show_prev=false)

# Generate training images
N = 1000
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, N)
for i=1:N
	y[:, i] = [rand(Uniform(-0.4, 0.4)), rand(Uniform(-2, 2))]
	X[:, :, i] = obsfn(y[:,i])
end

# Visualize some fo the training data
ps = [plot(obsfn(y[:,i]), xaxis=false, xticks=false, yticks=false, titlefont=font(pointsize=8), title=@sprintf("θ=%0.2f, ω=%0.2f", y[1,i], y[2,i])) for i=1:10]
plot(ps..., layout=(2,5))
savefig("inverted_pendulum/figures/training_data.png")

# Prepare training data and model
data = Flux.DataLoader((X,y), batchsize=128)
model = Chain(flatten, Dense(180, 64, relu), Dense(64,64,relu), Dense(64, 2))
opt = ADAM(1e-3)
loss(x,y) = Flux.Losses.mse(model(x), y)
function ρloss(x, y)
	# ϵ = model(x) .- y
	ϵ = zeros(4, size(y,2))
	res = zeros(1, size(y, 2))
	for i=1:size(y,2)
		res[i] = foo(ϵ[:,i])
	end
	Flux.mean((ϵ .^ 2) .* res)
end


# Train the model
# Flux.@epochs 100 Flux.train!(loss, Flux.params(model), data, opt)
Flux.@epochs 100 Flux.train!(ρloss, Flux.params(model), data, opt)

# Show the errors
ŷ = model(X)

scatter(y[2,:], ŷ[2,:], label = "ω", alpha=0.2, xlabel="Ground Truth", ylabel="Predicted", title="Perception Model Accuracy")
scatter!(y[1,:], ŷ[1,:], label = "θ", alpha=0.2)
savefig("inverted_pendulum/figures/perception_model_accuracy.png")


## Evaluate the combined system

# Construct the image-based pendulum environment
img_env = ImageInvertedPendulum(λcost=0.0f0, observation_fn=obsfn)

# Construct the full policy by chaining the perception and state-based
π_img = DiscreteNetwork(Chain((x) -> model(reshape(obsfn(x), 180,1)), (x) -> policy.A.network(x)), env.actions)


# Plot the resulting policy
heatmap(-0.4:0.05:0.4, -1:0.05:1, (θ, ω) -> action(policy, model(reshape(obsfn([θ, ω]), 180,1)))[1], title="Image Control Policy", xlabel="θ", ylabel="ω")
savefig("inverted_pendulum/figures/image_control_policy.png")

# Get the return and plot an episode
undiscounted_return(Sampler(img_env, π_img), Neps=10)
Crux.gif(img_env, π_img, "inverted_pendulum/figures/img_pendulum.gif")

