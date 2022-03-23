using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, Measures, BSON, Printf, Zygote, Images
include("../inverted_pendulum/controllers/rule_based.jl")

## Plots of the perception system
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0.))

frames = [obsfn([θ, 0])[:,:,1] for θ in sin.(0:.1:2π)]
save("inverted_pendulum/figures/noisefree_pendulum_images.gif", cat(frames..., dims=3), fps=10)


## Rule-based control policy
env = InvertedPendulumMDP(λcost = 0f0, failure_thresh = π/4)

simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))

heatmap(-1.:0.05:1., -1:0.05:1, (θ, ω) -> clamp(action(simple_policy, [θ, ω])[1], -2.0, 2.0), title="Pendulum Control Policy", xlabel="θ", ylabel="ω")
savefig("inverted_pendulum/figures/continuous_rule_control.png")

Crux.gif(env, simple_policy, "inverted_pendulum/figures/state_based_control.gif")

## Image-based performance
θmax = 1.0
ωmax = 2.0
N = 10000
mse_loss(model) = (x, y) -> Flux.Losses.mse(model(x), y)

# Generate noisefree training images
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0.))
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    X[:, :, :, i] = obsfn(y[:, i])
end
data = Flux.DataLoader((X, y), batchsize=1000)

model=Chain(flatten, Dense(360, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [θmax, ωmax])
opt = ADAM(1e-3)
Flux.@epochs 100 Flux.train!(mse_loss(model), Flux.params(model), data, opt)

ŷ = model(X)
scatter(y[2, :], ŷ[2, :], alpha = 0.2, label = "ω", color=2)
scatter!(y[1, :], ŷ[1, :], label="θ", alpha=0.2, xlabel="Ground Truth", ylabel = "Predicted", title = "Perception Model Accuracy", color=1)

savefig("inverted_pendulum/figures/noisefree_mseperception_accuracy.png")

# Generate gif of image-based control
π_img = ContinuousNetwork(Chain((x) -> model(reshape(obsfn(x), 360, 1)), (x) -> [action(simple_policy, x)]), 1)
Crux.gif(env, simple_policy, "inverted_pendulum/figures/noisefree_mseperception_episode.gif")

## Noisy images
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0.3))

# Make a gif of the pendulum
frames = [obsfn([θ, 0])[:,:,1] for θ in sin.(0:.1:2π)]
save("inverted_pendulum/figures/noisy_pendulum_images.gif", cat(frames..., dims=3), fps=10)

y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    X[:, :, :, i] = obsfn(y[:, i])
end

scatter(y[1, :], y[2, :])
ps = [plot(obsfn(y[:, i])[:,:,1], xaxis=false, xticks=false, yticks=false, titlefont=font(pointsize=8), title=@sprintf("θ=%0.2f, ω=%0.2f", y[1, i], y[2, i])) for i = 1:10]
plot(ps..., layout=(2, 5))
# savefig("inverted_pendulum/figures/training_data.png")


data = Flux.DataLoader((X, y), batchsize=1000)

model=Chain(flatten, Dense(360, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* [θmax, ωmax])
opt = ADAM(1e-3)
Flux.@epochs 100 Flux.train!(mse_loss(model), Flux.params(model), data, opt)

ŷ = model(X)
scatter(y[2, :], ŷ[2, :], alpha = 0.2, label = "ω", color=2)
scatter!(y[1, :], ŷ[1, :], label="θ", alpha=0.2, xlabel="Ground Truth", ylabel = "Predicted", title = "Perception Model Accuracy", color=1)

savefig("inverted_pendulum/figures/noisy_mseperception_accuracy.png")

# Generate gif of image-based control (noisy)
π_img = ContinuousNetwork(Chain((x) -> model(Float32.(reshape(obsfn(x), 360, 1))), (x) -> [action(simple_policy, x)]), 1)

undiscounted_return(Sampler(env, π_img, max_steps=200), Neps=10)

Crux.gif(env, π_img, "inverted_pendulum/figures/noisy_mseperception_episode.gif")

