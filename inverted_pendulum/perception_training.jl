using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, Measures, BSON, Printf, Zygote
include("../inverted_pendulum/controllers/rule_based.jl")

## Define the perception system
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0.5))

# Range of state variables
θmax = π/4
ωmax = 1.0

scale = [θmax, ωmax] |> gpu

# Generate training images
N = 10000
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    X[:, :, :, i] = obsfn(y[:, i])
end
X = X |> gpu
y = y |> gpu
data = Flux.DataLoader((X, y), batchsize=1024)



# Define the loss functions
mse_loss(model) = (x, y) -> Flux.Losses.mse(model(x), y)

function risk_loss(riskfn; λrisk::Float32, λmse::Float32=1f0)
    (model) -> begin
        (x, y) -> begin
            ŷ = model(x)
            ϵ = ŷ .- y
            ϵ_risk = riskfn([y; ϵ])
            return λmse*Flux.mse(ŷ, y) + λrisk*mean(ϵ_risk)
        end
    end
end

function train_perception(loss, name; 
                          model=Chain(flatten, Dense(360, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* scale),
                          opt=ADAM(1e-3),
                          epochs=400, 
                          data=data)
    model = model |> gpu
    evalcb(model, loss) = () -> println("train loss: ", loss(model)(X, y))
    throttlecb(model, loss) = Flux.throttle(evalcb(model, loss), 1.0)

    Flux.@epochs epochs Flux.train!(loss(model), Flux.params(model), data, opt, cb=throttlecb(model, loss))
    
    model = model |> cpu
    model = Chain(model[1:end-1]..., x -> x .* [θmax, ωmax])
    BSON.@save "$(name)_perception_network.bson" model
    return model
end

function plot_perception_errors(model, name=nothing; X=cpu(X), y=cpu(y))
    ŷ = model(X)
    p1 = scatter(y[1, :], ŷ[1, :], label="θ", alpha=0.2, xlabel="θ", ylabel = "Predicted", title = "Perception Model Accuracy (over θ)")
    scatter!(y[1, :], ŷ[2, :], label="ω", alpha=0.2, legend=:topleft)
    p2 = scatter(y[2, :], ŷ[1, :], alpha = 0.2, label = "θ̂", xlabel="ω", title = "Perception Model Accuracy (over ω)")
    scatter!(y[2, :], ŷ[2, :], alpha = 0.2, label = "ω̂", ylabel = "Predicted", legend=:topleft)
    
    
    p = plot(p1, p2, size=(1200, 400), margin=5mm)
    
    if isnothing(name)
        return p
    else
        savefig("$(name)_perception_error.png")
    end
end

function eval_perception(model, name; Neps=100, obsfn=obsfn, max_steps=1000)
    # Construct the image environment and the control policy
    img_env = ImageInvertedPendulum(λcost=0.0f0, observation_fn=obsfn, θ0=Uniform(-0.1, 0.1), ω0=Uniform(-0.1, 0.1), failure_thresh=π/4)
    simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))
    π_img = ContinuousNetwork(Chain((x) -> model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
    
    # Get the undiscounted return using the combined controller--write to disk
    r = undiscounted_return(Sampler(img_env, π_img, max_steps=max_steps), Neps=Neps)
    open("$(name)_return.txt", "w") do io
         write(io, "$(r) / $(max_steps)")
    end;
    
    # Plot and save the policy map
    heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> action(simple_policy, model(reshape(Float32.(obsfn([θ, ω])), 360, 1))[:]), title="Image Control Policy", xlabel="θ", ylabel="ω")
    savefig("$(name)_policymap.png")
    
    # Construct a gif of one episode
    Crux.gif(img_env, π_img, "$(name)_episode.gif")
    
    return r
end


dir = "inverted_pendulum/results/alpha_lambda_sweeps/"

# Load in the risk network
max_steps = 500
Neps = 200
Nepochs = 400

name = "$(dir)mse"
mse_model = train_perception(mse_loss, name, epochs=Nepochs)
plot_perception_errors(mse_model, name)
eval_perception(mse_model, name, max_steps=max_steps, Neps=Neps)

αs = [-0.8, -0.4, 0.0, 0.4, 0.8]
λs = [0.01f0, 0.1f0, 1f0, 10f0]
returns = zeros(length(αs), length(λs))

for α in αs
    risk_function = BSON.load("inverted_pendulum/risk_networks/rn_$(α).bson")[:model] |> gpu
    for λ in λs
        name = "$(dir)α=$(α)_λ=$(λ)"
        model = train_perception(risk_loss(risk_function, λrisk=λ), name, epochs=Nepochs)
        plot_perception_errors(model, name)
        val = eval_perception(model, name, max_steps=max_steps, Neps=Neps)
        returns[findfirst(αs .== α), findfirst(λs .== λ)] = val
    end
end

BSON.@save "inverted_pendulum/results/alpha_lambda_sweeps/returns.bson" returns

heatmap(αs, λs, (α, λ) -> returns[findfirst(αs .== α), findfirst(λs .== λ)], yscale=:log10, xlabel="α", ylabel="λ", title="Returns")
savefig("inverted_pendulum/results/alpha_lambda_sweeps/returns_heatmap.pdf")

