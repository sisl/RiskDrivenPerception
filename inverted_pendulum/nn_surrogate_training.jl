using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

# Load the environmetn and policy
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π)
# nn_policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]
simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))

# Setup the problem and solve for the risk distribution
rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)
Uw, Qw = solve_cvar_fixed_particle(rmdp, px.distribution, s_grid, 𝒮, s2pt, cost_points);

# Create the CVaR convenience Function
# ECVaR(s, α) = ECVaR(s2pt([0.0, s...]), s_grid, ϵ_grid, Qw, cost_points, px; α)
# normalized_CVaR(s, ϵ, α; kwargs...) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α, kwargs...)
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)


# Set the desired α
for α in [-0.8, -0.4, 0.0, 0.4, 0.8]
    # Generate training data
    big_grid = RectangleGrid(θs, ωs, ϵ1s, ϵ2s)

    # normalizers = Dict()
    X = zeros(4, length(big_grid))
    y = zeros(1, length(big_grid))
    for (i, (θ, ω, ϵ1, ϵ2)) in enumerate(big_grid)
        s = [θ, ω]
        # if !haskey(normalizers, s)
        #     normalizers[s] = ECVaR(s, α)
        # end

        (i % 10000) == 0 && println("i=$i")
        X[:, i] .= (θ, ω, ϵ1, ϵ2)
        # y[i] = normalized_CVaR(s, [ϵ1, ϵ2], α, normalizer = normalizers[s])
        y[i] = CVaR(s, [ϵ1, ϵ2], α)
    end
    X = X |> gpu
    y = y |> gpu
    data = Flux.DataLoader((X, y), batchsize=1024, shuffle=true)


    # Create the model and optimizer
    model = Chain(Dense(4, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 64, relu),
        Dense(64, 1)) |> gpu
    θ = Flux.params(model)
    opt = ADAM(1e-3)

    # Setup the training
    loss(x, y) = Flux.mse(model(x), y)
    evalcb() = println("train loss: ", loss(X, y))
    throttlecb = Flux.throttle(evalcb, 1.0)

    # Train
    Flux.@epochs 400 Flux.train!(loss, θ, data, opt, cb=throttlecb)

    model = model |> cpu
    BSON.@save "inverted_pendulum/risk_networks/rn_$(α).bson" model
end

# Load and visualize a model
α = -0.4
model = BSON.load("inverted_pendulum/risk_networks/rn_$(α).bson")[:model]

heatmap(θs, ωs, (x, y) -> CVaR([x, y], [0, 0], α), clims=(0, π))
heatmap(θs, ωs, (x, y) -> model([x, y, 0, 0])[1], clims=(0, π))


p1 = heatmap(ϵ1s, ϵ2s, (x, y) -> CVaR([0.2, 0], [x, y], α), clims=(0, π), xlabel="ϵ₁", ylabel="ϵ₂", title = "Tabular")
p2 = heatmap(ϵ1s, ϵ2s, (x, y) -> model([0.2, 0, x, y])[1], clims=(0, π), xlabel="ϵ₁", ylabel="ϵ₂", title = "Neural Network Surrogate")
plot(p1, p2, size = (1200, 400))
savefig("inverted_pendulum/figures/nn_surrogate.png")

