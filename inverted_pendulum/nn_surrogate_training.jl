using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")
# include("siren.jl")

s0 = [0.0, 0, 0.0]
# Load the environmetn and policy
env = InvertedPendulumMDP(λcost = 0.1f0, failure_thresh = π,
        θ0 = Uniform(s0[2], s0[2] + 1e-6),
        ω0 = Uniform(s0[3], s0[3] + 1e-6))
nn_policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]
simple_policy = FunPolicy(continuous_rule(0, 2.0, -1))

# k2a = 0:1:10
# k3a = -10:1:10

# rmap = reshape([undiscounted_return(Sampler(env, FunPolicy(continuous_rule(.0, 2.0, -1))), Neps = 1) for k2 in k2a for k3 in k3a], length(k2a), :)

# heatmap(k2a, k3a, rmap)

# undiscounted_return(Sampler(env, FunPolicy(continuous_rule(0.0, 2.0, -1))), Neps = 10)
# Crux.gif(env, FunPolicy(continuous_rule(0.0, 10.0, -10)), "test.gif")

# rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)
# Qw = solve_cvar_fixed_particle(rmdp, px.distribution, s_grid, 𝒮, s2pt, cost_points);

θs_small = θs[10:40]
ωs_small = ωs[10:40]

ECVaR(s, α) = ECVaR(s2pt([0.0, s...]), s_grid, ϵ_grid, Qw, cost_points, px; α)
normalized_CVaR(s, ϵ, α; kwargs...) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α, kwargs...)
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)


# Set the desired α
for α in [0.0]

    θs_small = θs[10:40]
    ωs_small = ωs[10:40]

    # Generate training data
    big_grid = RectangleGrid(θs_small, ωs_small, ϵ1s, ϵ2s)

    normalizers = Dict()
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
    data = Flux.DataLoader((X, y), batchsize = 1024, shuffle = true)


    # Create the model and optimizer
    # model = Chain(SirenDense(4, 256, isfirst = true),
    #     SirenDense(256, 256),
    #     SirenDense(256, 256),
    #     # SirenDense(256, 256),
    #     Dense(256, 1, softplus)) |> gpu
    model = Chain(Dense(4, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 1)) |> gpu
    θ = Flux.params(model)
    opt = ADAM(1e-3)

    # Setup the training
    loss(x, y) = Flux.mse(model(x), y)
    evalcb() = println("train loss: ", loss(X, y))
    throttlecb = Flux.throttle(evalcb, 0.1)

    # Train
    Flux.@epochs 50 Flux.train!(loss, θ, data, opt, cb = throttlecb)

    model = model |> cpu
    BSON.@save "inverted_pendulum/risk_networks/rn_$(α).bson" model
end

heatmap(θs_small, ωs_small, (x, y) -> model([x, y, 0, 0])[1], clims = (0, π))
heatmap(θs_small, ωs_small, (x, y) -> CVaR([x, y], [0, 0], 0.6), clims = (0, π))

heatmap(ϵ1s, ϵ2s, (x, y) -> model([0.3, 0, x, y])[1], clims = (0, π))
heatmap(ϵ1s, ϵ2s, (x, y) -> CVaR([0.3, 0], [x, y], 0.0), clims = (0, π))


