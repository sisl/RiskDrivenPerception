using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

# Load the environmetn and policy
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π/4)
# nn_policy = BSON.load("inverted_pendulum/controllers/policy.bson")[:policy]
simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))

ϵθ_ln=Normal(0, 0.2/5)
ϵω_ln=Normal(0, 0.5/5)

ϵθ_nominal=Normal(0, 0.2)
ϵω_nominal=Normal(0, 0.5)

ϵθ_hn=Uniform(-0.8, 0.8)
ϵω_hn=Uniform(-2.0, 2.0)

ϵθ_range = 10 .^ collect(range(log10(0.04), stop=log10(0.8), length=8))
ϵω_range = 10 .^ collect(range(log10(0.1), stop=log10(2.0), length=8))

p1 = plot(ϵθ_range, pdf.(ϵθ_hn, ϵθ_range), label="High Noise", marker=true)
plot!(ϵθ_range, pdf.(ϵθ_ln, ϵθ_range), label="Low Noise", marker=true)
plot!(ϵθ_range, pdf.(ϵθ_nominal, ϵθ_range), label="Nominal Noise", marker=true)

p2 = plot(ϵω_range, pdf.(ϵω_hn, ϵω_range), label="High Noise", marker=true)
plot!(ϵω_range, pdf.(ϵω_ln, ϵω_range), label="Low Noise", marker=true)
plot!(ϵω_range, pdf.(ϵω_nominal, ϵω_range), label="Nominal Noise", marker=true)
plot(p1, p2)
savefig("risk_models.png")

for (ϵθ_model, ϵω_model, folder) in [(ϵθ_ln, ϵω_ln, "low_noise_assumption"), (ϵθ_nominal, ϵω_nominal, "nominal_noise_assumption"), (ϵθ_hn, ϵω_hn, "high_noise_assumption")]

    # Setup the problem and solve for the risk distribution
    rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy, ϵθ=ϵθ_model, ϵω=ϵω_model; ϵθ_range, ϵω_range)
    Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt, cost_points);

    # Create the CVaR convenience Function
    # ECVaR(s, α) = ECVaR(s2pt([0.0, s...]), s_grid, ϵ_grid, Qw, cost_points, px; α)
    # normalized_CVaR(s, ϵ, α; kwargs...) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α, kwargs...)
    CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

    riskmin(x; α) = minimum([CVaR(x, [ϵ1, ϵ2], α)[1] for ϵ1 in ϵ1s, ϵ2 in ϵ2s])
    riskmax(x; α) = maximum([CVaR(x, [ϵ1, ϵ2], α)[1] for ϵ1 in ϵ1s, ϵ2 in ϵ2s])
    mean_risk(x; α) = mean([CVaR(x, [ϵ1, ϵ2], α)[1] for ϵ1 in ϵ1s, ϵ2 in ϵ2s])
    risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)


    plots_relrisk = []
    plots_risk = []
    for α in [-0.9999, -0.999, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999, 0.9999]
        push!(plots_risk, heatmap(θs, ωs, (x, y) -> mean_risk([x,y]; α), clims=(0,0.5), title="α=$(α)"))
        push!(plots_relrisk, heatmap(θs, ωs, (x, y) -> risk_weight([x,y]; α), clims=(0,0.5), title="α=$(α)"))
    end
    plot(plots_risk..., size = (600*3, 400*3))
    savefig("inverted_pendulum/risk_networks/$folder/mean_risk_tables.png")

    plot(plots_relrisk..., size = (600*3, 400*3))
    savefig("inverted_pendulum/risk_networks/$folder/relative_risk_tables.png")


    # Set the desired α
    for α in [-0.999, -0.99, -0.9, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999]
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
        model = Chain(Dense(4, 64, relu),
            Dense(64, 1)) |> gpu
        θ = Flux.params(model)
        opt = ADAM(1e-3)

        # Setup the training
        loss(x, y) = Flux.mse(model(x), y)
        evalcb() = println("train loss: ", loss(X, y))
        throttlecb = Flux.throttle(evalcb, 1.0)

        # Train
        Flux.@epochs 200 Flux.train!(loss, θ, data, opt, cb=throttlecb)

        model = model |> cpu
        BSON.@save "inverted_pendulum/risk_networks/$folder/rn_$(α).bson" model
    end

end

# Load and visualize a model
# α = 0.0
# model = BSON.load("inverted_pendulum/risk_networks/low_noise_assumption/rn_$(α).bson")[:model]

# heatmap(θs, ωs, (x, y) -> CVaR([x, y], [0, 0], α), clims=(0, π/4))x
# heatmap(θs, ωs, (x, y) -> model([x, y, 0, 0])[1], clims=(0, π/4))

# p1 = heatmap(ϵ1s, ϵ2s, (x, y) -> CVaR([0.2, 0], [x, y], α), clims=(0, π/4), xlabel="ϵ₁", ylabel="ϵ₂", title = "Tabular")
# p2 = heatmap(ϵ1s, ϵ2s, (x, y) -> model([0.2, 0, x, y])[1], clims=(0, π/4), xlabel="ϵ₁", ylabel="ϵ₂", title = "Neural Network Surrogate")
# plot(p1, p2, size = (1200, 400), margin=5mm)
# savefig("inverted_pendulum/figures/nn_surrogate.png")

