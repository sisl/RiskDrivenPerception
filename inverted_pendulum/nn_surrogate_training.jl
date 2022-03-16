using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)
Qw = solve_cvar_fixed_particle(rmdp, px.distribution, s_grid, 𝒮, s2pt, cost_points);

ECVaR(s, α) = ECVaR(s2pt([0.0, s...]), s_grid, ϵ_grid, Qw, cost_points, px; α)
normalized_CVaR(s, ϵ, α; kwargs...) = normalized_CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α, kwargs...)
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)


# Set the desired α
α = 0.0

# Generate training data
big_grid = RectangleGrid(θs, ωs, ϵ1s, ϵ2s)

normalizers =  Dict()
X = zeros(4, length(big_grid))
y = zeros(1, length(big_grid))
for (i, (θ, ω, ϵ1, ϵ2)) in enumerate(big_grid)
    s = [θ, ω]
    if !haskey(normalizers, s)
        normalizers[s] = ECVaR(s, α)
    end
        
    (i % 10000) == 0 && println("i=$i")
    X[:, i] .= (θ, ω, ϵ1, ϵ2)
    y[i] = normalized_CVaR(s, [ϵ1, ϵ2], α, normalizer=normalizers[s])
end
data = Flux.DataLoader((X,y), batchsize=length(y), shuffle=true) |> gpu


# Create the model and optimizer
model = Chain(Dense(4, 128, relu), Dense(128, 64, relu), Dense(64, 1, softplus)) |> gpu
θ = Flux.params(model)
opt = ADAM(1e-3)

# Setup the training
loss(x,y) = Flux.mse(model(x), y)
evalcb() = println("train loss: ", loss(X, y))
throttlecb = Flux.throttle(evalcb, 1)

# Train
Flux.@epochs 100 Flux.train!(loss, θ, data, opt, cb=throttlecb)


heatmap(θs, ωs, (x, y) -> model([x, y, 0, 0])[1], clims=(0,π))
heatmap(θs, ωs, (x, y) -> normalized_CVaR([x, y], [0, 0], 0.0), clims=(0,π))

heatmap(ϵ1s, ϵ2s, (x, y) -> model([0.2, 0, x, y])[1], clims=(0,π))
heatmap(ϵ1s, ϵ2s, (x, y) -> normalized_CVaR([0.2, 0], [x, y], 0.0), clims=(0,π))

y

