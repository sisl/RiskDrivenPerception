using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions
using Plots, Measures, BSON, Printf, Zygote, GridInterpolations
include("../src/risk_solvers.jl")
include("../inverted_pendulum/controllers/rule_based.jl")
include("problem_setup.jl")

# Solve for the risk function
env = InvertedPendulumMDP(λcost=0.1f0, failure_thresh=π / 4)
simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))
rmdp, px, θs, ωs, s_grid, 𝒮, s2pt, cost_points, ϵ1s, ϵ2s, ϵ_grid = rmdp_pendulum_setup(env, simple_policy)
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt, cost_points);
CVaR(s, ϵ, α) = CVaR(s2pt([0.0, s...]), ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

## Define the perception system
obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05)#, noise=Normal(0, 0.1))

# Range of state variables
θmax = π / 4
ωmax = 1.0

scale = [θmax, ωmax] #|> gpu

# Generate training images
N = 50 #100 #10000
y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
for i = 1:N
    y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    X[:, :, :, i] = obsfn(y[:, i])
end
# X = X |> gpu
# y = y |> gpu

# Generate eval images
yeval, Xeval = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
for i = 1:N
    yeval[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
    Xeval[:, :, :, i] = obsfn(y[:, i])
end

# Define the loss functions
mse_loss(model) = (x, y) -> Flux.Losses.mse(model(x), y)

riskmax(x; α) = maximum([CVaR(x, [ϵ1, ϵ2], α)[1] for ϵ1 in ϵ1s, ϵ2 in ϵ2s])
risk_weight(x; α) = riskmax(x; α) - CVaR(x, [0.0, 0.0], α)[1]

# Compute the weights for the training data
# w = [risk_weight(y[:, i], α = 0) for i = 1:size(y, 2)]'
# w = w |> gpu

# Generate biased data
function rejection_sample_data(N; baseline=0.5, α=0.0)
    y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
    ind = 1
    while ind ≤ N
        ysamp = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
        if rand() < risk_weight(ysamp, α=α) + baseline
            y[:, ind] = ysamp
            X[:, :, :, ind] = obsfn(ysamp)
            ind += 1
        end
        ind % 100 == 0 ? println(ind) : nothing
    end
    return y, X
end
yr, Xr = rejection_sample_data(N, baseline=0.2)
yreval, Xreval = rejection_sample_data(N, baseline=0.2)
scatter(yr[1, :], yr[2, :])

data = Flux.DataLoader((X, y), batchsize=10)
data_r = Flux.DataLoader((Xr, yr), batchsize=10)

heatmap(-θmax:0.01:θmax, -ωmax:0.01:ωmax, (θ, ω) -> risk_weight([θ, ω], α=0.0)[1])

# function weighted_mse_loss(model)
#     (x, y, w) -> begin
#         ŷ = model(x)
#         ϵ = ŷ .- y
#         return mean((ϵ .^ 2) .* w)
#     end
# end

function train_perception(loss;
    model=Chain(flatten, Dense(360, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* scale),
    opt=ADAM(1e-3),
    epochs=400,
    data=data,
    Xeval=X,
    yeval=y)
    model = model #|> gpu
    evalcb(model, loss) = () -> println("train loss: ", loss(model)(Xeval, yeval))
    throttlecb(model, loss) = Flux.throttle(evalcb(model, loss), 1.0)

    Flux.@epochs epochs Flux.train!(loss(model), Flux.params(model), data, opt)#, cb=throttlecb(model, loss))

    model = model #|> cpu
    model = Chain(model[1:end-1]..., x -> x .* [θmax, ωmax])
    #BSON.@save "$(name)_perception_network.bson" model
    return model
end

function plot_perception_errors(model, name=nothing; X=cpu(X), y=cpu(y))
    ŷ = model(X)
    p1 = scatter(y[1, :], ŷ[1, :], label="θ", alpha=0.2, xlabel="θ", ylabel="Predicted", title="Perception Model Accuracy (over θ)")
    scatter!(y[1, :], ŷ[2, :], label="ω", alpha=0.2, legend=:topleft)
    p2 = scatter(y[2, :], ŷ[1, :], alpha=0.2, label="θ̂", xlabel="ω", title="Perception Model Accuracy (over ω)")
    scatter!(y[2, :], ŷ[2, :], alpha=0.2, label="ω̂", ylabel="Predicted", legend=:topleft)


    p = plot(p1, p2, size=(1200, 400), margin=5mm)

    if isnothing(name)
        return p
    else
        savefig("$(name)_perception_error.png")
    end
end

# Train stuff and make comparisons
img_env = ImageInvertedPendulum(λcost=0.0f0, observation_fn=obsfn, θ0=Uniform(-0.1, 0.1), ω0=Uniform(-0.1, 0.1), failure_thresh=π / 4)

# MSE baseline
mse_model = train_perception(mse_loss, Xeval=Xeval, yeval=yeval, epochs=100)

π_img_mse = ContinuousNetwork(Chain((x) -> mse_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
plot_perception_errors(mse_model)
plot_perception_errors(mse_model, X=Xeval, y=yeval)
heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> action(simple_policy, mse_model(reshape(Float32.(obsfn([θ, ω])), 360, 1))[:]),
    title="Image Control Policy", xlabel="θ", ylabel="ω")
undiscounted_return(Sampler(img_env, π_img_mse, max_steps=500), Neps=100)
Crux.gif(img_env, π_img_mse, "out_mse.gif", max_steps=200, Neps=1)

# weighted_mse_model = train_perception(weighted_mse_loss)
# plot_perception_errors(weighted_mse_model)

# π_img_wmse = ContinuousNetwork(Chain((x) -> weighted_mse_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
# undiscounted_return(Sampler(img_env, π_img_wmse, max_steps=500), Neps=20)
# heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> action(simple_policy, weighted_mse_model(reshape(Float32.(obsfn([θ, ω])), 360, 1))[:]), title="Image Control Policy", xlabel="θ", ylabel="ω")
# Crux.gif(img_env, π_img_wmse, "out.gif", max_steps=200, Neps =1)

# Rejection sampling MSE model
rmse_model = train_perception(mse_loss, data=data_r, Xeval=Xreval, yeval=yreval, epochs=100)

π_img_rmse = ContinuousNetwork(Chain((x) -> rmse_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
plot_perception_errors(rmse_model)
plot_perception_errors(rmse_model, X=Xreval, y=yreval)
heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> action(simple_policy, rmse_model(reshape(Float32.(obsfn([θ, ω])), 360, 1))[:]),
    title="Image Control Policy", xlabel="θ", ylabel="ω")
undiscounted_return(Sampler(img_env, π_img_rmse, max_steps=500), Neps=100)
Crux.gif(img_env, π_img_rmse, "out_rmse.gif", max_steps=200, Neps=1)

function get_avg_error(x, model; N=10)
    sq_errs = [sum((model(reshape(Float32.(obsfn(x)), 360, 1)) - x) .^ 2) for _ in 1:N]
    return mean(sq_errs) / 2
end

heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> get_avg_error([θ, ω], mse_model, N=10),
    title="Average mean squared error", xlabel="θ", ylabel="ω", clims=(0, 1.3))
heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> get_avg_error([θ, ω], rmse_model, N=10),
    title="Average mean squared error", xlabel="θ", ylabel="ω", clims=(0, 1.3))

############################################################
# Trials
############################################################
function get_uniform_data(npoints)
    y, X = zeros(Float32, 2, npoints), zeros(Float32, 18, 10, 2, npoints)
    for i = 1:npoints
        y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
        X[:, :, :, i] = obsfn(y[:, i])
    end
    return X, y
end

function get_risk_data(npoints; baseline=0.2, α=0.0)
    y, X = rejection_sample_data(npoints, baseline=baseline, α=α)
    return X, y
end

function run_trials(ntrials; npoints=100, baseline=0.2, α=0.0)
    baseline_returns = zeros(ntrials)
    risk_returns = zeros(ntrials)

    for i = 1:ntrials
        println("trial: ", i)

        println("Generating baseline data...")
        X, y = get_uniform_data(npoints)
        baseline_data = Flux.DataLoader((X, y), batchsize=10)
        println("Training baseline network...")
        baseline_model = train_perception(mse_loss, Xeval=Xeval, yeval=yeval, epochs=100, data=baseline_data)
        π_baseline = ContinuousNetwork(Chain((x) -> baseline_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
        println("Evaluating baseline model...")
        baseline_returns[i] = undiscounted_return(Sampler(img_env, π_baseline, max_steps=500), Neps=100)

        println("Generating risk data...")
        Xr, yr = get_risk_data(npoints, baseline=baseline, α=α)
        risk_data = Flux.DataLoader((Xr, yr), batchsize=10)
        println("Training risk network...")
        risk_model = train_perception(mse_loss, Xeval=Xeval, yeval=yeval, epochs=100, data=risk_data)
        π_risk = ContinuousNetwork(Chain((x) -> risk_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
        println("Evaluating risk model...")
        risk_returns[i] = undiscounted_return(Sampler(img_env, π_risk, max_steps=500), Neps=100)
    end

    return baseline_returns, risk_returns
end

baseline_returns, risk_returns = run_trials(5, npoints=50, baseline=0.05, α=0.5)

function run_α_trials(ntrials, αs; npoints=100, baseline=0.2)
    nα = length(αs)
    baseline_returns = zeros(ntrials)
    risk_returns = zeros(nα, ntrials)

    for i = 1:ntrials
        println("Generating baseline data...")
        X, y = get_uniform_data(npoints)
        baseline_data = Flux.DataLoader((X, y), batchsize=10)
        println("Training baseline network...")
        baseline_model = train_perception(mse_loss, Xeval=Xeval, yeval=yeval, epochs=100, data=baseline_data)
        π_baseline = ContinuousNetwork(Chain((x) -> baseline_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
        println("Evaluating baseline model...")
        baseline_returns[i] = undiscounted_return(Sampler(img_env, π_baseline, max_steps=500), Neps=100)
    end

    for i = 1:nα
        println("α: ", αs[i])
        for j = 1:ntrials
            println("trial: ", i)

            println("Generating risk data...")
            Xr, yr = get_risk_data(npoints, baseline=baseline, α=αs[i])
            risk_data = Flux.DataLoader((Xr, yr), batchsize=10)
            println("Training risk network...")
            risk_model = train_perception(mse_loss, Xeval=Xeval, yeval=yeval, epochs=100, data=risk_data)
            π_risk = ContinuousNetwork(Chain((x) -> risk_model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
            println("Evaluating risk model...")
            risk_returns[i, j] = undiscounted_return(Sampler(img_env, π_risk, max_steps=500), Neps=100)
        end
    end

    return baseline_returns, risk_returns
end

baseline_returns, risk_returns = run_α_trials(5, [0.0, 0.2, 0.5, 0.8, 0.99], npoints=50, baseline=0.05)