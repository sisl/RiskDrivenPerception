using POMDPs, POMDPGym, Crux, Flux, Colors, Distributions, Plots, Measures, BSON, Printf, Zygote
include("../inverted_pendulum/controllers/rule_based.jl")

## Define the perception system
obsfn = (s) -> Float32.(POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0.5)))

# Range of state variables
θmax = π/4
ωmax = 1.0

scalevec = [θmax, ωmax] |> gpu

# Generate training images
function gen_data(N=10000, θmax=θmax, ωmax=ωmax)
    y, X = zeros(Float32, 2, N), zeros(Float32, 18, 10, 2, N)
    for i = 1:N
        y[:, i] = [rand(Uniform(-θmax, θmax)), rand(Uniform(-ωmax, ωmax))]
        X[:, :, :, i] = obsfn(y[:, i])
    end
    X = X |> gpu
    y = y |> gpu
    Flux.DataLoader((X, y), batchsize=1024)
end

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
                          model=Chain(Flux.flatten, Dense(360, 64, relu), Dense(64, 64, relu), Dense(64, 2, tanh), x -> x .* scalevec),
                          opt=ADAM(1e-3),
                          epochs=400, 
                          data,
                          write2disk=true)
    model = model |> gpu
    evalcb(model, loss) = () -> println("train loss: ", loss(model)(data.data[1], data.data[2]))
    throttlecb(model, loss) = Flux.throttle(evalcb(model, loss), 1.0)

    Flux.@epochs epochs Flux.train!(loss(model), Flux.params(model), data, opt, cb=throttlecb(model, loss))
    
    model = model |> cpu
    model = Chain(model[1:end-1]..., x -> x .* [θmax, ωmax])
    write2disk && BSON.@save "$(name)_perception_network.bson" model
    return model
end

function plot_perception_errors(model, name=nothing; data)
    X = data.data[1]
    y = data.data[2]
    
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

function eval_perception(model, name; Neps=100, obsfn=obsfn, max_steps=1000, testloss=nothing, riskloss=nothing, write2disk=true)
    # Construct the image environment and the control policy
    # img_env = ImageInvertedPendulum(λcost=0.0f0, observation_fn=obsfn, θ0=Uniform(-0.1, 0.1), ω0=Uniform(-0.1, 0.1), failure_thresh=π/4)
    img_env = InvertedPendulumMDP(λcost=0.0f0, θ0=Uniform(-0.1, 0.1), ω0=Uniform(-0.1, 0.1), failure_thresh=π/4)
    simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))
    # π_img = ContinuousNetwork(Chain((x) -> model(reshape(x, 360, 1)), (x) -> [action(simple_policy, x)]), 1)
    π_img = ContinuousNetwork(Chain((x) -> model(reshape(obsfn(x), 360, 1)), (x) -> [action(simple_policy, x)]), 1)
    
    # Get the undiscounted return using the combined controller--write to disk
    # r = undiscounted_return(Sampler(img_env, π_img, max_steps=max_steps), Neps=Neps)
    
    s = Sampler(img_env, π_img, max_steps=max_steps)
    data = episodes!(s, Neps=Neps)
    
    r = sum(data[:r]) / Neps
    a = mean(abs.(data[:a]))
    θ = mean(abs.(data[:s][1, :]))
    nominal_actions = [action(simple_policy, data[:s][:, i]) for i = 1:length(data[:r])]
    conservatism = mean( abs.(data[:a]) ./ abs.(nominal_actions))
    
    output = Dict(:return => r, :actions => a, :theta => θ, :conservatism => conservatism)
    if !isnothing(testloss)
        tloss = testloss(model)
        output[:testloss] = tloss
    end
    if !isnothing(riskloss)
        rloss = riskloss(model)
        output[:riskloss] = rloss
    end
    
    if write2disk
        open("$(name)_return.txt", "w") do io
            write(io, "return: $(r) / $(max_steps)\\n")
            write(io, "action: $(a)\\n")
            write(io, "avg angle: $(θ)\\n")
            write(io, "conservatism (a/theta): $(conservatism)\\n")
            !isnothing(testloss) && write(io, "\nTest loss: $tloss")
            !isnothing(riskloss) && write(io, "\nRisk loss: $rloss")
        end
    end
    
    # Plot and save the policy map
    write2disk && heatmap(-θmax:0.05:θmax, -ωmax:0.05:ωmax, (θ, ω) -> action(simple_policy, model(reshape(Float32.(obsfn([θ, ω])), 360, 1))[:]), title="Image Control Policy", xlabel="θ", ylabel="ω")
    write2disk && savefig("$(name)_policymap.png")
    
    # Construct a gif of one episode
    write2disk && Crux.gif(img_env, π_img, "$(name)_episode.gif")
    
    return output
end

function eval_perception_return(model; Neps=100, obsfn=obsfn, max_steps=500)
    # Construct the image environment and the control policy
    img_env = InvertedPendulumMDP(λcost=0.0f0, θ0=Uniform(-0.1, 0.1), ω0=Uniform(-0.1, 0.1), failure_thresh=π/4)
    simple_policy = FunPolicy(continuous_rule(0.0, 2.0, -1))
    π_img = ContinuousNetwork(Chain((x) -> model(reshape(obsfn(x), 360, 1)), (x) -> [action(simple_policy, x)]), 1)

    # Get the undiscounted return using the combined controller
    r = undiscounted_return(Sampler(img_env, π_img, max_steps=max_steps), Neps=Neps)

    return r
end

################################################
# Trials
################################################
function run_trials(ntrials; max_steps=500, Neps=50, Nepochs=400, λ=1f0, α=0)
    baseline_returns = zeros(ntrials)
    risk_returns = zeros(ntrials)

    risk_function = BSON.load("inverted_pendulum/risk_networks/$noise_model/rn_$(α).bson")[:model] |> gpu
    datas = [gen_data(10000) for i = 1:5]

    for (i, data) in enumerate(datas)
        println("Trial: ", i)
    
        # Train
        baseline_model = train_perception(risk_loss(risk_function, λrisk=λ), "baseline", epochs=Nepochs, data=data, write2disk=false)
        risk_model = train_perception(risk_loss(risk_function, λrisk=λ), "risk", epochs=Nepochs, data=data, write2disk=false)
    
        # Eval
        println("evaluating baseline model...")
        baseline_returns[i] = eval_perception_return(baseline_model, Neps=Neps, max_steps=max_steps)
        println("evaluating risk model...")
        risk_returns[i] = eval_perception_return(risk_model, Neps=Neps, max_steps=max_steps)
    end
    return baseline_returns, risk_returns
end

function run_α_trials(ntrials, αs; max_steps=500, Neps=50, Nepochs=400, λ=1.0f0, α=0)
    nα = length(αs)
    baseline_returns = zeros(ntrials)
    risk_returns = zeros(nα, ntrials)

    datas = [gen_data(10000) for i = 1:ntrials]

    # Baseline
    for (i, data) in enumerate(datas)
        # Train
        baseline_model = train_perception(mse_loss, "baseline", epochs=Nepochs, data=data, write2disk=false)
        # Eval
        println("evaluating baseline model...")
        baseline_returns[i] = eval_perception_return(baseline_model, Neps=Neps, max_steps=max_steps)
    end

    for i = 1:nα
        println("α: ", αs[i])
        risk_function = BSON.load("inverted_pendulum/risk_networks/nominal_noise_assumption/rn_$(αs[i]).bson")[:model] |> gpu
        for (j, data) in enumerate(datas)
            println("Trial: ", j)

            # Train
            risk_model = train_perception(risk_loss(risk_function, λrisk=λ), "risk", epochs=Nepochs, data=data, write2disk=false)

            # Eval
            println("evaluating risk model...")
            risk_returns[i, j] = eval_perception_return(risk_model, Neps=Neps, max_steps=max_steps)
        end
    end
    return baseline_returns, risk_returns
end

baseline_returns, risk_returns = run_α_trials(5, [0.0, 0.2, 0.5, 0.8, 0.99], Neps=100)

datas = [gen_data(10000) for i=1:5]
max_steps = 500
Neps = 50
Nepochs = 400

noise_models = ["low_noise_assumption", "nominal_noise_assumption", "high_noise_assumption"]
for noise_model in noise_models
    dir = "inverted_pendulum/results/alpha_comparison/$noise_model/"

    # Load in the risk network
    name = "$(dir)mse"
    # mse_model = train_perception(mse_loss, name, epochs=Nepochs, data=data)
    # plot_perception_errors(mse_model, name; data)
    # eval_perception(mse_model, name, max_steps=max_steps, Neps=Neps)

    αs = [-0.999, -0.9, -0.5, -0.2, 0, 0.2, 0.5, 0.9, 0.999]
    λs = [0.1f0, 1f0]
    outputs = Dict()

    for α in αs
        risk_function = BSON.load("inverted_pendulum/risk_networks/$noise_model/rn_$(α).bson")[:model] |> gpu
        for λ in λs
            ress = []
            for data in datas
                name = "$(dir)α=$(α)_λ=$(λ)"
                model = train_perception(risk_loss(risk_function, λrisk=λ), name, epochs=Nepochs, data=data, write2disk=false)
                # plot_perception_errors(model, name; data)
                # model = BSON.load("$(name)_perception_network.bson")[:model]
                println("evaluating α=$(α), λ=$(λ)")
                res = eval_perception(model, name, max_steps=max_steps, Neps=Neps, write2disk=false)
                push!(ress, res)
            end
            outputs[(α, λ)] = ress
            # returns[findfirst(αs .== α), findfirst(λs .== λ)] = res
        end
    end

    BSON.@save "inverted_pendulum/results/alpha_comparison/$noise_model/outputs.bson" outputs
end

hn_outputs = BSON.load("inverted_pendulum/results/alpha_comparison/high_noise_assumption/outputs.bson")[:outputs]
ln_outputs = BSON.load("inverted_pendulum/results/alpha_comparison/low_noise_assumption/outputs.bson")[:outputs]
nn_outputs = BSON.load("inverted_pendulum/results/alpha_comparison/nominal_noise_assumption/outputs.bson")[:outputs]

αs = [-0.999,  -0.9,  -0.5, -0.2, 0, 0.2, 0.5, 0.9, 0.999]
λ = 1f0

get_array(dict, key, i; λ=λ) = [dict[(α, λ)][i][key] for α in αs]

get_mean(dict, key, N=5; λ=λ) = mean([get_array(dict, key, i, λ=λ) for i=1:5])
get_std(dict, key, N=5; λ=λ) = std([get_array(dict, key, i, λ=λ) for i=1:5])

# plot(αs, get_mean(hn_outputs, :return, λ=0.1f0), marker = true, label="High Noise Assumption - λ=0.1", legend=:bottomright, title="Safety")
plot(αs, get_mean(hn_outputs, :return, λ=1.0f0), ribbon=get_std(hn_outputs, :return, λ=1.0f0), marker = true, label="High Noise Assumption - λ=1.0", legend=:bottomright, title="Safety")

# plot(αs, get_mean(ln_outputs, :return, λ=0.1f0), marker = true, label="Low Noise Assumption - λ=0.1", legend=:bottomright, title="Safety")
plot(αs, get_mean(ln_outputs, :return, λ=1.0f0), ribbon=get_std(ln_outputs, :return, λ=1.0f0), marker = true, label="Low Noise Assumption - λ=1.0", legend=:bottomright, title="Safety")

# plot(αs, get_mean(nn_outputs, :return, λ=0.1f0), marker = true, label="Nominal Noise Assumption - λ=0.1", legend=:bottomright, title="Safety")
plot(αs, get_mean(nn_outputs, :return, λ=1.0f0), ribbon=get_std(nn_outputs, :return, λ=1.0f0), marker = true, label="Nominal Noise Assumption - λ=1.0", legend=:bottomright, title="Safety")


heatmap(αs, λs, (α, λ) -> returns[findfirst(αs .== α), findfirst(λs .== λ)], yscale=:log10, xlabel="α", ylabel="λ", title="Returns")
savefig("inverted_pendulum/results/noise_0.5/alpha_lambda_sweeps/returns_heatmap.png")

## Get some different MSE perception systems
dir = "inverted_pendulum/results/random_perception/"

# Load in the risk network
max_steps = 100
Neps = 200
Nepochs = 100
α = 0.0
testdata = gen_data(1000)
testeval(model) = mse_loss(model)(first(testdata)...)


obsfn = (s) -> POMDPGym.simple_render_pendulum(s, dt=0.05, noise=Normal(0, 0))

outputs = Dict(α => [] for α in αs)
for i=1:20
    data = gen_data(10)

    name = "$(dir)mse_$i"
    mse_model = train_perception(mse_loss, name, epochs=Nepochs, data=data, write2disk=true)
    plot_perception_errors(mse_model, name)
    for α in αs
        riskfn = BSON.load("inverted_pendulum/risk_networks/rn_$(α).bson")[:model] |> gpu
        riskeval(model) = risk_loss(riskfn, λrisk=1f0, λmse=0f0)(model)(first(testdata)...)
        res = eval_perception(mse_model, name, max_steps=max_steps, Neps=Neps, testloss=testeval, riskloss=riskeval, write2disk=false)
        push!(outputs[α], res)
        println("α = $α is done")
    end
end

BSON.@save "inverted_pendulum/results/random_perception/all_results.bson" outputs

returns = [v[:return] for v in outputs[0.0]]
testlosses = [v[:testloss] for v in outputs[0.0]]


normalize(v) = (v .- mean(v)) ./ std(v)
p1 = scatter(testlosses, returns, title = "Test Loss vs. Return", label="", ylabel="Return", xlabel="MSE Test Loss")
p2 = scatter([v[:riskloss] for v in outputs[0.8]], returns, label="", title = "Risk (α=0.8) vs. Return", ylabel="Return", xlabel="Mean Risk")

plot(p1, p2, size=(1200, 400), margin=5mm)
savefig("inverted_pendulum/figures/predicting_return_without_sim.png")

