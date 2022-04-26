using PyCall, ProgressBars
using BSON
using BSON: @save
using CSV, DataFrames
using Distributions
using LinearAlgebra
using Plots
using Flux
using Flux: update!, DataLoader

const HNMAC = 100

# Imports for calling model
if pyimport("sys")."path"[1] != "collision_avoidance/"
    pushfirst!(pyimport("sys")."path", "collision_avoidance/")
end
xplane_ctrl = pyimport("util")
model = xplane_ctrl.load_model("collision_avoidance/models/traffic_detector_v3.pt")

# Get the detections for all of the images
# detects = zeros(10000)
# for i = ProgressBar(0:9999)
#     filename = "collision_avoidance/data_files/traffic_data/imgs/$(i).jpg"
#     detects[i+1], _, _, _, _ = xplane_ctrl.bb_from_file(model, filename)
# end

# @save "collision_avoidance/data_files/detections_v3.bson" detects

# Load in the state data
data_file = "collision_avoidance/data_files/traffic_data/state_data.csv"
df = DataFrame(CSV.File(data_file))
df[1, :]

# Calculate corresponding h and τ
function mdp_state(df; v_dist=Uniform(45, 55), θ_dist=Uniform(120, 240))
    h = df.n0 - df.n1

    v0 = rand(v_dist)
    v1 = rand(v_dist)
    θ = rand(θ_dist)

    # println("v0: ", v0)
    # println("v1: ", v1)
    # println("θ: ", θ)

    r0 = [df.e0, df.n0]
    r1 = [df.e1, df.n1]
    r = norm(r0 - r1)

    dt = 1.0
    r0_next = v0 * dt * [-sind(0), cosd(0)]

    r1_new = r * [sind(180 - θ), cosd(180 - θ)]
    r1_next = r1_new + v1 * dt * [-sind(θ), cosd(θ)]

    r = norm(r0 - r1)
    r_next = norm(r0_next - r1_next)

    # println("θ: ", θ)
    # println("r: ", r)

    # println("r0_next: ", r0_next)

    # println("r1_new: ", r1_new)
    # println("r1_next: ", r1_next)

    ṙ = (r - r_next) / dt

    τ = r < HNMAC ? 0 : (r - HNMAC) / ṙ
    if τ < 0
        τ = Inf
    end

    h, τ
end

hs = zeros(10000)
τs = zeros(10000)
for i = 1:10000
    hs[i], τs[i] = mdp_state(df[i, :])
end

histogram(hs, xlabel='h', legend=false)
histogram(τs, xlabel='τ', legend=false)

# Train logistic regression model
X = vcat(hs', τs')
y = reshape(detects, 1, :)

# Parameters
batch_size = 512
nepoch = 250
lr = 1e-3
data = DataLoader((X, y), batchsize=batch_size, shuffle=true, partial=false)
m = Dense(2, 1, sigmoid)
θ = Flux.params(m)
opt = ADAM(lr)

for e = 1:nepoch
    for (x, y) in data
        _, back = Flux.pullback(() -> Flux.mse(m(x), y), θ)
        update!(opt, θ, back(1.0f0))
    end
    loss_train = Flux.mse(m(X), y)
    println("Epoch: ", e, " Loss Train: ", loss_train)
end

heatmap(0:0.1:40, -500:10:500, (x, y) -> m([y, x])[1], xlabel='τ', ylabel='h')
Flux.params(m)

plot(0:0.1:40, (x)->sigmoid(-0.3232x + 3.2294), xlabel="τ", ylabel="probability of detection", legend=false)