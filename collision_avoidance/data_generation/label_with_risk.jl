# Calculate Risk
using POMDPs, POMDPGym, Crux, Flux, Distributions, BSON, GridInterpolations
using DataFrames, LinearAlgebra, CSV
using StatsBase
include("../../src/risk_solvers.jl")

s2pt(s) = s

function get_CVaR_items()
    # Load the environment and policy
    println("Loading environment and getting optimal policy...")
    s0 = [0.0, 0.0, 0.0, 40.0]
    env = DetectAndAvoidMDP(h0_dist=Uniform(s0[1] - 1e-16, s0[1] + 1e-16),
        dh0_dist=Uniform(s0[2] - 1e-16, s0[2] + 1e-16), ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
        actions=[-8.0, 0.0, 8.0])

    hmax = 300
    hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
    hs = [-hs_half[1:end-1]; reverse(hs_half)]
    dhs = range(-10, 10, length=21)
    τs = range(0, 40, length=41)

    policy = OptimalDetectAndAvoidPolicy(env, hs, dhs, τs)

    # Set up the cost function and risk mdp
    println("Setting up cost function and solving for risk...")
    costfn(m, s, sp) = isterminal(m, sp) ? 150 - abs(s[1]) : 0.0
    rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

    detect_model = BSON.load("collision_avoidance/models/nominal_error_model.bson")[:m]
    p_detect(s) = detect_model([abs(s[1]), s[4]])[1] # sigmoid(-0.006518117 * abs(s[1]) - 0.10433467s[4] + 1.2849158)
    function get_detect_dist(s)
        pd = p_detect(s)
        noises = [[ϵ, 0.0, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
        return ObjectCategorical(noises, [1 - pd, pd])
    end

    noises_detect = [0, 1]

    ϵ_grid = RectangleGrid(noises_detect)
    noises = [[ϵ[1], 0.0, 0.0, 0.0, 0.0] for ϵ in ϵ_grid]

    px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

    # Set up cost points, state grid, and other necessary data
    cost_points = collect(range(0, 150, 50))
    s_grid = RectangleGrid(hs, dhs, env.actions, τs)
    𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs]

    # Solve for distribution over costs
    @time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
        cost_points, mdp_type=:exp)

    return s_grid, ϵ_grid, Qw, cost_points, px, policy
end

s_grid, ϵ_grid, Qw, cost_points, px, policy = get_CVaR_items()
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

function get_latent_dists(px, policy)
    # Get nominal distribution of dh and a_prev
    env = DetectAndAvoidMDP(ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
        actions=[-8.0, 0.0, 8.0])

    costfn(m, s, sp) = isterminal(m, sp) ? 150 - abs(s[1]) : 0.0
    rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

    N = 10000
    D = episodes!(Sampler(rmdp, px), Neps=N)

    dh_samps = D[:s][2, :]
    a_prev_samps = D[:s][3, :]

    function get_counts(samps, bins)
        counts = zeros(length(bins) - 1)
        for i = 1:length(bins)-1
            counts[i] = length(findall(bins[i] .≤ samps .< bins[i+1]))
        end
        return counts
    end

    dh_counts = get_counts(dh_samps, collect(-10.5:1:10.5))
    wdh = dh_counts ./ sum(dh_counts)
    a_prev_counts = get_counts(a_prev_samps, collect(-12:8:12))
    wa_prev = a_prev_counts ./ sum(a_prev_counts)

    return Categorical(wdh), Categorical(wa_prev)
end

dh_dist, a_prev_dist = get_latent_dists(px, policy)

# Start labeling the data
state_data_file = "/scratch/smkatz/yolo_data/uniform_data_v3/state_data.csv"
df = DataFrame(CSV.File(state_data_file))

dhs = range(-10, 10, length=21)
a_prevs = [-8.0, 0.0, 8.0]

const HNMAC = 100

function mdp_state(e0, n0, u0, e1, n1, u1; v_dist=Uniform(45, 55), θ_dist=Uniform(120, 240))
    h = u0 - u1

    v0 = rand(v_dist)
    v1 = rand(v_dist)
    θ = rand(θ_dist)

    r0 = [e0, n0]
    r1 = [e1, n1]
    r = norm(r0 - r1)

    dt = 1.0
    r0_next = v0 * dt * [-sind(0), cosd(0)]

    r1_new = r * [sind(180 - θ), cosd(180 - θ)]
    r1_next = r1_new + v1 * dt * [-sind(θ), cosd(θ)]

    r = norm(r0 - r1)
    r_next = norm(r0_next - r1_next)

    ṙ = (r - r_next) / dt

    τ = r < HNMAC ? 0 : (r - HNMAC) / ṙ
    if τ < 0
        τ = Inf
    end

    dh = dhs[rand(dh_dist)]
    a_prev = a_prevs[rand(a_prev_dist)]

    [h, dh, a_prev, τ]
end

# Loop through files
for i = 1:9500
    # Get the mdp state
    s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
    # Get the detect risk
    detect_risk = round(CVaR(s, [1], 0.0), digits=6)
    # Get the no detect risk
    no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
    # Get name of text file
    fn = df[i, "filename"]
    text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v3_rl/train/labels/$(fn).txt"
    # Write the risks to it
    io = open(text_file_name, "r")
    temp = read(io, String)
    close(io)

    new_string = "$(temp[1:end-1]) $(detect_risk) $(no_detect_risk)"

    io = open(text_file_name, "w")
    write(io, new_string)
    close(io)
end

# Loop through files
for i = 9501:10000
    # Get the mdp state
    s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
    # Get the detect risk
    detect_risk = round(CVaR(s, [1], 0.0), digits=6)
    # Get the no detect risk
    no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
    # Get name of text file
    fn = df[i, "filename"]
    text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v3_rl/valid/labels/$(fn).txt"
    # Write the risks to it
    io = open(text_file_name, "r")
    temp = read(io, String)
    close(io)

    new_string = "$(temp[1:end-1]) $(detect_risk) $(no_detect_risk)"

    io = open(text_file_name, "w")
    write(io, new_string)
    close(io)
end

"""
Overwrite Existing
"""

using DelimitedFiles

text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v1_rl/train/labels/0.txt"
io = open(text_file_name, "r")
temp = read(io, String)
close(io)

labels = readdlm(text_file_name)
new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) 0 0"

# Loop through files
for i = 1:9500
    # Get the mdp state
    s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
    # Get the detect risk
    detect_risk = round(CVaR(s, [1], 0.0), digits=6)
    # Get the no detect risk
    no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
    # Get name of text file
    fn = df[i, "filename"]
    text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v3_rl/train/labels/$(fn).txt"
    # Write the risks to it
    labels = readdlm(text_file_name)
    new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) $(detect_risk) $(no_detect_risk)"

    io = open(text_file_name, "w")
    write(io, new_string)
    close(io)
end

# Loop through files
for i = 9501:10000
    # Get the mdp state
    s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
    # Get the detect risk
    detect_risk = round(CVaR(s, [1], 0.0), digits=6)
    # Get the no detect risk
    no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
    # Get name of text file
    fn = df[i, "filename"]
    text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v1_rl/valid/labels/$(fn).txt"
    # Write the risks to it
    labels = readdlm(text_file_name)
    new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) $(detect_risk) $(no_detect_risk)"

    io = open(text_file_name, "w")
    write(io, new_string)
    close(io)
end