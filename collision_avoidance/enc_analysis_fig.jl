using POMDPs, POMDPGym, Crux, Flux, Distributions, GridInterpolations, Plots
using PyCall
using Random
using BSON
using BSON: @save
using Images

include("encounter_model/straight_line_model.jl")

# Set up risk stuff
include("../src/risk_solvers.jl")

# Set up the cost function and risk mdp
const Px = DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1])
const HNMAC = 100
const VNMAC = 50

env = CollisionAvoidanceMDP(px=Px, ddh_max=1.0, actions=[-8.0, 0.0, 8.0])
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)
env_risk = DetectAndAvoidMDP(ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
    actions=[-8.0, 0.0, 8.0])
costfn(m, s, sp) = isterminal(m, sp) ? 150 - abs(s[1]) : 0.0
rmdp = RMDP(env_risk, policy, costfn, false, 1.0, 40.0, :both)

# Start with just detect noise
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

cost_points = collect(range(0, 150, 50))
s_grid = RectangleGrid(hs, dhs, env.actions, τs)
𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs];
s2pt(s) = s

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);

CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

function mdp_state(s0, s1, a_prev)
    h = s0.z - s1.z
    dh = s0.dh - s1.dh

    dt = 0.1
    r0 = [s0.x, s0.y]
    r0_next = r0 + s0.v * dt * [-sind(s0.θ), cosd(s0.θ)]

    r1 = [s1.x, s1.y]
    r1_next = r1 + s1.v * dt * [-sind(s1.θ), cosd(s1.θ)]

    r = norm(r0 - r1)
    r_next = norm(r0_next - r1_next)

    ṙ = (r - r_next) / dt

    τ = r < HNMAC ? 0 : (r - HNMAC) / ṙ
    if τ < 0
        τ = Inf
    end

    [h, dh, a_prev, τ]
end

# Load in the data
res = BSON.load("collision_avoidance/data_files/example_encounter_info.bson")
b, rd, rl, rdrl = res[:b], res[:rd], res[:rl], res[:rdrl]

# Vertical profiles
function plot_enc_diffs(enc1, enc2, enc3, enc4; kwargs...)
    pv = plot(enc1.z0, marker_z=enc1.a, line_z=enc1.a, marker=true, clims=(-5, 5), label="own 1", c=:binary, legend=:topleft)
    plot!(enc2.z0 .- 2, marker_z=enc2.a, line_z=enc2.a, marker=true, clims=(-5, 5), label="own 2", c=:blues)
    plot!(enc3.z0 .- 4, marker_z=enc3.a, line_z=enc3.a, marker=true, clims=(-5, 5), label="own 3", c=:greens)
    plot!(enc4.z0 .- 6, marker_z=enc4.a, line_z=enc4.a, marker=true, clims=(-5, 5), label="own 4", c=:reds)
    plot!(enc1.z1, marker=true, label="intruder 0"; kwargs...)

    return pv
end

plot_enc_diffs(b[1], rd[1], rl[1], rdrl[1])

# Analyze risk
function get_risks(enc::Encounter, bbs; α=0)
    N = length(bbs)
    risks = zeros(N)
    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=s0.z, v=s0.v, dh=enc.dh0[t], θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=s1.z, v=s1.v, dh=enc.dh1[t], θ=s1.θ)

        a_prev = t == 1 ? 0.0 : enc.a[t-1]
        s = mdp_state(s0, s1, a_prev)

        detect = bbs[t] ? 1 : 0
        risks[t] = CVaR(s, [detect], α)
    end
    return risks
end

function plot_risk_compare(b, rd, rl, rdrl)
    risks_b, risks_rd, risks_rl, risks_rdrl = get_risks(b[1], b[2]), get_risks(rd[1], rd[2]), get_risks(rl[1], rl[2]), get_risks(rdrl[1], rdrl[2])
    plot(risks_b, label="baseline")
    plot!(risks_rd, label="risk data")
    plot!(risks_rl, label="risk loss")
    plot!(risks_rdrl, label="risk data risk loss", legend=:topleft)
end

function plot_crisk_compare(b, rd, rl, rdrl)
    risks_b, risks_rd, risks_rl, risks_rdrl = get_risks(b[1], b[2]), get_risks(rd[1], rd[2]), get_risks(rl[1], rl[2]), get_risks(rdrl[1], rdrl[2])
    plot(cumsum(risks_b), label="baseline", legend=:topleft)
    plot!(cumsum(risks_rd), label="risk data")
    plot!(cumsum(risks_rl), label="risk loss")
    plot!(cumsum(risks_rdrl), label="risk data risk loss")
end

function plot_risk_compare(b, rdrl)
    risks_b, risks_rdrl = get_risks(b[1], b[2]), get_risks(rdrl[1], rdrl[2])
    plot(risks_b, marker_z=b[1].a, marker=true, label="baseline")
    plot!(risks_rdrl, marker_z=rdrl[1].a, marker=true, label="risk data risk loss", legend=:topleft)
end

function plot_crisk_compare(b, rdrl)
    risks_b, risks_rdrl = get_risks(b[1], b[2]), get_risks(rdrl[1], rdrl[2])
    plot(cumsum(risks_b), label="baseline", legend=:topleft)
    plot!(cumsum(risks_rdrl), label="risk data risk loss")
end

plot_risk_compare(b, rd, rl, rdrl)
plot_crisk_compare(b, rd, rl, rdrl)

plot_risk_compare(b, rdrl)
plot_crisk_compare(b, rdrl)

# Plot frames
rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])
function plot_frames(data, times; crop_w=1:1920, crop_h=1:1056)
    xmin, xmax = minimum(crop_w), maximum(crop_w)
    ymin, ymax = minimum(crop_h), maximum(crop_h)
    sw = xmax - xmin + 1
    sh = ymax - ymin + 1

    # Concatenate screenshots
    sss = [cat(data[7][times[i]][crop_h, crop_w, :], 0xff * ones(UInt8, sh, 20, 3), dims=2) for i = 1:length(times)-1]
    push!(sss, data[7][times[end]][crop_h, crop_w, :])
    im = cat(sss..., dims=2)

    th = size(im, 1)
    tw = size(im, 2)

    p = plot(colorview(RGB, permutedims(im ./ 255, [3, 1, 2])), legend=false, axis=([], false), framestyle=:none)

    # Bounding boxes
    for (i, t) in enumerate(times)
        bb = data[2][t]
        if bb
            bbw, bbh, xp, yp = data[5][t], data[6][t], data[3][t], data[4][t]
            xp = xp - xmin + 1
            yp = yp - ymin + 1
            plot!(p, rectangle(bbw, bbh, xp + (sw + 20) * (i - 1), yp), fillalpha=0.0, lc=:red,
                legend=false, axis=([], false), size=(2000, (th / tw) * 2000))
        end
    end
    p
end

p = plot_frames(rdrl, [24, 26, 32, 36], crop_w=700:1220, crop_h=200:856)
p = plot_frames(b, [24, 26, 32, 36], crop_w=700:1220, crop_h=200:856)

function plot_frames_compare(data1, data2, times; crop_w=1:1920, crop_h=1:1056)
    xmin, xmax = minimum(crop_w), maximum(crop_w)
    ymin, ymax = minimum(crop_h), maximum(crop_h)
    sw = xmax - xmin + 1
    sh = ymax - ymin + 1

    # Concatenate screenshots
    sss1 = [cat(data1[7][times[i]][crop_h, crop_w, :], 0xff * ones(UInt8, sh, 20, 3), dims=2) for i = 1:length(times)-1]
    push!(sss1, data1[7][times[end]][crop_h, crop_w, :])
    im1 = cat(sss1..., dims=2)
    im_width = size(im1, 2)
    im1 = cat(im1, 0xff * ones(UInt8, 20, im_width, 3), dims=1)

    sss2 = [cat(data2[7][times[i]][crop_h, crop_w, :], 0xff * ones(UInt8, sh, 20, 3), dims=2) for i = 1:length(times)-1]
    push!(sss2, data2[7][times[end]][crop_h, crop_w, :])
    im2 = cat(sss2..., dims=2)

    im = cat(im1, im2, dims=1)

    th = size(im, 1)
    tw = size(im, 2)

    p = plot(colorview(RGB, permutedims(im ./ 255, [3, 1, 2])), legend=false, axis=([], false), framestyle=:none)

    # Bounding boxes
    for (i, t) in enumerate(times)
        bb1 = data1[2][t]
        if bb1
            bbw, bbh, xp, yp = data1[5][t], data1[6][t], data1[3][t], data1[4][t]
            xp = xp - xmin + 1
            yp = yp - ymin + 1
            plot!(p, rectangle(bbw, bbh, xp + (sw + 20) * (i - 1), yp), fillalpha=0.0, lc=:red,
                legend=false, axis=([], false), size=(2000, (th / tw) * 2000))
        end

        bb2 = data2[2][t]
        if bb2
            bbw, bbh, xp, yp = data2[5][t], data2[6][t], data2[3][t], data2[4][t]
            xp = xp - xmin + 1
            yp = yp - ymin + 1
            plot!(p, rectangle(bbw, bbh, xp + (sw + 20) * (i - 1), yp + sh + 20), fillalpha=0.0, lc=:red,
                legend=false, axis=([], false), size=(2000, (th / tw) * 2000))
        end
    end
    p
end

p = plot_frames_compare(b, rdrl, [24, 26, 32, 36], crop_w=700:1220, crop_h=250:806)
savefig(p, "collision_avoidance/figures/im_ex_compare.png")

risks_b, risks_rdrl = get_risks(b[1], b[2]), get_risks(rdrl[1], rdrl[2])
@save "collision_avoidance/data_files/enc_pgf_info.bson" risks_b risks_rdrl b rdrl