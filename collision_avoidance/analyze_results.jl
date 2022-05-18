using POMDPs, POMDPGym, Crux, Flux, Distributions, GridInterpolations, Plots
using PyCall
using Random
using BSON
using BSON: @save
using Images

include("simulate.jl")

# Set up risk stuff
include("../src/risk_solvers.jl")

# Set up the cost function and risk mdp
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

# Load in the models
# uniform_v1 = xctrl.util.load_model("collision_avoidance/models/uniform_v1.pt")
# uniform_v2 = xctrl.util.load_model("collision_avoidance/models/uniform_v2.pt")
# uniform_v3 = xctrl.util.load_model("collision_avoidance/models/uniform_v3.pt")
# risk_v1 = xctrl.util.load_model("collision_avoidance/models/risk_v1.pt")
# risk_v2 = xctrl.util.load_model("collision_avoidance/models/risk_v2.pt")
# risk_v3 = xctrl.util.load_model("collision_avoidance/models/risk_v3.pt")
# uniform_v1_rl = xctrl.util.load_model("collision_avoidance/models/uniform_v1_rl.pt")
# uniform_v2_rl = xctrl.util.load_model("collision_avoidance/models/uniform_v2_rl.pt")
# uniform_v3_rl = xctrl.util.load_model("collision_avoidance/models/uniform_v3_rl.pt")
# risk_v1_rl = xctrl.util.load_model("collision_avoidance/models/risk_v1_rl.pt")
# risk_v2_rl = xctrl.util.load_model("collision_avoidance/models/risk_v2_rl.pt")
# risk_v3_rl = xctrl.util.load_model("collision_avoidance/models/risk_v3_rl.pt")

# Load in the results
res = BSON.load("collision_avoidance/data_files/uniform_res.bson")
sim_uniform_v1, sim_uniform_v2, sim_uniform_v3 = res[:sim_uniform_v1], res[:sim_uniform_v2], res[:sim_uniform_v3]
res = BSON.load("collision_avoidance/data_files/risk_res.bson")
sim_risk_v1, sim_risk_v2, sim_risk_v3 = res[:sim_risk_v1], res[:sim_risk_v2], res[:sim_risk_v3]
res = BSON.load("collision_avoidance/data_files/uniform_res_rl.bson")
sim_uniform_v1_rl, sim_uniform_v2_rl, sim_uniform_v3_rl = res[:sim_uniform_v1_rl], res[:sim_uniform_v2_rl], res[:sim_uniform_v3_rl]
sim_uniform_v1_rl = BSON.load("collision_avoidance/data_files/uniform_v1_rl_res.bson")[:sim_uniform_v1_rl]
res = BSON.load("collision_avoidance/data_files/risk_res_rl.bson")
sim_risk_v1_rl, sim_risk_v2_rl, sim_risk_v3_rl = res[:sim_risk_v1_rl], res[:sim_risk_v2_rl], res[:sim_risk_v3_rl]

# Get NMACs
nmacs_uniform_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v1])
nmacs_uniform_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v2])
nmacs_uniform_v3 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v3])
nmacs_risk_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v1])
nmacs_risk_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v2])
nmacs_risk_v3 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v3])
nmacs_uniform_v1_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v1_rl])
nmacs_uniform_v2_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v2_rl])
nmacs_uniform_v3_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v3_rl])
nmacs_risk_v1_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v1_rl])
nmacs_risk_v2_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v2_rl])
nmacs_risk_v3_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v3_rl])

# Get NMAC inds
inds_uniform_v1 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v1])
inds_uniform_v2 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v2])
inds_uniform_v3 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v3])
inds_risk_v1 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v1])
inds_risk_v2 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v2])
inds_risk_v3 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v3])
inds_uniform_v1_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v1_rl])
inds_uniform_v2_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v2_rl])
inds_uniform_v3_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v3_rl])
inds_risk_v1_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v1_rl])
inds_risk_v2_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v2_rl])
inds_risk_v3_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v3_rl])

# Get nmacs resolved by risk design
uniform_inds = inds_uniform_v3[findall(in(inds_uniform_v2[findall(in(inds_uniform_v1), inds_uniform_v2)]), inds_uniform_v3)]
risk_inds = inds_risk_v3[findall(in(inds_risk_v2[findall(in(inds_risk_v1), inds_risk_v2)]), inds_risk_v3)]
uniform_inds_rl = inds_uniform_v3_rl[findall(in(inds_uniform_v2_rl[findall(in(inds_uniform_v1_rl), inds_uniform_v2_rl)]), inds_uniform_v3_rl)]
risk_inds_rl = inds_risk_v3_rl[findall(in(inds_risk_v2_rl[findall(in(inds_risk_v1_rl), inds_risk_v2_rl)]), inds_risk_v3_rl)]

unr = uniform_inds[findall(!in(risk_inds), uniform_inds)]
unrl = uniform_inds[findall(!in(uniform_inds_rl), uniform_inds)]
unrrl = uniform_inds[findall(!in(risk_inds_rl), uniform_inds)]

resolved_inds = unrrl[findall(in(unrl[findall(in(unr), unrl)]), unrrl)]

# Plot some of the resolved ones
function plot_enc_diffs(enc1, enc2, enc3, enc4; kwargs...)
    pv = plot(enc1.z0, marker_z=enc1.a, line_z=enc1.a, marker=true, clims=(-5, 5), label="own 1", c=:binary, legend=:topleft)
    plot!(enc2.z0 .- 2, marker_z=enc2.a, line_z=enc2.a, marker=true, clims=(-5, 5), label="own 2", c=:blues)
    plot!(enc3.z0 .- 4, marker_z=enc3.a, line_z=enc3.a, marker=true, clims=(-5, 5), label="own 3", c=:greens)
    plot!(enc4.z0 .- 6, marker_z=enc4.a, line_z=enc4.a, marker=true, clims=(-5, 5), label="own 4", c=:reds)
    plot!(enc1.z1, marker=true, label="intruder 0"; kwargs...)

    return pv
end

ind = resolved_inds[8]
plot_enc_diffs(sim_uniform_v2[ind], sim_risk_v2[ind], sim_uniform_v2_rl[ind],
    sim_risk_v2_rl[ind])

enc1 = sim_uniform_v2[ind]
enc4 = sim_risk_v2_rl[ind]
pv = plot(enc1.z0, marker_z=enc1.a, line_z=enc1.a, marker=true, clims=(-5, 5), label="own 1", c=:blues, legend=:topleft)
plot!(pv, enc4.z0 .- 2.0, marker_z=enc4.a, line_z=enc4.a, marker=true, clims=(-5, 5), label="own 4", c=:reds)
plot!(pv, enc4.z0 .- 2.0, marker_z=enc4.a, line_z=enc4.a, marker=true, clims=(-5, 5), label="own 4", c=:greens)

# Simulate them
function sim_encs_compare(ind)
    sleep(2)
    b = simulate_encounter_detailed(new_encs[ind], policy, seed=ind, save=false, xplane_control=xctrl, model=uniform_v2, bb_error_tol=100.0)
    rd = simulate_encounter_detailed(new_encs[ind], policy, seed=ind, save=false, xplane_control=xctrl, model=risk_v2, bb_error_tol=100.0)
    rl = simulate_encounter_detailed(new_encs[ind], policy, seed=ind, save=false, xplane_control=xctrl, model=uniform_v2_rl, bb_error_tol=100.0)
    rdrl = simulate_encounter_detailed(new_encs[ind], policy, seed=ind, save=false, xplane_control=xctrl, model=risk_v2_rl, bb_error_tol=100.0)

    return b, rd, rl, rdrl
end

b, rd, rl, rdrl = sim_encs_compare(ind)

nb, nrd, nrl, nrdrl = (is_nmac(b[1]), is_nmac(rd[1]), is_nmac(rl[1]), is_nmac(rdrl[1]))
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
    plot!(risks_rdrl, label="risk data risk loss")
end

function plot_crisk_compare(b, rd, rl, rdrl)
    risks_b, risks_rd, risks_rl, risks_rdrl = get_risks(b[1], b[2]), get_risks(rd[1], rd[2]), get_risks(rl[1], rl[2]), get_risks(rdrl[1], rdrl[2])
    plot(cumsum(risks_b), label="baseline", legend=:topleft)
    plot!(cumsum(risks_rd), label="risk data")
    plot!(cumsum(risks_rl), label="risk loss")
    plot!(cumsum(risks_rdrl), label="risk data risk loss")
end

plot_risk_compare(b, rd, rl, rdrl)
plot_crisk_compare(b, rd, rl, rdrl)

# Plot some frames
rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])
function plot_frame(data, t)
    bb, ss, bbw, bbh, xp, yp, a = data[2][t], data[7][t], data[5][t], data[6][t], data[3][t], data[4][t], data[1].a[t]
    title = "Clear of Conflict"
    if a > 0
        title = "Climb!"
    elseif a < 0
        title = "Descend!"
    end
    p = plot(colorview(RGB, permutedims(ss ./ 255, [3, 1, 2])), title=title)
    if bb
        plot!(p, rectangle(bbw, bbh, xp, yp), fillalpha=0.0, lc=:red, legend=false, axis=([], false))
    end
    p
end

t = 36
plot_frame(b, t)

function plot_frames(data, times; crop_w=1:1920, crop_h=1:1056)
    xmin, xmax = minimum(crop_w), maximum(crop_w)
    ymin, ymax = minimum(crop_h), maximum(crop_h)
    sw = xmax - xmin + 1
    sh = ymax - ymin + 1

    # Concatenate screenshots
    sss = [cat(data[7][times[i]][crop_h, crop_w, :], ones(UInt8, sh, 20, 3), dims=2) for i = 1:length(times)-1]
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

plot_frames(b, [35, 36, 37])
plot_frames(b, [35, 36, 37, 38, 39], crop_w=500:1420)
plot_frames(rd, [35, 36, 37, 38, 39], crop_w=500:1420)
p = plot_frames(b, [34, 35, 36, 37], crop_w=500:1420, crop_h=200:856)

println(b[2][30:40])

# Aggregate risks
function get_risks(enc::Encounter; α=0)
    bbs = [a != 0 for a in enc.a]
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

function get_relative_risks(enc::Encounter; α=0)
    bbs = [a != 0 for a in enc.a]
    N = length(bbs)
    risks = zeros(N)
    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=s0.z, v=s0.v, dh=enc.dh0[t], θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=s1.z, v=s1.v, dh=enc.dh1[t], θ=s1.θ)

        a_prev = t == 1 ? 0.0 : enc.a[t-1]
        s = mdp_state(s0, s1, a_prev)

        actual = bbs[t] ? 1 : 0
        other = bbs[t] ? 0 : 1
        risks[t] = CVaR(s, [other], α) - CVaR(s, [actual], α)
    end
    return risks
end

# Risks
uniform_v1_risks = vcat([get_risks(enc) for enc in sim_uniform_v1]...)
uniform_v2_risks = vcat([get_risks(enc) for enc in sim_uniform_v2]...)
uniform_v3_risks = vcat([get_risks(enc) for enc in sim_uniform_v3]...)
risk_v1_risks = vcat([get_risks(enc) for enc in sim_risk_v1]...)
risk_v2_risks = vcat([get_risks(enc) for enc in sim_risk_v2]...)
risk_v3_risks = vcat([get_risks(enc) for enc in sim_risk_v3]...)
uniform_v1_rl_risks = vcat([get_risks(enc) for enc in sim_uniform_v1_rl]...)
uniform_v2_rl_risks = vcat([get_risks(enc) for enc in sim_uniform_v2_rl]...)
uniform_v3_rl_risks = vcat([get_risks(enc) for enc in sim_uniform_v3_rl]...)
risk_v1_rl_risks = vcat([get_risks(enc) for enc in sim_risk_v1_rl]...)
risk_v2_rl_risks = vcat([get_risks(enc) for enc in sim_risk_v2_rl]...)
risk_v3_rl_risks = vcat([get_risks(enc) for enc in sim_risk_v3_rl]...)

uniform_risks = vcat(uniform_v1_risks, uniform_v2_risks, uniform_v3_risks)
risk_risks = vcat(risk_v1_risks, risk_v2_risks, risk_v3_risks)
uniform_rl_risks = vcat(uniform_v1_rl_risks, uniform_v2_rl_risks, uniform_v3_rl_risks)
risk_rl_risks = vcat(risk_v1_rl_risks, risk_v2_rl_risks, risk_v3_rl_risks)

histogram(uniform_risks, bins=collect(range(0, 150, 50)), alpha=0.5)
histogram!(risk_risks, bins=collect(range(0, 150, 50)), alpha=0.5)
histogram!(uniform_rl_risks, bins=collect(range(0, 150, 50)), alpha=0.5)
histogram!(risk_rl_risks, bins=collect(range(0, 150, 50)), alpha=0.5)

function get_cdf(risks; points=collect(0:150))
    N = length(risks)
    vals = zeros(length(points))
    for (i, point) in enumerate(points)
        vals[i] = sum(risks .< point)
    end
    return vals ./ N
end

uniform_cdf = get_cdf(uniform_risks)
risk_cdf = get_cdf(risk_risks)
uniform_rl_cdf = get_cdf(uniform_rl_risks)
risk_rl_cdf = get_cdf(risk_rl_risks)

plot(collect(0:150), uniform_cdf, legend=:topleft)
plot!(collect(0:150), risk_cdf)
plot!(collect(0:150), uniform_rl_cdf)
plot!(collect(0:150), risk_rl_cdf)

#@save "collision_avoidance/data_files/cdfs.bson" uniform_cdf risk_cdf uniform_rl_cdf risk_rl_cdf

# Relative Risks
uniform_v1_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v1]...)
uniform_v2_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v2]...)
uniform_v3_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v3]...)
risk_v1_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v1]...)
risk_v2_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v2]...)
risk_v3_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v3]...)
uniform_v1_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v1_rl]...)
uniform_v2_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v2_rl]...)
uniform_v3_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_uniform_v3_rl]...)
risk_v1_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v1_rl]...)
risk_v2_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v2_rl]...)
risk_v3_rl_rrisks = vcat([get_relative_risks(enc) for enc in sim_risk_v3_rl]...)

uniform_rrisks = vcat(uniform_v1_rrisks, uniform_v2_rrisks, uniform_v3_rrisks)
risk_rrisks = vcat(risk_v1_rrisks, risk_v2_rrisks, risk_v3_rrisks)
uniform_rl_rrisks = vcat(uniform_v1_rl_rrisks, uniform_v2_rl_rrisks, uniform_v3_rl_rrisks)
risk_rl_rrisks = vcat(risk_v1_rl_rrisks, risk_v2_rl_rrisks, risk_v3_rl_rrisks)

uniform_rrcdf = get_cdf(uniform_rrisks)
risk_rrcdf = get_cdf(risk_rrisks)
uniform_rl_rrcdf = get_cdf(uniform_rl_rrisks)
risk_rl_rrcdf = get_cdf(risk_rl_rrisks)

plot(collect(0:150), uniform_rrcdf, legend=:bottomright)
plot!(collect(0:150), risk_rrcdf)
plot!(collect(0:150), uniform_rl_rrcdf)
plot!(collect(0:150), risk_rl_rrcdf)


histogram(uniform_rrisks, alpha=0.5)
histogram!(risk_rrisks, alpha=0.5)
histogram!(uniform_rl_rrisks, alpha=0.5)
histogram!(risk_rl_rrisks, alpha=0.5)

histogram(uniform_rrisks[uniform_rrisks.!=0], bins=collect(range(-30, 30, 25)), alpha=0.5)
histogram!(risk_rrisks[risk_rrisks.!=0], bins=collect(range(-30, 30, 25)), alpha=0.5)
histogram!(uniform_rl_rrisks[uniform_rl_rrisks.!=0], bins=collect(range(-30, 30, 25)), alpha=0.5)
histogram!(risk_rl_rrisks[risk_rl_rrisks.!=0], bins=collect(range(-30, 30, 25)), alpha=0.5)