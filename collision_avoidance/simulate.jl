using POMDPs, POMDPGym, Crux, Flux, Distributions, GridInterpolations, Plots
using PyCall
using Random
using BSON
using BSON: @save
using Images

include("encounter_model/straight_line_model.jl")

const HNMAC = 100
const VNMAC = 50

const DDH_MAX = 1.0
const Px = DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1])
# const Px = DiscreteNonParametric([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])

struct XPlaneControl
    util
    client
end

function XPlaneControl()
    if pyimport("sys")."path"[1] != "/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/"
        pushfirst!(pyimport("sys")."path", "/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/")
    end
    xplane_ctrl = pyimport("util")
    xplane_ctrl = pyimport("importlib")["reload"](xplane_ctrl)
    xpc3 = pyimport("data_generation.xpc3")
    xplane_client = xpc3.XPlaneConnect()
    xplane_client.pauseSim(true)
    xplane_client.sendDREF("sim/operation/override/override_joystick", 1)

    XPlaneControl(xplane_ctrl, xplane_client)
end

# xplane_ctrl.get_bounding_box(xplane_client, model, 0,0,0,0, 0,200,0,0)

function step_aircraft(s, a=0.0)
    h = s.z
    dh = s.dh

    h = h + dh
    if abs(a - dh) < DDH_MAX
        dh += a - dh
    else
        dh += sign(a - dh) * DDH_MAX
    end
    dh += rand(Px)
    h, dh
end

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

function bb_center(s0, s1; hfov=80.0, vfov=49.5, sw=1920, sh=1056)
    # Make ownship be the origin
    x = s1.y - s0.y
    y = -(s1.x - s0.x)  # right-handed coordinates
    z = s1.z - s0.z

    # Rotate x and y according to ownship heading
    xrot = x * cosd(-s0.θ) - y * sind(-s0.θ)
    yrot = -(x * sind(-s0.θ) + y * cosd(-s0.θ))

    # https://www.youtube.com/watch?v=LhQ85bPCAJ8
    xp = yrot / (xrot * tand(hfov / 2))
    yp = z / (xrot * tand(vfov / 2))

    # Get xp and yp between 0 and 1
    xp = (xp + 1) / 2
    yp = (yp + 1) / 2

    # Map to pixel location
    xp = xp * sw
    yp = (1 - yp) * sh

    return xp, yp
end

function simulate_encounter(enc::Encounter, policy; save=true, xplane_control=nothing, model=nothing, seed=1, bb_error_tol=Inf)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
    Random.seed!(seed)

    s0s = []
    s1s = []
    N = length(enc.x0)
    a_prev = 0
    s0 = get_ownship_state(enc, 1)
    s1 = get_intruder_state(enc, 1)
    z0, dh0 = s0.z, s0.dh
    z1, dh1 = s1.z, s1.dh
    as = []
    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=z0, v=s0.v, dh=dh0, θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=z1, v=s1.v, dh=dh1, θ=s1.θ)

        # Store the state
        push!(s0s, s0)
        push!(s1s, s1)

        # Optionally call python to set state and take screenshot
        if !isnothing(xplane_control)
            save_num = save ? t : -1
            bb, boxes = xplane_control.util.get_bounding_box(xplane_control.client, model, s0.x, s0.y, s0.z, s0.θ, s1.x, s1.y, s1.z, s1.θ, save_num)
            if bb
                xp_gt, yp_gt = bb_center(s0, s1)
                min_error = Inf
                for i = 1:size(boxes, 1)
                    xp, yp, _, _ = boxes[i, :]
                    e = norm([xp, yp] - [xp_gt, yp_gt])
                    if e < min_error
                        min_error = e
                    end
                end
                if min_error > bb_error_tol
                    bb = false
                end
            end
        else
            bb = true
        end

        # compute the next state
        a = bb ? action(policy, mdp_state(s0, s1, a_prev)) : 0.0
        a_prev = a
        push!(as, a)

        z0, dh0 = step_aircraft(s0, a)
        z1, dh1 = step_aircraft(s1)
    end
    save ? xplane_control.util.create_gif(N) : nothing
    return Encounter(s0s, s1s, as)
end

function simulate_encounter_detailed(enc::Encounter, policy; save=true, xplane_control=nothing, model=nothing, seed=1, bb_error_tol=Inf)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
    Random.seed!(seed)

    s0s = []
    s1s = []
    N = length(enc.x0)
    a_prev = 0
    s0 = get_ownship_state(enc, 1)
    s1 = get_intruder_state(enc, 1)
    z0, dh0 = s0.z, s0.dh
    z1, dh1 = s1.z, s1.dh
    as = []

    # Extra info
    bbs = []
    xps = []
    yps = []
    bbws = []
    bbhs = []
    sss = []

    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=z0, v=s0.v, dh=dh0, θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=z1, v=s1.v, dh=dh1, θ=s1.θ)

        # Store the state
        push!(s0s, s0)
        push!(s1s, s1)

        # Optionally call python to set state and take screenshot
        if !isnothing(xplane_control)
            save_num = save ? t : -1
            sleep(0.1)
            bb, boxes, ss = xplane_control.util.get_bounding_box_and_ss(xplane_control.client, model, s0.x, s0.y, s0.z, s0.θ, s1.x, s1.y, s1.z, s1.θ, save_num)
            if bb
                xp_gt, yp_gt = bb_center(s0, s1)
                min_error = Inf
                min_ind = 0
                for i = 1:size(boxes, 1)
                    xp, yp, _, _ = boxes[i, :]
                    e = norm([xp, yp] - [xp_gt, yp_gt])
                    if e < min_error
                        min_error = e
                        min_ind = i
                    end
                end
                if min_error > bb_error_tol
                    bb = false
                    push!(xps, 0.0)
                    push!(yps, 0.0)
                    push!(bbws, 0.0)
                    push!(bbhs, 0.0)
                    push!(sss, ss)
                else
                    xp, yp, w, h = boxes[min_ind, :]
                    push!(xps, xp)
                    push!(yps, yp)
                    push!(bbws, w)
                    push!(bbhs, h)
                    push!(sss, ss)
                end
            else
                push!(xps, 0.0)
                push!(yps, 0.0)
                push!(bbws, 0.0)
                push!(bbhs, 0.0)
                push!(sss, ss)
            end
        else
            bb = true
        end

        push!(bbs, bb)

        # compute the next state
        a = bb ? action(policy, mdp_state(s0, s1, a_prev)) : 0.0
        a_prev = a
        push!(as, a)

        z0, dh0 = step_aircraft(s0, a)
        z1, dh1 = step_aircraft(s1)
    end
    save ? xplane_control.util.create_gif(N) : nothing
    return Encounter(s0s, s1s, as), bbs, xps, yps, bbws, bbhs, sss
end

# sim_uniform_v1_test = simulate_encounters([new_encs[1]], policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v1, bb_error_tol=100.0);

function simulate_encounter_for_info(enc::Encounter, policy; sleep_time=0, save=true, xplane_control=nothing, model=nothing, seed=1)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
    sleep(sleep_time)
    Random.seed!(seed)

    s0s = []
    s1s = []
    N = length(enc.x0)
    a_prev = 0
    s0 = get_ownship_state(enc, 1)
    s1 = get_intruder_state(enc, 1)
    z0, dh0 = s0.z, s0.dh
    z1, dh1 = s1.z, s1.dh
    as = []

    # Extra info
    bbs = []
    xps = []
    yps = []
    bbws = []
    bbhs = []
    sss = []

    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=z0, v=s0.v, dh=dh0, θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=z1, v=s1.v, dh=dh1, θ=s1.θ)

        # Store the state
        push!(s0s, s0)
        push!(s1s, s1)

        # Optionally call python to set state and take screenshot
        if !isnothing(xplane_control)
            save_num = save ? t : -1
            bb, xp, yp, w, h, ss = xplane_control.util.get_bounding_box_and_ss(xplane_control.client, model, s0.x, s0.y, s0.z, s0.θ, s1.x, s1.y, s1.z, s1.θ, save_num)
            push!(xps, xp)
            push!(yps, yp)
            push!(bbws, w)
            push!(bbhs, h)
            push!(sss, ss)
        else
            bb = true
        end

        push!(bbs, bb)

        # compute the next state
        a = bb ? action(policy, mdp_state(s0, s1, a_prev)) : 0.0
        a_prev = a
        push!(as, a)

        z0, dh0 = step_aircraft(s0, a)
        z1, dh1 = step_aircraft(s1)
    end
    save ? xplane_control.util.create_gif(N) : nothing
    return Encounter(s0s, s1s, as), bbs, xps, yps, bbws, bbhs, sss
end

function simulate_encounters(encs, policy, sleep_time; kwargs...)
    sleep(sleep_time)
    return [simulate_encounter(enc, policy; seed=i, kwargs...) for (i, enc) in enumerate(encs)]
end

function is_nmac(enc::Encounter)
    for i = 1:length(enc.x0)
        hsep = sqrt((enc.x0[i] - enc.x1[i])^2 + (enc.y0[i] - enc.y1[i])^2)
        vsep = abs(enc.z0[i] - enc.z1[i])
        if hsep < HNMAC && vsep < VNMAC
            return true
        end
    end
    return false
end

function plot_enc(enc; kwargs...)
    # horizontal
    ph = plot(enc.x0, enc.y0, marker_z=enc.a, line_z=enc.a, marker=true, clims=(-5, 5), label="own"; kwargs...)
    plot!(enc.x1, enc.y1, marker=true, aspect_ratio=:equal, label="intruder")

    pv = plot(enc.z0, marker_z=enc.a, line_z=enc.a, marker=true, clims=(-5, 5), label="own")
    plot!(enc.z1, marker=true, label="intruder"; kwargs...)

    plot(ph, pv, layout=(2, 1), size=(600, 800))
end

function plot_enc_diffs(enc1, enc2; kwargs...)
    pv = plot(enc1.z0, marker_z=enc1.a, line_z=enc1.a, marker=true, clims=(-5, 5), label="own 1")
    plot!(enc2.z0, marker_z=enc2.a, line_z=enc2.a, marker=true, clims=(-5, 5), label="own 2", c=:summer)
    plot!(enc1.z1, marker=true, label="intruder 0"; kwargs...)

    return pv
end

# Create environment
env = CollisionAvoidanceMDP(px=Px, ddh_max=1.0, actions=[-8.0, 0.0, 8.0])
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)

# Get 100 encounters
Random.seed!(13)
encs = get_encounter_set(sampler, 1000)
nmacs = sum([is_nmac(enc) for enc in encs])

# Rotate and shift
new_encs = rotate_and_shift_encs(encs)
nmacs = sum([is_nmac(enc) for enc in new_encs])
#plot_enc(new_encs[2])

# # connect to xplane
# xctrl = XPlaneControl()

# # Full run
# uniform_v1 = xctrl.util.load_model("models/uniform_v1.pt")
# uniform_v2 = xctrl.util.load_model("models/uniform_v2.pt")
# uniform_v3 = xctrl.util.load_model("models/uniform_v3.pt")
# risk_v1 = xctrl.util.load_model("models/risk_v1.pt")
# risk_v2 = xctrl.util.load_model("models/risk_v2.pt")
# risk_v3 = xctrl.util.load_model("models/risk_v3.pt")
# uniform_v1_rl = xctrl.util.load_model("models/uniform_v1_rl.pt")
# uniform_v2_rl = xctrl.util.load_model("models/uniform_v2_rl.pt")
# uniform_v3_rl = xctrl.util.load_model("models/uniform_v3_rl.pt")
# risk_v1_rl = xctrl.util.load_model("models/risk_v1_rl.pt")
# risk_v2_rl = xctrl.util.load_model("models/risk_v2_rl.pt")
# risk_v3_rl = xctrl.util.load_model("models/risk_v3_rl.pt")

# # sim_gt = simulate_encounters(new_encs, policy, 0.0, save=false);
# # @time sim_uniform_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v1, bb_error_tol=100.0);
# # @time sim_uniform_v2 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v2, bb_error_tol=100.0);
# # @time sim_uniform_v3 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v3, bb_error_tol=100.0);
# # @time sim_risk_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v1, bb_error_tol=100.0);
# # @time sim_risk_v2 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v2, bb_error_tol=100.0);
# # @time sim_risk_v3 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v3, bb_error_tol=100.0);
# # @time sim_uniform_v1_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v1_rl, bb_error_tol=100.0);
# # @time sim_uniform_v2_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v2_rl, bb_error_tol=100.0);
# # @time sim_uniform_v3_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v3_rl, bb_error_tol=100.0);
# # @time sim_risk_v1_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v1_rl, bb_error_tol=100.0);
# # @time sim_risk_v2_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v2_rl, bb_error_tol=100.0);
# # @time sim_risk_v3_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v3_rl, bb_error_tol=100.0);

# # Load in the results
# res = "data_file/uniform_res.bson"
# sim_uniform_v1, sim_uniform_v2, sim_uniform_v3 = res[:sim_uniform_v1], res[:sim_uniform_v2], res[:sim_uniform_v3]
# res = "data_file/risk_res.bson"
# sim_risk_v1, sim_risk_v2, sim_risk_v3 = res[:sim_risk_v1], res[:sim_risk_v2], res[:sim_risk_v3]
# res = "data_file/uniform_res.bson"
# sim_uniform_v1, sim_uniform_v2, sim_uniform_v3 = res[:sim_uniform_v1], res[:sim_uniform_v2], res[:sim_uniform_v3]
# res = "data_file/risk_res.bson"
# sim_risk_v1, sim_risk_v2, sim_risk_v3 = res[:sim_risk_v1], res[:sim_risk_v2], res[:sim_risk_v3]

# nmacs_gt = sum([is_nmac(sim_enc) for sim_enc in sim_gt])
# nmacs_uniform_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v1])
# nmacs_uniform_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v2])
# nmacs_uniform_v3 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v3])
# nmacs_risk_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v1])
# nmacs_risk_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v2])
# nmacs_risk_v3 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v3])
# nmacs_uniform_v1_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v1_rl])
# nmacs_uniform_v2_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v2_rl])
# nmacs_uniform_v3_rl = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v3_rl])
# nmacs_risk_v1_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v1_rl])
# nmacs_risk_v2_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v2_rl])
# nmacs_risk_v3_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v3_rl])

# # @save "data_files/ground_truth.bson" sim_gt
# # @save "data_files/uniform_res.bson" sim_uniform_v1 sim_uniform_v2 sim_uniform_v3
# # @save "data_files/risk_res.bson" sim_risk_v1 sim_risk_v2 sim_risk_v3
# # @save "data_files/uniform_res_rl.bson" sim_uniform_v1_rl sim_uniform_v2_rl sim_uniform_v3_rl
# # @save "data_files/risk_res_rl.bson" sim_risk_v1_rl sim_risk_v2_rl sim_risk_v3_rl

# println("Perfect perception NMACs: ", nmacs_gt)
# println("Uniform Data NMACs: ", [nmacs_uniform_v1, nmacs_uniform_v2, nmacs_uniform_v3])
# println("Risk Data NMACs: ", [nmacs_risk_v1, nmacs_risk_v2, nmacs_risk_v3])
# println("Uniform Data Risk Loss NMACs: ", [nmacs_uniform_v1_rl, nmacs_uniform_v2_rl, nmacs_uniform_v3_rl])
# println("Risk Data Risk Loss NMACs: ", [nmacs_risk_v1_rl, nmacs_risk_v2_rl, nmacs_risk_v3_rl])

# # Analysis
# inds_uniform_v1 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v1])
# inds_uniform_v2 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v2])
# inds_uniform_v3 = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v3])
# inds_risk_v1 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v1])
# inds_risk_v2 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v2])
# inds_risk_v3 = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v3])
# inds_uniform_v1_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v1_rl])
# inds_uniform_v2_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v2_rl])
# inds_uniform_v3_rl = findall([is_nmac(sim_enc) for sim_enc in sim_uniform_v3_rl])
# inds_risk_v1_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v1_rl])
# inds_risk_v2_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v2_rl])
# inds_risk_v3_rl = findall([is_nmac(sim_enc) for sim_enc in sim_risk_v3_rl])

# uniform_inds = findall(in(findall(in(inds_uniform_v1), inds_uniform_v2)), inds_uniform_v3)
# risk_inds = findall(in(findall(in(inds_risk_v1), inds_risk_v2)), inds_risk_v3)
# uniform_inds_rl = findall(in(findall(in(inds_uniform_v1_rl), inds_uniform_v2_rl)), inds_uniform_v3_rl)
# risk_inds_rl = findall(in(findall(in(inds_risk_v1_rl), inds_risk_v2_rl)), inds_risk_v3_rl)

# unr = findall(!in(uniform_inds), risk_inds)
# unrl = findall(!in(uniform_inds_rl), uniform_inds)
# unrrl = findall(!in(risk_inds_rl), uniform_inds)

# resolved_inds = findall(in(findall(in(unr), unrl)), unrrl)

# # 5/11 run
# uniform_v1 = xctrl.util.load_model("collision_avoidance/models/uniform_v1.pt")
# risk_v1 = xctrl.util.load_model("collision_avoidance/models/risk_v1.pt")
# risk_v2_rl = xctrl.util.load_model("collision_avoidance/models/risk_v2_rl.pt")

# sim_gt = simulate_encounters(new_encs, policy, 0.0, save=false);
# @time sim_uniform_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=uniform_v1, bb_error_tol=100.0);
# @time sim_risk_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v1, bb_error_tol=100.0);
# @time sim_risk_v2_rl = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_v2_rl, bb_error_tol=100.0);

# nmacs_gt = sum([is_nmac(sim_enc) for sim_enc in sim_gt])
# nmacs_uniform_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_uniform_v1])
# nmacs_risk_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v1])
# nmacs_risk_v2_rl = sum([is_nmac(sim_enc) for sim_enc in sim_risk_v2_rl])
# nmacs

# # Load in the perception model
# baseline_v1 = xctrl.util.load_model("collision_avoidance/models/uniform_data_v1.pt")
# baseline_v2 = xctrl.util.load_model("collision_avoidance/models/uniform_data_v2.pt")
# risk_data_v1 = xctrl.util.load_model("collision_avoidance/models/risk_data_v1.pt")
# risk_data_v2 = xctrl.util.load_model("collision_avoidance/models/risk_data_v2.pt")
# risk_loss_1 = xctrl.util.load_model("collision_avoidance/models/risk_loss_100/uniform_1.pt")
# risk_loss_risk_data_1 = xctrl.util.load_model("collision_avoidance/models/risk_loss_100/risk_1.pt")
# risk_loss_0p1 = xctrl.util.load_model("collision_avoidance/models/risk_loss_100/uniform_0p1.pt")
# risk_loss_risk_data_0p1 = xctrl.util.load_model("collision_avoidance/models/risk_loss_100/risk_0p1.pt")

# # Simulate them
# sim_encs_gt = simulate_encounters(new_encs, policy, 0.0, save=false);
# @time sim_encs_baseline_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=baseline_v1);
# @time sim_encs_baseline_v2 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=baseline_v2);
# @time sim_encs_risk_data_v1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_data_v1);
# @time sim_encs_risk_data_v2 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_data_v2);
# @time sim_encs_risk_loss_1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_loss_1);
# @time sim_encs_risk_loss_risk_data_1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_loss_risk_data_1);

# #@save "collision_avoidance/data_files/baseline_v_risk_encs_corrections.bson" sim_encs_gt sim_encs_baseline sim_encs_risk_data

# # Count the number of nmacs
# nmacs_gt = sum([is_nmac(sim_enc) for sim_enc in sim_encs_gt])
# nmacs_baseline_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_baseline_v1])
# nmacs_baseline_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_baseline_v2])
# nmacs_risk_data_v1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_data_v1])
# nmacs_risk_data_v2 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_data_v2])
# nmacs_risk_loss_1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_loss_1])
# nmacs_risk_loss_risk_data_1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_loss_risk_data_1])
# nmacs_risk_loss_0p1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_loss_0p1])
# nmacs_risk_loss_risk_data_0p1 = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_loss_risk_data_0p1])

# # function run_me()
# #     sim_encs_risk_loss_1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_loss_0p1)
# #     sim_encs_risk_loss_risk_data_1 = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=risk_loss_risk_data_0p1)
# #     return sim_encs_risk_loss_1, sim_encs_risk_loss_risk_data_1
# # end

# # run_me()

# # Count the number of alerts
# alerts_gt = sum([sum(enc.a .!= 0.0) for enc in sim_encs_gt])
# alerts_baseline = sum([sum(enc.a .!= 0.0) for enc in sim_encs_baseline])
# alerts_risk_data = sum([sum(enc.a .!= 0.0) for enc in sim_encs_risk_data])

# alert_rate_gt = alerts_gt / 41000
# alert_rate_baseline = alerts_baseline / 41000
# alert_rate_risk_data = alerts_risk_data / 41000

# nmac_inds_baseline = findall([is_nmac(sim_enc) for sim_enc in sim_encs_baseline])
# nmac_inds_risk_data = findall([is_nmac(sim_enc) for sim_enc in sim_encs_risk_data])

# diff_inds = setdiff(nmac_inds_baseline, nmac_inds_risk_data)

# plot_enc(sim_encs_baseline[8])
# plot_enc(sim_encs_risk_data[8])

# plot_enc_diffs(sim_encs_baseline[16], sim_encs_risk_data[16])

# function run_it()
#     sleep(1)
#     simulate_encounter(encs[13], policy; save=true, xplane_control=xctrl, model=model_baseline, seed=13)
# end

# run_it()

# # Analyze risk
# include("../src/risk_solvers.jl")

# # Set up the cost function and risk mdp
# env_risk = DetectAndAvoidMDP(ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
#     actions=[-8.0, 0.0, 8.0])
# costfn(m, s, sp) = isterminal(m, sp) ? 150 - abs(s[1]) : 0.0
# rmdp = RMDP(env_risk, policy, costfn, false, 1.0, 40.0, :both)

# # Start with just detect noise
# detect_model = BSON.load("collision_avoidance/models/nominal_error_model.bson")[:m]
# p_detect(s) = detect_model([abs(s[1]), s[4]])[1] # sigmoid(-0.006518117 * abs(s[1]) - 0.10433467s[4] + 1.2849158)
# function get_detect_dist(s)
#     pd = p_detect(s)
#     noises = [[ϵ, 0.0, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
#     return ObjectCategorical(noises, [1 - pd, pd])
# end

# noises_detect = [0, 1]

# ϵ_grid = RectangleGrid(noises_detect)
# noises = [[ϵ[1], 0.0, 0.0, 0.0, 0.0] for ϵ in ϵ_grid]

# px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

# cost_points = collect(range(0, 150, 50))
# s_grid = RectangleGrid(hs, dhs, env.actions, τs)
# 𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs];
# s2pt(s) = s

# # Solve for distribution over costs
# @time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
#     cost_points, mdp_type=:exp);

# CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# enc_b, bbs_b, xps_b, yps_b, bbws_b, bbhs_b, sss_b = simulate_encounter_for_info(encs[16], policy; sleep_time=1, save=false, xplane_control=xctrl, model=model_baseline, seed=16)
# enc_r, bbs_r, xps_r, yps_r, bbws_r, bbhs_r, sss_r = simulate_encounter_for_info(encs[16], policy; sleep_time=1, save=false, xplane_control=xctrl, model=model_risk_data, seed=16)

# plot(bbs_b)
# plot!(bbs_r)

# function get_risks(enc::Encounter, bbs; α=0)
#     N = length(bbs)
#     risks = zeros(N)
#     for t in 1:N
#         s0 = get_ownship_state(enc, t)
#         s0 = (x=s0.x, y=s0.y, z=s0.z, v=s0.v, dh=enc.dh0[t], θ=s0.θ)
#         s1 = get_intruder_state(enc, t)
#         s1 = (x=s1.x, y=s1.y, z=s1.z, v=s1.v, dh=enc.dh1[t], θ=s1.θ)

#         a_prev = t == 1 ? 0.0 : enc.a[t-1]
#         s = mdp_state(s0, s1, a_prev)

#         detect = bbs[t] ? 1 : 0
#         risks[t] = CVaR(s, [detect], α)
#     end
#     return risks
# end

# risks_b = get_risks(enc_b, bbs_b)
# risks_r = get_risks(enc_r, bbs_r)

# plot(risks_b)
# plot!(risks_r)

# function get_relative_risks(enc::Encounter, bbs; α=0)
#     N = length(bbs)
#     risks = zeros(N)
#     for t in 1:N
#         s0 = get_ownship_state(enc, t)
#         s0 = (x=s0.x, y=s0.y, z=s0.z, v=s0.v, dh=enc.dh0[t], θ=s0.θ)
#         s1 = get_intruder_state(enc, t)
#         s1 = (x=s1.x, y=s1.y, z=s1.z, v=s1.v, dh=enc.dh1[t], θ=s1.θ)

#         a_prev = t == 1 ? 0.0 : enc.a[t-1]
#         s = mdp_state(s0, s1, a_prev)

#         actual = bbs[t] ? 1 : 0
#         other = bbs[t] ? 0 : 1
#         risks[t] = CVaR(s, [other], α) - CVaR(s, [actual], α)
#     end
#     return risks
# end

# relative_risks_b = get_relative_risks(enc_b, bbs_b)
# relative_risks_r = get_relative_risks(enc_r, bbs_r)

# plot(relative_risks_b)
# plot!(relative_risks_r)

# rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])

# function plot_frame(ss, bbw, bbh, xp, yp, a)
#     title = "Clear of Conflict"
#     if a > 0
#         title = "Climb!"
#     elseif a < 0
#         title = "Descend!"
#     end
#     plot(colorview(RGB, permutedims(ss ./ 255, [3, 1, 2])), title=title)
#     plot!(rectangle(bbw, bbh, xp, yp), fillalpha=0.0, lc=:red, legend=false, axis=([], false))
# end

# plot_frame(sss_b[35], bbws_b[35], bbhs_b[35], xps_b[35], yps_b[35], enc_b.a[35])

# plot(rectangle(2, 10, 0, 0), xlims=(0, 2), ylims=(-15, 15), aspect_ratio=:equal, legend=false, xaxis=([], false))

# function plot_frame_with_risk(ss, bb, bbw, bbh, xp, yp, a, risk)
#     title = "Clear of Conflict"
#     if a > 0
#         title = "Climb!"
#     elseif a < 0
#         title = "Descend!"
#     end
#     p1 = plot(colorview(RGB, permutedims(ss ./ 255, [3, 1, 2])), title=title, size=(600, 300),
#               legend=false, axis=([], false))
#     #if bb
#         plot!(p1, rectangle(bbw, bbh, xp, yp), fillalpha=0.0, lc=:red)
#     #end

#     if risk < 0
#         p2 = plot(rectangle(abs(risk), 2, risk, 0), xlims=(-15, 15), ylims=(0, 2), aspect_ratio=:equal,
#             legend=false, yaxis=([], false), lc=:red, fillcolor=:red, size=(600, 50), title="Relative Risk")
#     else
#         p2 = plot(rectangle(risk, 2, 0, 0), xlims=(-15, 15), ylims=(0, 2), aspect_ratio=:equal,
#             legend=false, yaxis=([], false), lc=:green, fillcolor=:green, size=(400, 50), title="Relative Risk")
#     end

#     p3 = plot(p1, p2, layout=grid(2, 1, heights=[0.9, 0.1]), size=(300, 300))
#     return p3
# end

# p1 = plot_frame_with_risk(sss_b[35], bbs_b[35], bbws_b[35], bbhs_b[35], xps_b[35], yps_b[35], enc_b.a[35], relative_risks_b[35])
# p2 = plot_frame_with_risk(sss_r[35], bbs_r[35], bbws_r[35], bbhs_r[35], xps_r[35], yps_r[35], enc_r.a[35], relative_risks_r[35])

# plot(p1, p2, size=(600, 300))

# function get_gif_frame(sss_b, bbs_b, bbws_b, bbhs_b, xps_b, yps_b, a_b, rrs_b,
#     sss_r, bbs_r, bbws_r, bbhs_r, xps_r, yps_r, a_r, rrs_r,
#     t)
#     p1 = plot_frame_with_risk(sss_b[t], bbs_b[t], bbws_b[t], bbhs_b[t], xps_b[t], yps_b[t], a_b[t], rrs_b[t])
#     p2 = plot_frame_with_risk(sss_r[t], bbs_r[t], bbws_r[t], bbhs_r[t], xps_r[t], yps_r[t], a_r[t], rrs_r[t])

#     plot(p1, p2, size=(600, 300))
# end

# anim = @animate for t = 1:50
#     get_gif_frame(sss_b, bbs_b, bbws_b, bbhs_b, xps_b, yps_b, enc_b.a, relative_risks_b,
#                   sss_r, bbs_r, bbws_r, bbhs_r, xps_r, yps_r, enc_r.a, relative_risks_r,
#                   t)
# end
# Plots.gif(anim, "collision_avoidance/figures/risk_comparison_ex2.gif", fps=2)