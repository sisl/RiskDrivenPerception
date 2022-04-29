using POMDPs, POMDPGym, Crux, Flux, Distributions, GridInterpolations, Plots
using PyCall
using Random

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
    if pyimport("sys")."path"[1] != "collision_avoidance/"
        pushfirst!(pyimport("sys")."path", "collision_avoidance/")
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

function simulate_encounter(enc::Encounter, policy; save=true, xplane_control=nothing, model=nothing)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
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
            bb, xp, yp, w, h = xplane_control.util.get_bounding_box(xplane_control.client, model, s0.x, s0.y, s0.z, s0.θ, s1.x, s1.y, s1.z, s1.θ, save_num)
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

function simulate_encounters(encs, policy, sleep_time; kwargs...)
    sleep(sleep_time)
    return [simulate_encounter(enc, policy; kwargs...) for enc in encs]
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
Random.seed!(12)
encs = get_encounter_set(sampler, 1000)
nmacs = sum([is_nmac(enc) for enc in encs])

# Rotate and shift
new_encs = rotate_and_shift_encs(encs)
nmacs = sum([is_nmac(enc) for enc in new_encs])
plot_enc(new_encs[2])

# connect to xplane
xctrl = XPlaneControl()

# Load in the perception model
model_baseline = xctrl.util.load_model("collision_avoidance/models/traffic_detector_v3.pt")
model_risk_data = xctrl.util.load_model("collision_avoidance/models/traffic_detector_risk_data_v2.pt")

# Simulate them
sim_encs_gt = simulate_encounters(new_encs, policy, 0.0, save=false);
@time sim_encs_baseline = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=model_baseline);
@time sim_encs_risk_data = simulate_encounters(new_encs, policy, 2.0, save=false, xplane_control=xctrl, model=model_risk_data);

# Count the number of nmacs
nmacs_gt = sum([is_nmac(sim_enc) for sim_enc in sim_encs_gt])
nmacs_baseline = sum([is_nmac(sim_enc) for sim_enc in sim_encs_baseline])
nmacs_risk_data = sum([is_nmac(sim_enc) for sim_enc in sim_encs_risk_data])

# Count the number of alerts
alerts_gt = sum([sum(enc.a .!= 0.0) for enc in sim_encs_gt])
alerts_baseline = sum([sum(enc.a .!= 0.0) for enc in sim_encs_baseline])
alerts_risk_data = sum([sum(enc.a .!= 0.0) for enc in sim_encs_risk_data])

alert_rate_gt = alerts_gt / 41000
alert_rate_baseline = alerts_baseline / 41000
alert_rate_risk_data = alerts_risk_data / 41000


plot_enc(sim_encs_baseline[1])