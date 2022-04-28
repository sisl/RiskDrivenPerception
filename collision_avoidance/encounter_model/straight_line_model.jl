using Distributions, Rotations, LinearAlgebra

struct EncounterDescription
    v0 # ownship horizontal speed
    v1 # intruder horizontal speed
    hmd # horizontal miss distance (m)
    vmd # vertical miss displacement (m)
    θcpa # relative heading at closest point of approach (degrees)
    tcpa # time into encounter closest point of approach occurs (seconds)
    ttot # total time of the encounter (seconds)
end

struct Encounter
    x0::Vector # array of ownship x-values for each time step
    y0::Vector # array of ownship y-values for each time step
    z0::Vector # array of ownship z-values for each time step (will all be same value)
    v0::Vector # array of ownship horizontal speeds for each time step (will all be same value)
    dh0::Vector # array of ownship vertical rates for each time step (will all zeros)
    θ0::Vector # array of ownship heading
    x1::Vector # array of intruder x-values for each time step
    y1::Vector # array of intruder y-values for each time step
    z1::Vector # array of intruder z-values for each time step (will all be same value)
    v1::Vector # array of intruder horizontal speeds for each time step (will all be same value)
    dh1::Vector # array of intruder vertical rates for each time step (will all zeros)
    θ1::Vector # array of intruder heading
    a::Vector # array of advisories
end

Encounter(s0s, s1s, as = zeros(length(s0s))) = Encounter([e.x for e in s0s],
                                [e.y for e in s0s],
                                [e.z for e in s0s],
                                [e.v for e in s0s],
                                [e.dh for e in s0s],
                                [e.θ for e in s0s],
                                [e.x for e in s1s],
                                [e.y for e in s1s],
                                [e.z for e in s1s],
                                [e.v for e in s1s],
                                [e.dh for e in s1s],
                                [e.θ for e in s1s], as)

function get_ownship_state(enc::Encounter, index)
    (x=enc.x0[index], y=enc.y0[index], z=enc.z0[index], v=enc.v0[index], dh=enc.dh0[index], θ=enc.θ0[index])
end

function get_intruder_state(enc::Encounter, index)
    (x=enc.x1[index], y=enc.y1[index], z=enc.z1[index], v=enc.v1[index], dh=enc.dh1[index], θ=enc.θ1[index])
end



function sampler(; v0_dist=Uniform(45, 55),
    v1_dist=Uniform(45, 55),
    hmd_dist=Uniform(0, 200),
    vmd_disp=Uniform(-50, 50),
    θcpa_dist=Uniform(100, 260),
    tcpa=40,
    ttot=50)

    return EncounterDescription(rand(v0_dist),
        rand(v1_dist),
        rand(hmd_dist),
        rand(vmd_disp),
        rand(θcpa_dist),
        tcpa,
        ttot)
end

function get_encounter_states(enc_des::EncounterDescription; dt = 1)::Encounter
    """
    Inputs: 
        - enc_des (EncounterDescription): properties of the encounter

    Outputs:
        - enc (Encounter): resulting encounter
    """
    num_steps::Int = enc_des.ttot ÷ dt


    # Assumption : Ownship starts at origin at "closest point of approach" time
    ownship_2d_cpa = [0.0, 0.0]
    intruder_2d_cpa = [ownship_2d_cpa[1] + enc_des.hmd * -sind(enc_des.θcpa),
                    ownship_2d_cpa[2] + enc_des.hmd * cosd(enc_des.θcpa)]
                    

    # Assumption: Default "forward" heading is along y-axis
    ownship_heading = [0.0, 1.0]

    # Assumption: Intruder Ship's heading points directly along relative heading vector
    intruder_heading = normalize(intruder_2d_cpa - ownship_2d_cpa)
    # intruder_heading = [sind(enc_des.θcpa), cosd(enc_des.θcpa)] # Check: negative cos?

    ownship_vel = ownship_heading * enc_des.v0 * dt
    intruder_vel = intruder_heading * enc_des.v1 * dt

    timestep_cpa = enc_des.tcpa ÷ dt

    # Warning: Ships may end up getting closer after "closest point of approach", but we assume we don't care
    ownship_2d = [ownship_2d_cpa + i * ownship_vel for i in -timestep_cpa + 1:(num_steps-timestep_cpa)]
    intruder_2d = [intruder_2d_cpa + i * intruder_vel for i in -timestep_cpa + 1:(num_steps-timestep_cpa)]

    @assert size(ownship_2d)[1] == num_steps
    @assert size(intruder_2d)[1] == num_steps
    @assert ownship_2d[timestep_cpa] == ownship_2d_cpa
    @assert intruder_2d[timestep_cpa] == intruder_2d_cpa

    x0 = [p[1] for p in ownship_2d]
    y0 = [p[2] for p in ownship_2d]
    z0 = zeros(num_steps)
    θ0 = zeros(num_steps)

    v0 = fill(enc_des.v0, num_steps)
    dh0 = zeros(num_steps)

    x1 = [p[1] for p in intruder_2d]
    y1 = [p[2] for p in intruder_2d]
    z1 = fill(enc_des.vmd, num_steps)
    θ1 = fill(enc_des.θcpa, num_steps)

    v1 = fill(enc_des.v1, num_steps)
    dh1 = zeros(num_steps)
    
    a = zeros(num_steps)

    return Encounter(
        x0, y0, z0, v0, dh0, θ0, x1, y1, z1, v1, dh1, θ1, a
    )
end

function get_encounter_set(get_sample_feats, nencs)
    """
    Inputs:
        - get_sample_feats (function): function to sample encounter features (return EncounterDescription)
        - nencs: Number of encounters to generate

    Outputs:
        - encs: vector of encounters
    """

    encs = [get_encounter_states(get_sample_feats()) for _ in 1:nencs]
    return encs
end

