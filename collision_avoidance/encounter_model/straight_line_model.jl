using Distributions

struct EncounterDescription
    v0 # ownship horizontal speed
    v1 # intruder horizontal speed
    hmd # horizontal miss distance (m)
    vmd # vertical miss distance (m)
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
    x1::Vector # array of intruder x-values for each time step
    y1::Vector # array of intruder y-values for each time step
    z1::Vector # array of intruder z-values for each time step (will all be same value)
    v1::Vector # array of intruder horizontal speeds for each time step (will all be same value)
    dh1::Vector # array of intruder vertical rates for each time step (will all zeros)
end

function sampler(; v0_dist=Uniform(45, 55),
    v1_dist=Uniform(45, 55),
    hmd_dist=Uniform(-200, 200),
    vmd_dist=Uniform(-50, 50),
    θcpa_dist=Uniform(0, 360),
    tcpa=40,
    ttot=50)

    return EncounterDescription(rand(v0_dist),
        rand(v1_dist),
        rand(hmd_dist),
        rand(vmd_dist),
        rand(θcpa_dist),
        tcpa,
        ttot)
end

function get_encounter_states(enc_des::EncounterDescription; dt=1)
    """
    Inputs: 
        - enc_des (EncounterDescription): properties of the encounter

    Outputs:
        - enc (Encounter): resulting encounter
    """

    # TODO: fill in!!!!
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