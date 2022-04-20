using POMDPs, POMDPGym, Crux, Flux, Distributions, GridInterpolations

include("encounter_model/straight_line_model.jl")

function simulate_encounter(enc::Encounter, policy)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """

    # TODO: fill this in!!

    # At each time step
    # Get current positions and speeds of ownship and intruder
    # Use this to calculate mdp state
    # Get action from policy
    # Adjust ownship state accordingly
end

simulate_encounters(encs, policy) = [simulate_encounter(enc, policy) for enc in encs]

function is_nmac(enc::Encounter)
    for i = 1:length(enc.x0)
        hsep = sqrt((enc.x0[i] - enc.x1[i])^2 + (enc.y0[i] - enc.y1[i])^2)
        vsep = abs(enc.z0[i] - enc.z1[i])
        if hsep < 100 && vsep < 50
            return true
        end
    end
    return false
end

# Create environment
env = CollisionAvoidanceMDP()
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)

# Get 100 encounters
encs = get_encounter_set(sampler, nencs)

# Simulate them
sim_encs = simulate_encounters(encs, policy)

# Count the number of nmacs
nmacs = sum([is_nmac(sim_enc) for sim_enc in sim_encs])