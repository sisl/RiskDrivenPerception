using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
using DataFrames, LinearAlgebra
using StatsBase
include("../src/risk_solvers.jl")

# Load the environment and policy
s0 = [0.0, 0.0, 0.0, 40.0]
env = DetectAndAvoidMDP(h0_dist=Uniform(s0[1] - 1e-16, s0[1] + 1e-16),
    dh0_dist=Uniform(s0[2] - 1e-16, s0[2] + 1e-16), ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
    actions=[-8.0, 0.0, 8.0])

hmax = 300
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
scatter(hs, zeros(length(hs)))

dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

policy = OptimalDetectAndAvoidPolicy(env, hs, dhs, τs)

# Plot a slice of the policy
heatmap(τs, hs, (τ, h) -> action(policy, [h, 0.0, 0.0, τ]), xlabel="τ (s)", ylabel="h (m)", title="CAS Policy")

# Set up the cost function and risk mdp
costfn(m, s, sp) = isterminal(m, sp) ? 150 - abs(s[1]) : 0.0
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

# Start with just detect noise
p_detect(s) = sigmoid(-0.006518117 * abs(s[1]) - 0.10433467s[4] + 1.2849158)
function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    return ObjectCategorical(noises, [1 - pd, pd])
end

#detect_prob = 0.9
noises_detect = [0, 1]
#probs_detect = [1.0 - detect_prob, detect_prob]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0, 0.0, 0.0] for ϵ in ϵ_grid]
#probs = probs_detect # NOTE: will need to change once also have additive noise

#px = DistributionPolicy(ObjectCategorical(noises, probs))
px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

# Get the distribution of returns and plot
N = 1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, title="CAS Costs", bins=range(0, 150, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 150, 50))
s_grid = RectangleGrid(hs, dhs, env.actions, τs)
𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs];
s2pt(s) = s

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);

# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt(s0))
si = si[argmax(wi)]

p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 200, 50), normalize=true, alpha=0.4, label="DP")

# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# Plot one sample
heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0], 0.0), title="α = 0")

anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="τ (s)", ylabel="h (m)")
end
Plots.gif(anim, "collision_avoidance/figures/daa_CVaR_v3.gif", fps=6)

# Most important states
riskmin(x; α) = minimum([CVaR(x, [noise], α) for noise in noises_detect])
riskmax(x; α) = maximum([CVaR(x, [noise], α) for noise in noises_detect])
risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)

heatmap(0:0.5:40, -200:5:200, (x, y) -> risk_weight([y, 0.0, 0.0, x], α=0.0), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors")#, clims=(0, 20))

anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(0:0.5:40, -200:5:200, (x, y) -> risk_weight([y, 0.0, 0.0, x], α=α), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors: α = $(α)", clims=(0, 20))
end
Plots.gif(anim, "collision_avoidance/figures/daa_risk_weights.gif", fps=6)

# # Marginal Risk Weights
# marginal_risk_weight(h, τ; α) = sum([risk_weight([h, dh, a_prev, τ], α=α) for dh in dhs for a_prev in env.actions]) / (length(dhs) * length(env.actions))
# heatmap(0:0.5:40, -300:5:300, (x, y) -> marginal_risk_weight(y, x, α=0.0), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors")

# anim = @animate for α in range(-1.0, 1.0, length=21)
#     heatmap(0:0.5:40, -300:5:300, (x, y) -> marginal_risk_weight(y, x, α=α), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors: α = $(α)", clims=(0, 15))
# end
# Plots.gif(anim, "collision_avoidance/figures/daa_marginal_risk_weights.gif", fps=3)

# Get nominal distribution of dh and a_prev for marginal risk weight
env = DetectAndAvoidMDP(ddh_max=1.0, px=DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1]),
    actions=[-8.0, 0.0, 8.0])

N = 10000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

dh_samps = D[:s][2, :]
a_prev_samps = D[:s][3, :]

histogram(dh_samps)
histogram(a_prev_samps)

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

# Marginal CVaR (done right?)
function marginal_CVaR(h, τ, dhs, a_prevs, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)
    w = zeros(length(cost_points))
    for (i, dh) in enumerate(dhs)
        for (j, a_prev) in enumerate(a_prevs)
            s = [h, dh, a_prev, τ]
            sis, sws = interpolants(s_grid, s)
            ϵis, ϵws = interpolants(ϵ_grid, ϵ)
            for (si, sw) in zip(sis, sws)
                for (ϵi, ϵw) in zip(ϵis, ϵws)
                    w .+= wdh[i] * wa_prev[j] * sw * ϵw .* Qw[ϵi][si]
                end
            end
        end
    end
    #w ./= (length(dhs) * length(a_prevs))

    if α == 0
        return w' * cost_points#, 0.0
    else
        return cvar_categorical(cost_points, w, α=α)[1]
    end
end

marginal_CVaR(h, τ, ϵ, α) = marginal_CVaR(h, τ, dhs, env.actions, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)
# Most important states marginal
marginal_riskmin(h, τ; α) = minimum([marginal_CVaR(h, τ, [noise], α) for noise in noises_detect])
marginal_riskmax(h, τ; α) = maximum([marginal_CVaR(h, τ, [noise], α) for noise in noises_detect])
marginal_risk_weight(h, τ; α) = marginal_riskmax(h, τ; α) - marginal_riskmin(h, τ; α)

heatmap(0:0.5:40, -300:5:300, (x, y) -> marginal_risk_weight(y, x, α=0.0), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors")

anim = @animate for α in range(-1.0, 1.0, length=21)
    heatmap(0:0.5:40, -300:5:300, (x, y) -> marginal_risk_weight(y, x, α=α), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors: α = $(α)", clims=(0, 15))
end
Plots.gif(anim, "collision_avoidance/figures/daa_marginal_cvar_risk_weights_v2.gif", fps=3)

# Sampling distribution for risk-driven data generation
function get_intruder_position(e0, n0, u0, h0, z, hang, vang)
    e1 = z * tand(hang)
    n1 = z
    u1 = z * tand(vang)

    # Rotate
    n1 = (z / cosd(hang)) * cosd(h0 + hang)
    e1 = (z / cosd(hang)) * sind(h0 + hang)

    # Translate
    e1 += e0
    n1 += n0
    u1 += u0

    return e1, n1, u1
end

function sample_random_state()
    # Ownship state
    e0 = rand(Uniform(-5000.0, 5000.0))  # m
    n0 = rand(Uniform(-5000.0, 5000.0))  # m
    u0 = rand(Uniform(-500.0, 500.0))  # m
    h0 = rand(Uniform(0.0, 360.0))  # degrees

    # Info about relative position of intruder
    vang = rand(Uniform(-25.0, 25.0)) # degrees
    hang = rand(Uniform(-38.0, 38.0))  # degrees
    z = rand(Uniform(20, 2000))  # meters

    # Intruder state
    e1, n1, u1 = get_intruder_position(e0, n0, u0, h0, z, hang, vang)
    h1 = rand(Uniform(0.0, 360.0))  # degrees

    return e0, n0, u0, h0, vang, hang, z, e1, n1, u1, h1
end

const HNMAC = 100

function mdp_state(e0, n0, e1, n1; v_dist=Uniform(45, 55), θ_dist=Uniform(120, 240))
    h = n0 - n1

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

    h, τ
end

function rejection_sample_states(N; baseline=0.2, α=0.0)
    # Store samples in dataframe
    samples = DataFrame(e0=Float64[], n0=Float64[], u0=Float64[], h0=Float64[],
        vang=Float64[], hang=Float64[], z=Float64[],
        e1=Float64[], n1=Float64[], u1=Float64[], h1=Float64[],
        h=Float64[], τ=Float64[])

    ind = 1
    while ind ≤ N
        e0, n0, u0, h0, vang, hang, z, e1, n1, u1, h1 = sample_random_state()
        h, τ = mdp_state(e0, n0, e1, n1)
        rw = marginal_risk_weight(h, τ, α=α) / 15
        if rand() < rw + baseline
            # Store the sample
            push!(samples, [e0, n0, u0, h0, vang, hang, z, e1, n1, u1, h1, h, τ])
            ind += 1
        end
        ind % 500 == 0 ? println(ind) : nothing
    end

    return samples
end

samples = rejection_sample_states(10000, baseline=0.01)

hsamps = samples[:, :h]
τsamps = samples[:, :τ]
histogram2d(τsamps, hsamps, ylims=(-300, 300), bins=(0:1:40, -300:10:300), xlabel="τ (s)", ylabel="h (m)", title="Density of Sampled States")

hranges = sqrt.((samples[:, :e0] .- samples[:, :e1]) .^ 2 .+ (samples[:, :n0] .- samples[:, :n1]) .^ 2)
histogram(hranges, xlabel="Horizontal Range (m)", title="Ranges of Sampled States")