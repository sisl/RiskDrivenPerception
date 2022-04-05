using POMDPs, POMDPGym, Crux, Flux, Distributions, Plots, BSON, GridInterpolations
include("../src/risk_solvers.jl")

# Load the environment and policy
s0 = [0.0, 0.0, 0.0, 40.0]
env = CollisionAvoidanceMDP(h0_dist=Uniform(s0[1] - 1e-16, s0[1] + 1e-16),
    dh0_dist=Uniform(s0[2] - 1e-16, s0[2] + 1e-16))

# hs = range(-200, 200, length=21)
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
scatter(hs, zeros(length(hs)))
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)

# Plot a slice of the policy
heatmap(τs, hs, (τ, h) -> action(policy, [h, 0.0, 0.0, τ]))

# Set up the cost function and risk mdp
# costfn(m, s, sp) = isterminal(m, sp) && (-50.0 < s[1] < 50.0) ? 1.0 : 0.0
costfn(m, s, sp) = isterminal(m, sp) ? 900 - abs(s[1]) : 0.0
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :noise)

# Set up the noise distribution
ϵh = Normal(0, 10)
ϵτ = Normal(0, 3)

noises_h = collect(range(-20, stop=20, length=11))
noises_τ = collect(range(-6, stop=6, length=11))

ϵ_grid = RectangleGrid(noises_h, noises_τ)
noises = [[ϵ[1]; 0.0; 0.0; ϵ[2]] for ϵ in ϵ_grid]
probs = [pdf(ϵh, ϵ[1]) * pdf(ϵτ, ϵ[2]) for ϵ in ϵ_grid]
probs ./= sum(probs)

px = DistributionPolicy(ObjectCategorical(noises, probs))

# Get the distribution of returns and plot
N = 1000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]

p1 = histogram(samples, title="CAS Costs", bins=range(0, 900, 50), normalize=true, alpha=0.3)

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 900, 50))
s_grid = RectangleGrid(hs, dhs, env.actions, τs)
𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in env.actions, τ in τs];
s2pt(s) = s

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px.distribution, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);

# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt(s0))
si = si[argmax(wi)]

p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 900, 50), normalize=true, alpha=0.4)

# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)

# Plot one sample
heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0, 0], 0), title="α = 0")

anim = @animate for α in range(-1.0, 1.0, length=51)
    heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0, 0], α), title="CVaR (α = $α)", clims=(400, 900), xlabel="τ (s)", ylabel="h (m)")
end
Plots.gif(anim, "collision_avoidance/figures/CVaR.gif", fps=6)

# Heatmap over noise
heatmap(noises_τ, noises_h, (x, y) -> CVaR([0.0, 0.0, 0.0, 30.0], [y, x], 0.0), xlabel="ϵτ", ylabel="ϵh")

# Most important states
riskmin(x; α) = minimum([CVaR(x, [ϵh, ϵτ], α) for ϵh in noises_h for ϵτ in noises_τ])
riskmax(x; α) = maximum([CVaR(x, [ϵh, ϵτ], α) for ϵh in noises_h for ϵτ in noises_τ])
risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)

heatmap(τs, hs, (x, y) -> risk_weight([y, 0.0, 0.0, x], α=0.0), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors")

τs

# Debugging

# Uw[si]
# sum(Uw[si])

# plot(p1, p2)

# function sim_rmdp(s0, noise)
#     curr_s = s0
#     hs = zeros(41)
#     hs[1] = s0[1]
#     for i = 1:40
#         curr_s, r = gen(rmdp, curr_s, noise)
#         hs[i+1] = curr_s[1]
#     end
#     return hs
# end

# hs = sim_rmdp(s0, [0, 0, 0, 0])
# 900 - abs(hs[end])

# function get_dist_gen(s0, noise; nsamps=100)
#     costs = zeros(nsamps)
#     for i = 1:nsamps
#         hs = sim_rmdp(s0, noise)
#         costs[i] = 900 - abs(hs[end])
#     end
#     return costs
# end

# cgen = get_dist_gen(s0, [0, 0, 0, 0], nsamps=1000)
# p3 = histogram(cgen, bins=range(0, 900, 50))
# plot(p1, p3)

# function sim_rmdp_trans(s0, noise)
#     curr_s = s0
#     hs = zeros(41)
#     hs[1] = s0[1]
#     for i = 1:40
#         t = transition(rmdp, curr_s, noise)
#         curr_s = rand(t)
#         hs[i+1] = curr_s[1]
#     end
#     return hs
# end

# function get_dist_trans(s0, noise; nsamps=100)
#     costs = zeros(nsamps)
#     for i = 1:nsamps
#         hs = sim_rmdp_trans(s0, noise)
#         costs[i] = 900 - abs(hs[end])
#     end
#     return costs
# end

# ctrans = get_dist_trans(s0, [0, 0, 0, 0], nsamps=1000)
# p4 = histogram(ctrans, bins=range(0, 900, 50))
# plot(p3, p4)

# function sim_rmdp_trans_interp(s0, noise)
#     curr_s = s0
#     hs = zeros(41)
#     hs[1] = s0[1]
#     for i = 1:40
#         t = transition(rmdp, curr_s, noise)
#         s′ = rand(t)
#         sis, sps = GridInterpolations.interpolants(s_grid, s2pt(s′))
#         ind = rand(Categorical(sps))
#         curr_si = sis[ind]
#         curr_s = ind2x(s_grid, curr_si)
#         hs[i+1] = curr_s[1]
#     end
#     return hs
# end

# function get_dist_trans_interp(s0, noise; nsamps=100)
#     costs = zeros(nsamps)
#     for i = 1:nsamps
#         hs = sim_rmdp_trans_interp(s0, noise)
#         costs[i] = 900 - abs(hs[end])
#     end
#     return costs
# end

# ctransinterp = get_dist_trans_interp(s0, [0, 0, 0, 0], nsamps=1000)
# p5 = histogram(ctransinterp, bins=range(0, 900, 50))