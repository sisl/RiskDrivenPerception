function rmdp_pendulum_setup(env, policy; Nnoise_half=5, noise_dist=Normal(0, 0.5), Nt=20, Ncost=100, Nθ=25, Nω=25)
	# Create RMDP
	tmax = Nt * env.dt
	costfn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
	rmdp = RMDP(env, policy, costfn, true, env.dt, tmax, :noise)

	# Set the nominal distribution of noise
	noises_1_half = 10 .^ (collect(range(log10(.2*std(noise_dist)), stop = log10(2*std(noise_dist)), length = Nnoise_half)))
	noises_1 = [reverse(-noises_1_half); 0.0; noises_1_half]
	probs_1 = [pdf(noise_dist, n) for n in noises_1]
	probs_1 ./= sum(probs_1)
	noises_2_half = 10 .^ (collect(range(log10(.2*std(noise_dist)), stop = log10(2*std(noise_dist)), length = Nnoise_half)))
	noises_2 = [reverse(-noises_2_half); 0.0; noises_2_half]
	probs_2 = [pdf(noise_dist, n) for n in noises_2]
	probs_2 ./= sum(probs_2)
	noises = [[n1, n2] for n1 in noises_1 for n2 in noises_2]
	probs = [p1 * p2 for p1 in probs_1 for p2 in probs_2]
	px = DistributionPolicy(ObjectCategorical(noises, probs))

	# Define the grid for interpolation
	θs_half = π .- sqrt.(collect(range(0, stop = π^2, length = Nθ)))
	θs = [-θs_half[1:end-1]; reverse(θs_half)]
	
	ωs_half = 8 .- sqrt.(collect(range(0, stop = 8^2, length = Nω)))
	ωs = [-ωs_half[1:end-1]; reverse(ωs_half)]

	ts = 0:env.dt:tmax
	state_grid = RectangleGrid(θs, ωs, ts)

	# Define the state space and mapping to the grid
	𝒮 = [[tmax - t, θ, ω] for θ in θs, ω in ωs, t in ts]
	s2pt(s) = [s[2:end]..., tmax - s[1]]

	# cost_points = range(0, stop = π, length = 100)
	cost_points = reverse(π .- (collect(range(0, stop = π^(1/0.8), length=Ncost))).^0.8)
	
	ϵ_grid = RectangleGrid(noises_1, noises_2)
	
	rmdp, px, θs, ωs, state_grid, 𝒮, s2pt, cost_points, noises_1, noises_2, ϵ_grid
end


# function costfn(m, s, sp)
#     if isterminal(m, sp)
#         if abs(s[2]) ≤ deg2rad(15) && abs(s[3]) ≤ deg2rad(5)
#             return 0.0
#         else
#             return abs(s[2]) + s[3]^2
#         end
#     else
#         return 0.0
#     end
# end
# 
# function failure_costfn(m, s, sp)
#     if isterminal(m, sp)
#         if abs(s[2]) ≤ deg2rad(15) && abs(s[3]) ≤ deg2rad(5)
#             return 0.0
#         else
#             return 1.0
#         end
#     end
# end
