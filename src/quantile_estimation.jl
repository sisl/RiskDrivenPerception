using Plots, Flux

x = rand(1,1000)
y = rand(1,1000)
data = Flux.DataLoader((x,y), batchsize=128,)

model = Dense(1,1, init=zeros)
opt = ADAM(1e-3)
θ = Flux.params(model)

ρ(τ) = (u) -> τ*max(u, 0) + (1-τ)*max(-u, 0)
ρ_huber(τ) = (u) -> u*(τ - u>0)



plot(-1:0.01:1, ρ(0.3))
plot(-1:0.01:1, ρ_huber(0.3))
 
quantile_loss(ρ) = (x,y) -> Flux.mean(ρ.(y .- model(x)))


p = scatter(x[:], y[:])
for τ in 0:0.1:1
	Flux.@epochs 100 Flux.train!(quantile_loss(ρ(τ)), Flux.params(model), data, opt)
	plot!(0:0.1:1, (x)->model([x])[1], label="τ=$τ")
end
p

