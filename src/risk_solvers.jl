function solve_conditional_bellman(mdp, pa, rcondition, grid, 𝒮, s2pt)
    as = support(pa)
    ps = pa.p
    
    U = zeros(length(𝒮)) # Values
    Q = [zeros(length(𝒮)) for a in as] # state-ation values

    # Solve with backwards induction value iteration
    for (si, s) in enumerate(𝒮)
        for (ai, a) in enumerate(as)
            s′, r = gen(mdp, s, a)
            Q[ai][si] = rcondition(r)
            Q[ai][si] += isterminal(mdp, s′) ? 0.0 : GridInterpolations.interpolate(grid, U, s2pt(s′))
        end
        U[si] = sum(p*q[si] for (q, p) in zip(Q, ps))
    end
    Q
end

function solve_cvar_particle(mdp, pa, grid, 𝒮, s2pt)
    as = support(pa) # TODO: remove for state-dep disturbance model
    ps = pa.p # TODO: remove for state-dep disturbance model
    
    Up = [Float64[] for i=1:length(𝒮)] # Values
    Uw = [Float64[] for i=1:length(𝒮)] # Values
    Qp = [[Float64[] for i=1:length(𝒮)] for a in as] # state-ation values
    Qw = [[Float64[] for i=1:length(𝒮)] for a in as] # state-ation values

    # Solve with backwards induction value iteration
    for (si, s) in enumerate(𝒮)
        for (ai, a) in enumerate(as)
            s′, r = gen(mdp, s, a)
            if isterminal(mdp, s′)
                println("r: ", r)
                push!(Qp[ai][si], r)
                push!(Qw[ai][si], 1.0)
                
            else
                s′i, s′w = GridInterpolations.interpolants(grid, s2pt(s′))
                s′i = s′i[argmax(s′w)]
                # for (i, w) in zip(s′i, s′w)
                push!(Qp[ai][si], Up[s′i]...)
                push!(Qw[ai][si], Uw[s′i]...)
                # end
            end
        end
        for ai in 1:length(as)
            push!(Up[si], Qp[ai][si]...)
            push!(Uw[si], ps[ai] .* Qw[ai][si]...) # TODO: Replace ps with pa(s) for state-dependent disturbance model
        end
    end
    Qp, Qw
end

function solve_cvar_fixed_particle(rmdp, pa, grid, 𝒮, s2pt, cost_points; mdp_type=:gen, ngen=1)
    # as = support(pa)
    # ps = pa.p
    as = action_space(pa).vals
    N = length(cost_points)
    cost_grid = RectangleGrid(cost_points)

    Uw = [zeros(N) for i = 1:length(𝒮)] # Values
    Qw = [[zeros(N) for i = 1:length(𝒮)] for a in as] # state-ation values

    # Solve with backwards induction value iteration
    for (si, s) in enumerate(𝒮)
        a_dist = pa.pa(s)
        as = support(a_dist)
        ps = a_dist.p
        # si % 1000 == 0 ? println(si) : nothing
        for (ai, a) in enumerate(as)
            if mdp_type == :gen
                q_ai_si_gen!(Qw, Uw, rmdp, ai, a, si, s, grid, cost_grid; ngen)
            elseif mdp_type == :exp
                q_ai_si_exp!(Qw, Uw, rmdp, ai, a, si, s, grid, cost_grid)
            else
                error("Invalid MDP type")
            end
        end
        for ai in 1:length(as)
            Uw[si] .+= ps[ai] .* Qw[ai][si]
        end
    end
    Uw, Qw
end

function q_ai_si_exp!(Qw, Uw, rmdp, ai, a, si, s, grid, cost_grid)
    t = transition(rmdp, s, a)
    for (s′, p) in t
        if isterminal(rmdp, s′)
            r = reward(rmdp, s, s′)
            ris, rps = interpolants(cost_grid, [r])
            for (ri, rp) in zip(ris, rps)
                Qw[ai][si][ri] += p * rp
            end
        else
            s′i, s′w = GridInterpolations.interpolants(grid, s2pt(s′))
            for (i, w) in zip(s′i, s′w)
                Qw[ai][si] .+= p * w .* Uw[i]
            end
        end
    end
end

function q_ai_si_gen!(Qw, Uw, rmdp, ai, a, si, s, grid, cost_grid; ngen)
    for i = 1:ngen
        s′, r = gen(rmdp, s, a)
        if isterminal(rmdp, s′)
            ris, rps = interpolants(cost_grid, [r])
            for (ri, rp) in zip(ris, rps)
                Qw[ai][si][ri] += (1 / ngen) * rp
            end
        else
            s′i, s′w = GridInterpolations.interpolants(grid, s2pt(s′))
            for (i, w) in zip(s′i, s′w)
                Qw[ai][si] .+= (1 / ngen) * w .* Uw[i]
            end
        end
    end
end

function ECVaR(s, s_grid, ϵ_grid, Qw, cost_points, px; α)
    # Get all ρs
    ρϵs = zeros(length(px.distribution.objs))
    for (i, ep) in enumerate(px.distribution.objs)
        ρϵs[i] = CVaR(s, ep, s_grid, ϵ_grid, Qw, cost_points, α = α)[1]
    end
    normalizer = ρϵs' * px.distribution.p
end

function normalized_CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points, px; α, normalizer=ECVaR(s, s_grid, ϵ_grid, Qw, cost_points, px; α))
    ρ_curr = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points, α = α)[1]
    if ρ_curr == 0 && normalizer==0
        return 0f0
    elseif normalizer==0
        println("Error! only normalizer was zero")
    else
        return ρ_curr / normalizer
    end
end


function CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α)
    w = zeros(length(cost_points))
    sis, sws = interpolants(s_grid, s)
    ϵis, ϵws = interpolants(ϵ_grid, ϵ)
    for (si, sw) in zip(sis, sws)
        for (ϵi, ϵw) in zip(ϵis, ϵws)
            w .+= sw * ϵw .* Qw[ϵi][si]
        end
    end

    if α == 0
        return w' * cost_points#, 0.0
    else
        return cvar_categorical(cost_points, w, α = α)[1]
    end
end

function cvar_categorical(xs, ws; α = 0.95)
    perm = α > 0 ? sortperm(xs, rev = true) : sortperm(xs) # descending/ascending order
    xs = xs[perm]
    ws = ws[perm]
    partial_ws = cumsum(ws)
    # Should it be searchsortedfirst or last?
    idx = α > 0 ? findfirst(partial_ws .> 1 - α) : findfirst(partial_ws .> 1 + α)

    if isnothing(idx)
        idx = 1
    end

    cvar_xs = xs[1:idx]
    cvar_ws = ws[1:idx]
    cvar_ws ./= sum(cvar_ws)

    cvar = cvar_ws' * cvar_xs

    return cvar, xs[idx]
end

