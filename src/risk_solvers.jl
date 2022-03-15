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
    as = support(pa)
    ps = pa.p
    
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
            push!(Uw[si], ps[ai] .* Qw[ai][si]...)
        end
    end
    Qp, Qw
end

function solve_cvar_fixed_particle(mdp, pa, grid, 𝒮, s2pt, cost_points)
    as = support(pa)
    ps = pa.p
    N = length(cost_points)
    cost_grid = RectangleGrid(cost_points)

    Uw = [zeros(N) for i = 1:length(𝒮)] # Values
    Qw = [[zeros(N) for i = 1:length(𝒮)] for a in as] # state-ation values

    # Solve with backwards induction value iteration
    for (si, s) in enumerate(𝒮)
        for (ai, a) in enumerate(as)
            s′, r = gen(mdp, s, a)
            if isterminal(mdp, s′)
                ris, rps = interpolants(cost_grid, [r])
                for (ri, rp) in zip(ris, rps)
                    Qw[ai][si][ri] = rp
                end
            else
                s′i, s′w = GridInterpolations.interpolants(grid, s2pt(s′))
                # s′i = s′i[argmax(s′w)]
                for (i, w) in zip(s′i, s′w)
                    Qw[ai][si] .+= w .* Uw[i]
                end
            end
        end
        for ai in 1:length(as)
            Uw[si] .+= ps[ai] .* Qw[ai][si]
        end
    end
    Qw
end

function ρ(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; α = 0.95)
    w = zeros(length(cost_points))
    sis, sws = interpolants(s_grid, s)
    ϵis, ϵws = interpolants(ϵ_grid, ϵ)
    for (si, sw) in zip(sis, sws)
        for (ϵi, ϵw) in zip(ϵis, ϵws)
            w .+= sw * ϵw .* Qw[ϵi][si]
        end
    end
    if α == 0
        return w' * cost_points, 0.0
    else
        return cvar_categorical(cost_points, w, α = α)
    end
end

function cvar_categorical(xs, ws; α = 0.95)
    perm = α > 0 ? sortperm(xs, rev = true) : sortperm(xs) # descending/ascending order
    xs = xs[perm]
    ws = ws[perm]
    partial_ws = cumsum(ws)
    idx = α > 0 ? searchsortedlast(partial_ws, 1 - α) : searchsortedlast(partial_ws, -α)

    if idx < 1
        idx = 1
    end

    cvar_xs = xs[1:idx]
    cvar_ws = ws[1:idx]
    cvar_ws ./= sum(cvar_ws)

    cvar = cvar_ws' * cvar_xs

    return cvar, xs[idx]
end
