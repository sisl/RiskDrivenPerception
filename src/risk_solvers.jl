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
        println("s index: $si, $s")
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

