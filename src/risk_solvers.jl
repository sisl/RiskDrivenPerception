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

