struct FunPolicy <: Policy
    f
end

Crux.action_space(p::FunPolicy) = ContinuousSpace(1)

function POMDPs.action(p::FunPolicy, s)
    return p.f(s)
end

function discrete_rule(s)
    return (-8 / π) * s[1] < s[2] ? -1.0 : 1.0
end

function continuous_rule(k1, k2, k3)
     (s) -> begin
        ωtarget = sign(s[1])*sqrt(6*10*(1-cos(s[1])))
        # 
        # I = 1/3
        # dt = 0.05
        # τ = k*I*(ωtarget - s[2]) / dt
        # return  clamp(τ, -1, 1)
        # clamp(-k1*s[1] - k2*s[2] + k3*(ωtarget - s[2]), -2.5, 2.5)
        -k1*s[1] - k2*s[2] + k3*(ωtarget - s[2])
    end
end

