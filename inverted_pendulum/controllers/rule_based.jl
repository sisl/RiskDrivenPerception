struct FunPolicy <: Policy
    f
end

function POMDPs.action(p::FunPolicy, s)
    return f(s)
end

function f(s)
    return (-8 / π) * s[1] < s[2] ? -1.0 : 1.0
end