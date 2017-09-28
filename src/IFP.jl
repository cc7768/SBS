module SBS

using BasisMatrices
using Clustering
using Optim
using QuantEcon

import QuantEcon.simulate, QuantEcon.solve

include("stable_regressions.jl")
using .StableReg


#
# Set up types for holding model and solution
#
struct IFP
    β::Float64  # Discount factor
    γ::Float64  # Risk aversion/Intertemporal elasticity of substitution
    r::Float64  # Risk-free interest rate
    ρ::Float64  # log income persistence
    σ::Float64  # standard deviation of log income

    # Nodes and weight for integration of standard normal
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function IFP(;β=0.96, γ=2.0, r=0.025, ρ=0.95, σ=0.01, n_nodes=5)
    n, w = qnwnorm(n_nodes, 0.0, 1.0)

    return IFP(β, γ, r, ρ, σ, n, w)
end


u(m::IFP, c) = ifelse(c > 1e-10, c^(1.0 - m.γ) / (1.0 - m.γ), -1e10)
du(m::IFP, c) = ifelse(c > 1e-10, c^(-m.γ), 1e10)
duinv(m::IFP, x) = x^(-1.0 / m.γ)
expendables_t(m::IFP, a, w) = (1.0 + m.r)*a + w


struct IFP_Sol
    c_a::Vector{Float64}  # Coefficients for policy function
    c_V::Vector{Float64}  # Coefficients for value function
end

function IFP_Sol()
    # Guess policy of
    # atp1 = 0.9*at + 0.15*wt
    #      1    x    x^2  xy     y    y^2
    c_a = [-1.0, 1.0, 0.0, 0.0, 0.75, 0.0]
    c_V = zeros(6)
    #      1    x    x^2  x^3  x^2y xy  xy^2  y    y^2  y^3
    # c_a = [0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0]
    # c_V = zeros(10)

    return IFP_Sol(c_a, c_V)
end


"""
Function for evaluation the value function at (a, 2) given some coefficients
for the value tomorrow and the choice of savings
"""
function eval_V!(temp::Vector{Float64}, m::IFP, pol::IFP_Sol,
                 a::Float64, w::Float64, atp1::Float64)
    # Pull out relevant information
    β, ρ, σ = m.β, m.ρ, m.σ
    nodes, weights = m.nodes, m.weights
    c_V = pol.c_V
    d = BasisMatrices.Degree{2}()

    # Compute expected value
    EV = 0.0
    for (ϵ, weight) in zip(nodes, weights)
        wtp1 = w^ρ * exp(σ*ϵ)
        EV += weight * dot(c_V, complete_polynomial!(temp, [atp1, wtp1], d))
    end

    return u(m, expendables_t(m, a, w) - atp1) + β*EV
end


"""
Takes a given policy and simulates it N times for T periods... It then stores the
Tth observation of each chain
"""
function simulate(m::IFP, policy::IFP_Sol, N::Int, T::Int,
                  a0::Float64, w0::Float64)
    # Pull out relevant information
    ρ, σ = m.ρ, m.σ
    c_a = policy.c_a

    #
    # Simulate given a policy and store last period of each simulation
    #
    temp = Array{Float64}(n_complete(2, 2))  # Hard coded two dimensions and cubic polys
    d = BasisMatrices.Degree{2}()
    sim_out = Array{Float64}(2, N)

    srand(42)  # Use same seed each iteration
    for n in 1:N
        a, w = a0, w0
        for t in 1:T
            a = max(dot(c_a, complete_polynomial!(temp, [a, w], d)), 0.0)
            w = w^ρ * exp(σ*randn())
        end

        # Store the current asset and wage
        sim_out[1, n] = a
        sim_out[2, n] = w
    end

    return sim_out
end


"""
Builds a grid using the cluster grids described in Maliar Maliar 2015

Uses the most basic format which uses a k means algorithm to classify
points into k clusters.
"""
function build_grid(m::IFP, policy::IFP_Sol, k::Int, N::Int, T::Int,
                    a0::Float64, w0::Float64)
    #
    # Simulate
    #
    sim_out = simulate(m, policy, N, T, a0, w0)

    #
    # Use simulation output to build the grid
    #
    res = kmeans(sim_out, k)

    return res.centers
end


function iterate_given_policy!(
    V::Array{Float64}, m::IFP, policy::IFP_Sol, G::Array{Float64, 2}, k::Int;
    tol=1e-5, maxiter=2500
   )

    # Initialize counters
    dist, iter = 10.0, 0
    temp = Array{Float64}(n_complete(2, 2))
    copy!(policy.c_V, zeros(n_complete(2, 2)))  # Always start guess at all zeros

    # Initialize basis matrix
    Φ = complete_polynomial(G', 2)  # Get big basis matrix
    Φ2 = view(Φ, :, 2:size(Φ, 2))

    # Initialize types to speed things up
    d = BasisMatrices.Degree{2}()
    reg = RLSSVD(intercept=true)

    while (dist > tol) & (iter < maxiter)
        for i in 1:k
            a_t, w_t = G[:, i]
            atp1 = dot(policy.c_a, complete_polynomial!(temp, [a_t, w_t], d))
            V[i] = eval_V!(temp, m, policy, a_t, w_t, atp1)
        end

        c_V_upd = regress(reg, Φ2, V)
        dist = maximum(abs, policy.c_V - c_V_upd)
        copy!(policy.c_V, c_V_upd)
    end

    return V
end


function solve(
    m::IFP, policy::IFP_Sol; δ=0.05, tol=1e-8, maxiter=2500, k=250, N=2500, T=1250
   )

    # Pull out useful info
    β, ρ, σ, r = m.β, m.ρ, m.σ, m.r
    nodes, weights = m.nodes, m.weights

    # Initialize counters
    dist, iter = 10.0, 0
    d = BasisMatrices.Degree{2}()
    reg = RLSSVD(intercept=true)

    # Allocate some array space
    G = build_grid(m, policy, k, N, T, 0.0, 1.0)
    Φ = complete_polynomial(G', 2)  # Get big basis matrix
    Φ2 = view(Φ, :, 2:size(Φ, 2))
    Astar = Array{Float64}(k)
    V = Array{Float64}(k)
    temp = Array{Float64}(n_complete(2, 2))

    # Iterate till convergence
    while (dist > tol) & (iter < maxiter)

        # Rebuild the grid
        if (dist < 1e-2) & (iter % 15 == 0)
            println("Rebuilding grid")
            ts = time()
            G = build_grid(m, policy, k, N, T, 10.0, 1.0)
            te = time() - ts
            println("\tRebuilding grid took $(round(te, 2)) seconds")
            println("\tNew grid extrema are $(extrema(G, 2))")

            # Build full basis matrix
            Φ = complete_polynomial(G', 2)
            Φ2 = view(Φ, :, 2:size(Φ, 2))
        end

        # Iterate over states
        for i_s in 1:k
            # Pull out current state
            a_t, w_t = G[:, i_s]

            # Evaluate current savings policy
            a_tp1 = dot(policy.c_a, complete_polynomial!(temp, [a_t, w_t], d))

            # Compute RHS of EE
            rhs_ee = 0.0
            for (_node, _weight) in zip(nodes, weights)
                # Tomorrow's wage and corresponding savings
                w_tp1 = w_t^ρ * exp(σ*randn())
                a_tp2 = dot(policy.c_a, complete_polynomial!(temp, [a_tp1, w_tp1], d))

                c_tp1 = expendables_t(m, a_t, w_t) - a_tp2
                rhs_ee += β * du(m, c_tp1) * (1.0 + r)
            end

            # Compute optimal savings from euler equation
            c_t = duinv(m, rhs_ee)
            a_tp1_star = expendables_t(m, a_t, w_t) - c_t
            Astar[i_s] = a_tp1_star

        end

        # Update coefficients and compute distance
        c_a_upd = regress(reg, Φ2, Astar)
        dist = maximum(abs, policy.c_a - c_a_upd)
        copy!(policy.c_a, (1.0-δ)*policy.c_a .+ δ*c_a_upd)
        iter = iter + 1
        println("After $iter the distance is $dist with coeffs $(policy.c_a)")
    end

    # Iterate on VF using the new policy
    iterate_given_policy!(V, m, policy, G, k)

    return policy
end

end

#=
# Initialize stuff
m = IFP(r=0.035)
policy = IFP_Sol()

solve(m, policy; δ=0.15, tol=1e-6, maxiter=1500, k=75, N=1500, T=750)

=#
