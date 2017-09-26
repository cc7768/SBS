"""
A collection of various stable way to do a linear regression

#### References

Sections 3 and 4 of "Numerically stable and accurate stochastic simulation
approaches for solving dynamic economic models."

Kenneth L Judd, Serguei Maliar, and Lilia Maliar.

Author: Spencer Lyon
Date: July 2016

"""
module StableReg

using Parameters
using MathProgBase

export AbstractLinReg, AbstractLSMethod, AbstractLADMethod,
    OLS, LSSVD, LSLdiv, RLSTikhonov, RLST, LADPP, LADDP,
    RLADPP, RALDDP, RLSSVD, regress, standardize

abstract type AbstractLinReg end
abstract type AbstractLSMethod <: AbstractLinReg end
abstract type AbstractLADMethod <: AbstractLinReg end

# --------- #
# Utilities #
# --------- #

function standardize(x::AbstractMatrix, intercept::Bool=false)
    _x = intercept ? view(x, :, 2:size(x, 2)) : view(x, :, :)
    μ = mean(_x, 1)
    σ = std(_x, 1)
    xn = (_x .- μ) ./ σ
    xn, vec(μ), vec(σ)
end

function standardize(y::AbstractVector, intercept=false)
    μ = mean(y)
    σ = std(y)
    yn = (y - μ) / σ
    yn, μ, σ
end

add_intercept{T}(x::AbstractArray{T}) = [ones(T, size(x, 1)) x]

# --- #
# OLS #
# --- #

@with_kw struct OLS <: AbstractLSMethod
    standardize::Bool = true
    intercept::Bool = true
    @assert !(standardize && !(intercept)) "must have intercept if standardizing"
end

# API methods
should_standardize(m::OLS) = m.standardize
should_add_intercept(m::OLS) = m.intercept
slopes(::OLS, x, y) = pinv(x'x)*x'y

# ------ #
# LSLdiv #
# ------ #

@with_kw struct LSLdiv <: AbstractLSMethod
    standardize::Bool = true
    intercept::Bool = true
    @assert !(standardize && !(intercept)) "must have intercept if standardizing"
end

# API methods
should_standardize(m::LSLdiv) = m.standardize
should_add_intercept(m::LSLdiv) = m.intercept
slopes(::LSLdiv, x, y) = x\y

# ----- #
# LSSVD #
# ----- #

@with_kw struct LSSVD <: AbstractLSMethod
    standardize::Bool = true
    intercept::Bool = true
    @assert !(standardize && !(intercept)) "must have intercept if standardizing"
end

# API methods
should_standardize(m::LSSVD) = m.standardize
should_add_intercept(m::LSSVD) = m.intercept
function slopes(m::LSSVD, x, y)
    U, S, V = svd(x, thin=true)
    S_inv = diagm(1./S)
    V*S_inv*U'y
end


# ----------- #
# RLSTikhonov #
# ----------- #

@with_kw struct RLSTikhonov <: AbstractLSMethod
    η::Float64 = -5.0
    standardize::Bool = true
    intercept::Bool = true
    @assert !(standardize && !(intercept)) "must have intercept if standardizing"
    @assert η < 0.0 "penalty parameter for RLSTikhonov must be negative"
end

const RLST = RLSTikhonov

# API methods
should_standardize(m::RLSTikhonov) = m.standardize
should_add_intercept(m::RLSTikhonov) = m.intercept
function slopes(m::RLSTikhonov, x, y)
    nobs = size(x, 1); nx = size(x, 2)
    pinv(x'x + (nobs/nx*10^(m.η))*I)*x'y
end

# ----- #
# LADPP #
# ----- #

@with_kw struct LADPP <: AbstractLADMethod
    standardize::Bool = true
    intercept::Bool = true
    @assert (standardize && intercept) "standardize and intercept must both be true"
end

doc"""
Solves the linear program

$min_{v⁺; v⁻; β} 1v⁺ + 1v⁻$

such that

$\begin{cases}v⁺ - v⁻ + X \beta &= y \\ v⁺ \ge 0, \quad v⁻ \ge 0$

"""
function slopes(m::LADPP, x, y::AbstractMatrix)
    nobs = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    # The linear programming coefficient vector, x, is ordered [v⁺; v⁻; β]
    # we will now construct the matrices for the upper and lower bounds. Note
    # that we artificailly say that β must be between -300 and 300
    lb = [fill(0.0, 2*nobs); fill(-300, nx)]
    ub = [fill(Inf, 2*nobs); fill(300, nx)]

    # objective
    c = [ones(2*nobs); zeros(nx)]

    # equality constraints
    A = [eye(nobs, nobs) -eye(nobs, nobs) x]

    # we need to do linprog equation by equation (1 column of y at a time),
    # so we will pre-allocate a coefficient matrix and fill it in
    β = Array(Float64, nx, ny)

    for i in 1:ny
        # equality constraint
        b = y[:, i]
        out = linprog(c, A, '=', b, lb, ub).sol
        β[:, i] = out[end-nx+1:end]

    end
    β
end

# ----- #
# LADDP #
# ----- #

@with_kw struct LADDP <: AbstractLADMethod
    standardize::Bool = true
    intercept::Bool = true
    @assert (standardize && intercept) "standardize and intercept must both be true"
end

doc"""
Solves the linear program

$min_{q} -y'q

such that

$X'q = 0 \quad -1 \le q \le 1$

"""
function slopes(m::LADDP, x, y::AbstractMatrix)
    nobs = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    # The linear programming coefficient vector, x, is ordered [q]
    # we will now construct the matrices for the upper and lower bounds. Note
    # that we artificailly say that β must be between -300 and 300
    lb = fill(-1.0, nobs)
    ub = fill(1.0, nobs)

    # equality constraint
    b = zeros(nx)

    # equality constraints
    A = x'

    # we need to do linprog equation by equation (1 column of y at a time),
    # so we will pre-allocate a coefficient matrix and fill it in
    β = Array(Float64, nx, ny)

    for i in 1:ny
        # equality constraint
        c = -y[:, i]
        out = linprog(c, A, '=', b, lb, ub)
        β[:, i] = -out.attrs[:lambda][1:nx]

    end
    β
end

# ------ #
# RLADPP #
# ------ #

@with_kw struct RLADPP <: AbstractLADMethod
    η::Float64 = -5.0
    standardize::Bool = true
    intercept::Bool = true
    @assert η < 0.0 "penalty parameter for RLSTikhonov must be negative"
    @assert (standardize && intercept) "standardize and intercept must both be true"
end

doc"""
Solves the linear program

$min_{c:=[v⁺; v⁻; ψ⁺; ψ⁻]} 1v⁺ + 1v⁻ + 1ψ⁺+ 1ψ⁻$

such that

$\begin{cases}v⁺ - v⁻ + Xψ⁺ - Xψ⁻ &= y \\ c &\ge 0$

"""
function slopes(m::RLADPP, x, y::AbstractMatrix)
    nobs = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    # The linear programming coefficient vector, x, is ordered [v⁺; v⁻; ψ⁺; ψ⁻]
    lb = fill(0.0, 2*nobs + 2*nx)
    ub = fill(Inf, 2*nobs + 2*nx)

    # objective
    c = [ones(2*nobs); fill(10^(m.η)*nobs/nx, 2*nx)]

    # equality constraints
    A = [eye(nobs, nobs) -eye(nobs, nobs) x -x]

    # we need to do linprog equation by equation (1 column of y at a time),
    # so we will pre-allocate a coefficient matrix and fill it in
    β = Array(Float64, nx, ny)

    for i in 1:ny
        # equality constraint
        b = y[:, i]
        out = linprog(c, A, '=', b, lb, ub).sol
        ψ⁺ = out[end-(2*nx)+1:end-nx]
        ψ⁻ = out[end-nx+1:end]
        β[:, i] = ψ⁺ - ψ⁻
    end
    β
end

# ------ #
# RLADDP #
# ------ #

@with_kw struct RLADDP <: AbstractLADMethod
    η::Float64 = -5.0
    standardize::Bool = true
    intercept::Bool = true
    @assert η < 0.0 "penalty parameter for RLSTikhonov must be negative"
    @assert (standardize && intercept) "standardize and intercept must both be true"
end

doc"""
Solves the linear program

$min_{q} -y'q

such that

$X'q \le η1 \quad -X'q \le η1, \quad -1 \le q \le 1$

"""
function slopes(m::RLADDP, x, y::AbstractMatrix)
    nobs = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    # The linear programming coefficient vector, x, is ordered [q]
    # we will now construct the matrices for the upper and lower bounds. Note
    # that we artificailly say that β must be between -300 and 300
    lb = fill(-1.0, nobs)
    ub = fill(1.0, nobs)

    # constraints equations
    A = [x'; -x']
    b = fill(10^(m.η)*nobs/nx, 2*nx)

    # we need to do linprog equation by equation (1 column of y at a time),
    # so we will pre-allocate a coefficient matrix and fill it in
    β = Array(Float64, nx, ny)

    for i in 1:ny
        # equality constraint
        c = -y[:, i]
        out = linprog(c, A, '<', b, lb, ub)
        ψ⁺ = -out.attrs[:lambda][1:nx]
        ψ⁻ = -out.attrs[:lambda][nx+1:2*nx]
        β[:, i] = ψ⁺ - ψ⁻
    end
    β

end

# ------ #
# RLSSVD #
# ------ #

@with_kw struct RLSSVD <: AbstractLSMethod
    κ::Float64=100_000.0
    standardize::Bool = true
    intercept::Bool = true
    @assert κ > 0 "κ must be positive"
    @assert !(standardize && !(intercept)) "must have intercept if standardizing"
end

# API methods
should_standardize(m::RLSSVD) = m.standardize
should_add_intercept(m::RLSSVD) = m.intercept
function slopes(m::RLSSVD, x, y)
    U, S, V = svd(x, thin=true)

    # find number of sufficiently independent principal components
    condition_numbers = S[1] ./ S[2:end]
    r = findfirst(_xx -> _xx > m.κ, condition_numbers)
    r = r == 0 ? length(S) : r

    # extract these columns of V, U, S
    Vr = V[:, 1:r]
    Ur = U[:, 1:r]
    Sr_inv = diagm(1./S[1:r])

    # compute estimate
    Vr*Sr_inv*Ur'y
end


# -------- #
# Generics #
# -------- #

function slopes(m::AbstractLADMethod, x, y::AbstractVector)
    βmat = slopes(m, x, y[:, :])
    βmat[:, 1]
end

should_standardize(::AbstractLinReg) = true
should_add_intercept(::AbstractLinReg) = true

_reg_nonorm_noint(m::AbstractLinReg, x, y) = slopes(m, x, y)
_reg_nonorm_int(m::AbstractLinReg, x, y) = slopes(m, add_intercept(x), y)

function _reg_norm_noint(m::AbstractLinReg, x, y)
    xn, μx, σx = standardize(x)
    yn, μy, σy = standardize(y)

    # Compute standardized coefficients, then undo normalization
    β = slopes(m, xn, yn)
    # update slopes
    for j in 1:size(y, 2)
        for i in 1:size(β, 1)
            β[i, j] *= σy[j]/σx[i]
        end
    end
    β
end

function _reg_norm_int(m::AbstractLinReg, x, y::AbstractVector)
    xn, μx, σx = standardize(x)
    yn, μy, σy = standardize(y)

    # Compute standardized coefficients, then undo normalization
    β1 = slopes(m, xn, yn)
    for i in eachindex(β1)
        β1[i] *= σy/σx[i]
    end
    β0 = μy - dot(μx, β1)

    β = similar(β1, length(β1) + 1)
    β[1] = β0
    β[2:end] = β1
    β
end

function _reg_norm_int(m::AbstractLinReg, x, y::AbstractMatrix)
    xn, μx, σx = standardize(x)
    yn, μy, σy = standardize(y)

    # Compute standardized coefficients, then undo normalization
    β1 = slopes(m, xn, yn)
    β0 = similar(β1, 1, size(y, 2))

    # update slopes
    for j in 1:size(y, 2)
        for i in 1:size(β1, 1)
            β1[i, j] *= σy[j]/σx[i]
        end
    end

    # compute intercept
    for j in 1:size(y, 2)
        β0[j] = μy[j] - dot(β1[:, j], μx)
    end

    Float64[β0; β1]
end

regress(m::AbstractLinReg, x, y) =
    (should_standardize(m)  && should_add_intercept(m))  ? _reg_norm_int(m, x, y) :
    (!should_standardize(m) && should_add_intercept(m))  ? _reg_nonorm_int(m, x, y) :
    (should_standardize(m)  && !should_add_intercept(m)) ? _reg_norm_noint(m, x, y) :
                                                         _reg_nonorm_noint(m, x, y)


end  # module
