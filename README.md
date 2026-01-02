# LoopToVecs.jl

[![Build Status](https://github.com/NoFixedPoint/LoopToVecs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/NoFixedPoint/LoopToVecs.jl/actions/workflows/CI.yml?query=branch%3Amaster)

LoopToVecs.jl is a lightweight package that converts loop-based computations in einsum notation into vectorized code. It exports a single macro, `@t`. For usage examples, see below and in the `test/` directory.

```julia
using LoopToVecs
using Random

Random.seed!(123)
b_ij = rand(2, 3)
c_jk = rand(3, 4)

# broadcasting
@t b2_ij[i,j] = b_ij[i,j] * 2.0
@assert all(b2_ij .≈ b_ij * 2.0)
@macroexpand @t b2_ij[i,j] = b_ij[i,j] * 2.0
#=
:(b2_ij = Base.broadcast(*, reshape(b_ij, (size(b_ij, 1), size(b_ij, 2))), 2.0))
=#

# reduction, aligning index
@t (+) a_ik[i,k] = b_ij[i,j] * c_jk[j,k]
@assert all(a_ik .≈ b_ij * c_jk)
@macroexpand @t (+) a_ik[i,k] = b_ij[i,j] * c_jk[j,k]
#=
:(a_ik = dropdims(sum(Base.broadcast(*, reshape(b_ij, (size(b_ij, 1), 1, size(b_ij, 2))), reshape(PermutedDimsArray(c_jk, (2, 1)), (1, size(c_jk, 2), size(c_jk, 1)))); dims = 3); dims = 3))
=#

# permutation
@t a_ki[k,i] = a_ik[i,k]
@assert all(a_ki .≈ permutedims(a_ik, (2, 1)))
@macroexpand @t a_ki[k,i] = a_ik[i,k]
#=
:(a_ki = reshape(PermutedDimsArray(a_ik, (2, 1)), (size(a_ik, 2), size(a_ik, 1))))
=#

# reduction
@t (max) max_a_k[k] = a_ki[k,i]
@assert all(max_a_k .≈ maximum(a_ki, dims=2))
@macroexpand @t (max) max_a_k[k] = a_ki[k,i]
#=
:(max_a_k = dropdims(maximum(reshape(a_ki, (size(a_ki, 1), size(a_ki, 2))); dims = 2); dims = 2))
=#

# broadcasting
@t a_ki_normalized[k,i] = a_ki[k,i] / max_a_k[k]
@assert all(a_ki_normalized .≈ a_ki ./ max_a_k)
@macroexpand @t a_ki_normalized[k,i] = a_ki[k,i] / max_a_k[k]
#=
:(a_ki_normalized = Base.broadcast(/, reshape(a_ki, (size(a_ki, 1), size(a_ki, 2))), reshape(max_a_k, (size(max_a_k, 1), 1))))
=#
```

The package is meant to be used with compilers that require vectorized code, such as [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

```julia
using Reactant

function f(b_ij, c_jk)
    @t (+) a_ik[i,k] = b_ij[i,j] * c_jk[j,k]
    @t a_ki[k,i] = a_ik[i,k]
    @t (max) max_a_k[k] = a_ki[k,i]
    @t a_ki_normalized[k,i] = a_ki[k,i] / max_a_k[k]
    return a_ki_normalized
end

b_ij = Reactant.to_rarray(b_ij)
c_jk = Reactant.to_rarray(c_jk)
f_compiled = @compile f(b_ij, c_jk)

f_compiled(b_ij, c_jk)
#=
4×2 ConcretePJRTArray{Float64,2}:
 1.0  0.431422
 1.0  0.387322
 1.0  0.785391
 1.0  0.742992
=#
```

The package is motivated by other einsum packages:

- [Tullio.jl](https://github.com/mcabbott/Tullio.jl)
- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
