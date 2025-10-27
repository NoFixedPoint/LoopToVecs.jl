# test/test_rewrite.jl
using Test
using LoopToVecs

@testset "Rewriting refs and RHS" begin
    ni, nj, nk = 2, 4, 3

    # A[i,k] -> reshape(permute(A), (i, 1, k))
    A = reshape(collect(1:ni*nk), ni, nk) .+ 0.1
    canon = [:i, :j, :k]
    ex = LoopToVecs._rewrite_ref(:A, [:i, :k], canon)
    L = eval(:(let A=$A; $ex; end))
    @test size(L) == (ni, 1, nk)
    @test L[:, 1, :] ≈ A

    # permutation case: D[k,i] -> canon [:i, :j, :k] => (i, 1, k)
    D = reshape(collect(1:nk*ni), nk, ni)
    ex2 = LoopToVecs._rewrite_ref(:D, [:k, :i], canon)
    L2 = eval(:(let D=$D; $ex2; end))
    @test size(L2) == (ni, 1, nk)
    @test dropdims(L2; dims=2) ≈ permutedims(D, (2, 1))

    # rewrite a full RHS with broadcasted ops
    B = rand(ni, nj, nk)
    C = rand(ni, nj, nk)
    rhs = :(A[i,k] * B[i,j,k] + log(1 + C[i,j,k]))
    rhs_rw = LoopToVecs._rewrite_rhs(rhs, canon)
    val = eval(:(let A=$A, B=$B, C=$C; $rhs_rw; end))
    base = reshape(A, (ni, 1, nk)) .* B .+ log.(1 .+ C)
    @test val ≈ base
end
