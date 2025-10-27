# test/test_macro_reduction.jl
using Test
using LoopToVecs

@testset "@t reductions (sum, max) + permutations" begin
    ni, nj, nk = 3, 4, 5

    # (+) L[i,j] = A[i,k] * B[i,j,k]
    A = rand(ni, nk)
    B = rand(ni, nj, nk)
    L1 = similar(B, ni, nj)
    @t (+) L1[i,j] = A[i,k] * B[i,j,k]
    L2 = dropdims(sum(reshape(A, (ni, 1, nk)) .* B; dims=3), dims=3)
    @test L1 ≈ L2

    # (+) with permutation on an RHS tensor: Aki[k,i]
    Aki = rand(nk, ni)
    L3 = similar(B, ni, nj)
    @t (+) L3[i,j] = Aki[k,i] * B[i,j,k]
    L4 = dropdims(sum(reshape(permutedims(Aki, (2,1)), (ni, 1, nk)) .* B; dims=3), dims=3)
    @test L3 ≈ L4

    # (max) over i: M[n,j] = C[n,i,j]
    n = 2
    C = rand(n, ni, nj) .* 10 .- 5
    M1 = similar(C, n, nj)
    @t (max) M1[n,j] = C[n,i,j]
    M2 = dropdims(maximum(C; dims=2), dims=2)
    @test M1 ≈ M2
end

@testset "@t scalar reductions" begin
    ni, nk = 4, 3
    A = rand(ni, nk)

    # (+) to scalar
    s1 = 0.0
    @t (+) s1 = A[i,k]
    @test s1 ≈ sum(A)

    # (max) to scalar
    m1 = -Inf
    @t (max) m1 = A[i,k]
    @test m1 ≈ maximum(A)

    # scalar add-assign with reduction
    s2 = 1.0
    @t (+) s2 += A[i,k]
    @test s2 ≈ 1.0 + sum(A)
end

@testset "@t error handling" begin
    # RHS has a reduction index but no reducer specified
    @test_throws LoadError begin
        ex = quote
            A = rand(3,4,2); B = rand(3,4,2)
            L = zeros(3,4)
            @t L[i,j] = A[i,j,k] + B[i,j,k]
        end
        eval(ex)
    end
end

