# test/test_macro_broadcast.jl
using Test
using LoopToVecs

@testset "@t broadcast & add-assign (no reduction)" begin
    ni, nj = 5, 4

    # add-assign: Q[i,j] += G[i,j] * w[i] - H[i,j]
    G = rand(ni, nj)
    w = rand(ni)
    H = rand(ni, nj)
    Q1 = rand(ni, nj); Q2 = copy(Q1)
    @t Q1[i,j] += G[i,j] * w[i] - H[i,j]
    Q2 = Q2 .+ G .* reshape(w, (ni,)) .- H
    @test Q1 ≈ Q2

    # pure broadcast, build a 3D result; include a 2D term K[i,j]
    nn = 3
    D = rand(nn, ni, nj)
    T = rand(nn, ni, nj)
    K = rand(ni, nj)
    P1 = similar(D)
    @t P1[n,i,j] := D[n,i,j] + log(1 + T[n,i,j]) + K[i,j]
    P2 = D .+ log.(1 .+ T) .+ reshape(K, (1, ni, nj))
    @test P1 ≈ P2

    # simple unary broadcast
    X = rand(ni)
    @t Y1[i] := exp(X[i])
    @test Y1 ≈ exp.(X)
end

@testset "@t in-place = and +=" begin
    ni, nj = 5, 4

    # `=` writes in-place into pre-allocated array
    X = rand(ni, nj)
    Y = similar(X)
    Y_ptr = pointer(Y)
    @t Y[i,j] = X[i,j] * 2.0
    @test Y ≈ X .* 2.0
    @test pointer(Y) === Y_ptr  # same memory, not a new allocation

    # `+=` writes in-place into pre-allocated array
    A = rand(ni, nj)
    B = rand(ni, nj)
    A_orig = copy(A)
    A_ptr = pointer(A)
    @t A[i,j] += B[i,j]
    @test A ≈ A_orig .+ B
    @test pointer(A) === A_ptr  # same memory
end