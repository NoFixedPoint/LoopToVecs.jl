# test/test_bare_indices.jl
using Test
using LoopToVecs

@testset "Bare loop indices: simple comparisons" begin
    A = zeros(5, 3)
    @t A[i, j] = 1.0 * (i <= 3)
    @test A[1:3, :] == ones(3, 3)
    @test A[4:5, :] == zeros(2, 3)

    A = zeros(5, 3)
    @t A[i, j] = 1.0 * (j == 2)
    @test A[:, 2] == ones(5)
    @test A[:, 1] == zeros(5)
    @test A[:, 3] == zeros(5)
end

@testset "Bare loop indices: two-index comparison" begin
    A = zeros(4, 4)
    @t A[i, j] = 1.0 * (i != j)
    for ii in 1:4, jj in 1:4
        @test A[ii, jj] == (ii != jj ? 1.0 : 0.0)
    end

    A = zeros(3, 3)
    @t A[n, np] = 5.0 * (n != np)
    for n in 1:3, np in 1:3
        @test A[n, np] == (n != np ? 5.0 : 0.0)
    end
end

@testset "Bare loop indices: combined with array refs" begin
    num_a, num_s = 5, 2
    ss_as = zeros(num_a, num_s)
    ss_bar = 0.3
    aR = 3
    @t ss_as[a, s] = ss_bar * (a < aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test ss_as[a, s] ≈ ss_bar * (a < aR) * (s == 2)
    end

    B = rand(5, 2)
    A = zeros(5, 2)
    aR = 3
    @t A[a, s] = B[a, s] * (a >= aR)
    for a in 1:5, s in 1:2
        @test A[a, s] ≈ B[a, s] * (a >= aR)
    end
end

@testset "Bare loop indices: with += and reduction" begin
    A = zeros(3, 3)
    @t A[n, i] += 1.0 * (n != i)
    for n in 1:3, i in 1:3
        @test A[n, i] == (n != i ? 1.0 : 0.0)
    end

    num_a, num_s = 5, 2
    B = rand(num_a, num_s)
    ss_bar = 0.1
    aR = 3
    @t (+) total := B[a, s] * ss_bar * (a < aR) * (s == 2)
    expected = sum(
        B[a, s] * ss_bar * (a < aR) * (s == 2)
        for a in 1:num_a, s in 1:num_s
    )
    @test total ≈ expected
end

@testset "Bare loop indices: no array refs on RHS" begin
    A = zeros(3, 3)
    @t A[i, j] = 1.0 * (i == j)
    @test A ≈ [1 0 0; 0 1 0; 0 0 1]
end
