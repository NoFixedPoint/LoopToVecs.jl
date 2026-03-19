# test/test_diagonal_indices.jl
using Test
using LoopToVecs
using LinearAlgebra

@testset "Diagonal indices: basic 2-repeat" begin
    A = zeros(3, 3)
    @t A[n, n] = 1.0
    @test A ≈ Matrix(I, 3, 3)

    A = zeros(4, 4)
    B = rand(4, 4)
    @t A[n, n] = B[n, n]
    @test diag(A) ≈ diag(B)
    @test A[1, 2] == 0.0
end

@testset "Diagonal indices: += on diagonal" begin
    A = zeros(3, 3)
    @t A[n, n] += 5.0
    @test diag(A) == [5.0, 5.0, 5.0]
    @test A[1, 2] == 0.0
end

@testset "Diagonal indices: *= on diagonal" begin
    A = 2.0 * ones(3, 3)
    @t A[n, n] *= 3.0
    @test diag(A) == [6.0, 6.0, 6.0]
    @test A[1, 2] == 2.0
end

@testset "Diagonal indices: mixed with non-repeated dim" begin
    A = zeros(3, 3, 4)
    B = rand(3, 4)
    @t A[n, n, j] = B[n, j]
    for n in 1:3, j in 1:4
        @test A[n, n, j] ≈ B[n, j]
    end
    @test A[1, 2, 1] == 0.0
end

@testset "Diagonal indices: non-contiguous A[n,j,n]" begin
    A = zeros(3, 4, 3)
    B = rand(3, 4)
    @t A[n, j, n] = B[n, j]
    for n in 1:3, j in 1:4
        @test A[n, j, n] ≈ B[n, j]
    end
    @test A[1, 1, 2] == 0.0
end

@testset "Diagonal indices: 3-repeat" begin
    A = zeros(3, 3, 3)
    @t A[n, n, n] = 1.0
    for n in 1:3
        @test A[n, n, n] == 1.0
    end
    @test A[1, 2, 3] == 0.0
    @test A[1, 1, 2] == 0.0
end

@testset "Diagonal indices: combined with fixed index" begin
    A = zeros(5, 3, 3)
    idx = 2
    @t A[$idx, n, n] = 1.0
    for n in 1:3
        @test A[2, n, n] == 1.0
    end
    @test A[1, 1, 1] == 0.0
    @test A[2, 1, 2] == 0.0
end

@testset "Diagonal indices: RHS diagonal extraction" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @t C[n] := A[n, n]
    @test C ≈ [1.0, 5.0, 9.0]
end

@testset "Diagonal indices: RHS diagonal with broadcasting" begin
    B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    A = zeros(3, 3)
    @t A[i, j] = B[i, i]
    for j in 1:3
        @test A[:, j] ≈ diag(B)
    end
end

@testset "Diagonal indices: errors" begin
    @test_throws LoadError begin
        eval(quote
            A = zeros(3, 3)
            @t A[n, n] := 1.0
        end)
    end
end
