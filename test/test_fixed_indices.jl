# test/test_fixed_indices.jl
using Test
using LoopToVecs

@testset "Fixed indices: integer literal" begin
    A = zeros(3, 4)
    B = [10.0, 20.0, 30.0, 40.0]
    @t A[2, j] = B[j]
    @test A[2, :] == B
    @test A[1, :] == zeros(4)
    @test A[3, :] == zeros(4)

    A = zeros(3, 4)
    B = rand(3, 4)
    @t A[1, j] = B[3, j]
    @test A[1, :] == B[3, :]
    @test A[2, :] == zeros(4)
end

@testset "Fixed indices: \$var" begin
    A = zeros(3, 4)
    B = rand(3, 4)
    idx = 2
    @t A[$idx, j] = B[$idx, j]
    @test A[2, :] == B[2, :]
    @test A[1, :] == zeros(4)

    # $var on RHS only, := creates new array
    B = rand(3, 4)
    idx = 2
    @t C[j] := B[$idx, j]
    @test C == B[2, :]

    # $var with +=
    A = ones(3, 4)
    B = rand(4)
    idx = 2
    @t A[$idx, j] += B[j]
    @test A[2, :] ≈ 1.0 .+ B
    @test A[1, :] == ones(4)

    # $var with *=
    A = 2.0 * ones(3, 4)
    B = [1.0, 2.0, 3.0, 4.0]
    idx = 3
    @t A[$idx, j] *= B[j]
    @test A[3, :] ≈ 2.0 .* B
    @test A[1, :] == 2.0 * ones(4)
end

@testset "Fixed indices: \$expr arithmetic" begin
    A = zeros(5, 3)
    B = rand(5, 3)
    idx = 2
    @t A[$idx+1, j] = B[$idx+1, j]
    @test A[3, :] == B[3, :]
    @test A[1, :] == zeros(3)
end

@testset "Fixed indices: combined with broadcasting" begin
    A = zeros(3, 4, 5)
    B = rand(4)
    C = rand(4, 5)
    idx = 2
    @t A[$idx, j, k] = B[j] + C[j, k]
    expected = reshape(B, (4, 1)) .+ C
    @test A[2, :, :] ≈ expected
    @test A[1, :, :] == zeros(4, 5)
end

@testset "Fixed indices: combined with reduction" begin
    A = rand(3, 4, 5)
    idx = 2
    @t (+) result[k] := A[$idx, j, k]
    @test result ≈ vec(sum(A[2, :, :], dims=1))
end

@testset "Fixed indices: errors" begin
    # integer literal fixed index with := should error at macro expansion
    @test_throws LoadError begin
        eval(quote
            A = rand(3, 4)
            @t A[2, j] := rand(4)
        end)
    end
end
