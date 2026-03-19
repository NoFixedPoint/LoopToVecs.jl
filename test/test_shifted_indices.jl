# test/test_shifted_indices.jl
using Test
using LoopToVecs

@testset "Shifted indices: a+1 on RHS with :=" begin
    N = [1.0, 3.0, 6.0, 10.0, 15.0]
    @t eq[a] := N[a+1] - N[a]
    @test length(eq) == 4
    for a in 1:4
        @test eq[a] ≈ N[a+1] - N[a]
    end
end

@testset "Shifted indices: a+1 with scalar multiply" begin
    N = [1.0, 2.0, 3.0, 4.0, 5.0]
    zeta_a = [0.9, 0.85, 0.8, 0.75]
    gN = 0.05
    @t eq1[a] := N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    @test length(eq1) == 4
    for a in 1:4
        @test eq1[a] ≈ N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    end
end

@testset "Shifted indices: a-1 on RHS" begin
    N = [1.0, 3.0, 6.0, 10.0, 15.0]
    @t eq[a] := N[a] - N[a-1]
    @test length(eq) == 4
    for a in 2:5
        @test eq[a-1] ≈ N[a] - N[a-1]
    end
end

@testset "Shifted indices: a+1 on LHS (in-place)" begin
    A = zeros(5)
    B = [10.0, 20.0, 30.0, 40.0]
    @t A[a+1] = B[a]
    @test A[2:5] == B
    @test A[1] == 0.0
end

@testset "Shifted indices: multi-dimensional" begin
    A = rand(4, 3)
    @t B[a, j] := A[a+1, j] - A[a, j]
    @test size(B) == (3, 3)
    for a in 1:3, j in 1:3
        @test B[a, j] ≈ A[a+1, j] - A[a, j]
    end
end

@testset "Shifted indices: with reduction" begin
    A = rand(4, 3)
    @t (+) s := A[a+1, j] - A[a, j]
    expected = sum(A[a+1, j] - A[a, j] for a in 1:3, j in 1:3)
    @test s ≈ expected
end

@testset "Shifted indices: := on LHS with shift error" begin
    @test_throws LoadError begin
        eval(quote
            A = zeros(5)
            B = rand(4)
            @t A[a+1] := B[a]
        end)
    end
end

@testset "Shifted indices: combined with bare index" begin
    N = [1.0, 2.0, 3.0, 4.0, 5.0]
    @t eq[a] := N[a+1] * (a > 1)
    @test eq[1] ≈ 0.0
    @test eq[2] ≈ N[3]
    @test eq[3] ≈ N[4]
    @test eq[4] ≈ N[5]
end
