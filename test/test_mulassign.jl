# test/test_mulassign.jl
using Test
using LoopToVecs

@testset "@t *= operator" begin
    # scalar *= scalar
    s = 5.0
    @t s *= 2.0
    @test s == 10.0

    # array *= scalar
    A = [1.0 2.0; 3.0 4.0]
    A_ptr = pointer(A)
    @t A[i,j] *= 2.0
    @test A == [2.0 4.0; 6.0 8.0]
    @test pointer(A) === A_ptr  # in-place

    # array *= array (broadcasting)
    A = ones(3, 4)
    B = [1.0, 2.0, 3.0, 4.0]
    A_ptr = pointer(A)
    @t A[i,j] *= B[j]
    @test A ≈ ones(3) * B'
    @test pointer(A) === A_ptr

    # *= with reduction → error
    @test_throws LoadError begin
        eval(quote
            X = rand(3, 4)
            @t (+) X[i] *= X[i,j]
        end)
    end
end
