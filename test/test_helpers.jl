# test/test_helpers.jl
using Test
using LoopToVecs

@testset "Helper utilities" begin
    # _unique_syms
    @test LoopToVecs._unique_syms([:i, :k, :i, :j, :k, :j]) == [:i, :k, :j]

    # _collect_rhs_inds: first-appearance order from array refs only
    ex = :(A[i,k] * B[i,j,k] + log(1 + C[i,j,k]))
    inds = LoopToVecs._collect_rhs_inds(ex)
    @test inds == [:i, :k, :j]

    # _should_broadcast
    @test LoopToVecs._should_broadcast(:+)   === true
    @test LoopToVecs._should_broadcast(:log) === true
    for f in [:reshape, :dropdims, :sum, :maximum, :minimum, :prod,
              :tuple, :size, :axes, :length, :PermutedDimsArray]
        @test LoopToVecs._should_broadcast(f) === false
    end

    # _dims_arg
    @test LoopToVecs._dims_arg([2]) == 2
    d = LoopToVecs._dims_arg([1,3])
    @test d isa Expr
    @test d.head == :tuple
    @test d.args == Any[1, 3]

    # _tuple: returns tuple Expr we can eval in the module's scope
    tup_expr = LoopToVecs._tuple([1, 2, 3])
    @test LoopToVecs.eval(tup_expr) == (1, 2, 3)

    # _broadcastify: turn calls into Base.broadcast(…)
    let X = rand(3, 2)
        ex  = :(log(1 + X))
        exb = LoopToVecs._broadcastify(ex)
        val = eval(:(let X=$X; $exb; end))
        @test val ≈ log.(1 .+ X)
    end
end
