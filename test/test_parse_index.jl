# test/test_parse_index.jl
using Test
using LoopToVecs

@testset "_parse_index" begin
    # plain symbol → loop
    @test LoopToVecs._parse_index(:a) == (:loop, :a, 0)
    @test LoopToVecs._parse_index(:xyz) == (:loop, :xyz, 0)

    # integer literal → fixed
    @test LoopToVecs._parse_index(1) == (:fixed, 1, 0)
    @test LoopToVecs._parse_index(42) == (:fixed, 42, 0)

    # $expr → fixed
    @test LoopToVecs._parse_index(Expr(:$, :a)) == (:fixed, :a, 0)
    @test LoopToVecs._parse_index(Expr(:$, :num_a)) == (:fixed, :num_a, 0)

    # sym + k → shifted
    @test LoopToVecs._parse_index(Expr(:call, :+, :a, 1)) == (:shifted, :a, 1)
    @test LoopToVecs._parse_index(Expr(:call, :-, :a, 1)) == (:shifted, :a, -1)
    @test LoopToVecs._parse_index(Expr(:call, :+, :a, 3)) == (:shifted, :a, 3)

    # k + sym → shifted (reversed operands)
    @test LoopToVecs._parse_index(Expr(:call, :+, 1, :a)) == (:shifted, :a, 1)

    # ($sym) + k → fixed with arithmetic (runtime value)
    ex = Expr(:call, :+, Expr(:$, :a), 1)
    result = LoopToVecs._parse_index(ex)
    @test result[1] == :fixed
    @test result[3] == 0

    # ($sym) - k → fixed with arithmetic
    ex2 = Expr(:call, :-, Expr(:$, :a), 1)
    result2 = LoopToVecs._parse_index(ex2)
    @test result2[1] == :fixed
    @test result2[3] == 0

    # unsupported → error
    @test_throws ErrorException LoopToVecs._parse_index(1.5)
    @test_throws ErrorException LoopToVecs._parse_index("bad")
end
