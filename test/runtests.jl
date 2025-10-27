using LoopToVecs
using Test
using Random

Random.seed!(0xDEADBEEF)
include("test_helpers.jl")
include("test_rewrite.jl")
include("test_macro_broadcast.jl")
include("test_macro_reduction.jl")