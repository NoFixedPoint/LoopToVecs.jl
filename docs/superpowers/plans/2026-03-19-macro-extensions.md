# Macro Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `@t` macro with fixed indices (`$`), bare loop indices in expressions, `*=` operator, shifted indices (`a+1`), and diagonal/repeated indices.

**Architecture:** Each feature is added incrementally to the existing macro. A `_parse_index` helper classifies each index position as `:loop`, `:fixed`, or `:shifted`. Array refs with fixed/shifted positions use `view()`, diagonal positions use `PermutedDimsArray` + `CartesianIndex`. Bare loop indices are replaced with reshaped range vectors.

**Tech Stack:** Julia, `Base.broadcast`, `PermutedDimsArray`, `CartesianIndex`, `view`

**Spec:** `docs/superpowers/specs/2026-03-19-macro-extensions-design.md`

---

## File Structure

**Modified:**
- `src/LoopToVecs.jl` — all new features added here (single-file module)
- `test/runtests.jl` — add includes for new test files

**Created:**
- `test/test_parse_index.jl` — unit tests for `_parse_index` helper
- `test/test_mulassign.jl` — tests for `*=` operator
- `test/test_fixed_indices.jl` — tests for `$expr` and integer literal indices
- `test/test_bare_indices.jl` — tests for bare loop indices in expressions
- `test/test_shifted_indices.jl` — tests for `a+1`/`a-1` shifted indices
- `test/test_diagonal_indices.jl` — tests for repeated/diagonal indices
- `test/test_integration.jl` — real-world patterns from `code_samples/`

---

## Task 1: `_parse_index` helper and `*=` operator

**Files:**
- Modify: `src/LoopToVecs.jl`
- Create: `test/test_parse_index.jl`
- Create: `test/test_mulassign.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write `_parse_index` unit tests**

Create `test/test_parse_index.jl`:
```julia
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
```

- [ ] **Step 2: Write `*=` macro tests**

Create `test/test_mulassign.jl`:
```julia
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: FAIL — `_parse_index` not defined, `*=` not handled

- [ ] **Step 4: Implement `_parse_index` and `*=` support**

In `src/LoopToVecs.jl`, add `_parse_index` after the existing helpers (after `_broadcastify`):

```julia
# parse a single index expression from an array subscript
# Returns (type::Symbol, value, offset::Int)
#   (:loop, sym, 0)       — plain loop index
#   (:fixed, expr, 0)     — fixed scalar ($expr or integer literal)
#   (:shifted, sym, k)    — shifted loop index sym+k
function _parse_index(ex)
    if ex isa Integer
        return (:fixed, ex, 0)
    elseif ex isa Symbol
        return (:loop, ex, 0)
    elseif !(ex isa Expr)
        error("Unsupported index expression: $ex")
    end
    if ex.head == :$
        return (:fixed, ex.args[1], 0)
    elseif ex.head == :call && length(ex.args) == 3
        op, arg1, arg2 = ex.args
        if (op === :+ || op === :-) && arg2 isa Integer
            if arg1 isa Expr && arg1.head == :$
                # ($sym) + k → fixed with runtime arithmetic
                return (:fixed, Expr(:call, op, arg1.args[1], arg2), 0)
            elseif arg1 isa Symbol
                # sym + k → shifted loop index
                offset = (op === :+) ? arg2 : -arg2
                return (:shifted, arg1, offset)
            end
        end
        # k + sym → shifted (Julia may parse 1+a as +(1, a))
        if op === :+ && arg1 isa Integer && arg2 isa Symbol
            return (:shifted, arg2, arg1)
        end
    end
    error("Unsupported index expression: $ex")
end
```

Add `:view` to `_NO_BCAST_FUNS`:
```julia
const _NO_BCAST_FUNS = Set([:reshape, :dropdims, :sum, :maximum, :minimum, :prod,
                            :PermutedDimsArray, :tuple, :size, :axes, :length,
                            :view, :CartesianIndex])
```

For `*=`, modify the operator parsing section of the macro. Change:
```julia
    is_addassign = (ex.head == :+=)
    is_inplace = (ex.head == :(=))
    is_newvar = (ex.head == :(:=))
    is_newvar || is_inplace || is_addassign || error("Use `:=`, `=`, or `+=` with @t")
```
to:
```julia
    is_addassign = (ex.head == :+=)
    is_mulassign = (ex.head == :*=)
    is_inplace = (ex.head == :(=))
    is_newvar = (ex.head == :(:=))
    is_newvar || is_inplace || is_addassign || is_mulassign ||
        error("Use `:=`, `=`, `+=`, or `*=` with @t")
```

Add error check after reduction parsing:
```julia
    if is_mulassign && redsym !== nothing
        error("Cannot combine `*=` with a reduction operator")
    end
```

In the assignment generation section, add `*=` case. Change:
```julia
    out = if is_addassign
        ...
    elseif is_inplace
        ...
    else
        ...
    end
```
to:
```julia
    out = if is_mulassign
        if is_scalar_lhs
            :( $L = $L * $rhs_final )
        else
            Expr(:(.=), L, :(Base.broadcast(*, $L, $rhs_final)))
        end
    elseif is_addassign
        if is_scalar_lhs
            :( $L = $L + $rhs_final )
        else
            Expr(:(.=), L, :(Base.broadcast(+, $L, $rhs_final)))
        end
    elseif is_inplace
        if is_scalar_lhs
            :( $L = $rhs_final )
        else
            Expr(:(.=), L, rhs_final)
        end
    else
        :( $L = $rhs_final )
    end
```

Update `test/runtests.jl` to include new test files:
```julia
using LoopToVecs
using Test
using Random

Random.seed!(0xDEADBEEF)
include("test_helpers.jl")
include("test_parse_index.jl")
include("test_rewrite.jl")
include("test_macro_broadcast.jl")
include("test_macro_reduction.jl")
include("test_mulassign.jl")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS (including existing tests)

- [ ] **Step 6: Commit**

```bash
cd D:/macro_local/LoopToVecs.jl
git add src/LoopToVecs.jl test/test_parse_index.jl test/test_mulassign.jl test/runtests.jl
git commit -m "feat: add _parse_index helper and *= operator support"
```

---

## Task 2: Fixed indices (`$expr` and integer literals)

**Files:**
- Modify: `src/LoopToVecs.jl`
- Create: `test/test_fixed_indices.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write fixed index tests**

Create `test/test_fixed_indices.jl`:
```julia
# test/test_fixed_indices.jl
using Test
using LoopToVecs

@testset "Fixed indices: integer literal" begin
    # integer literal on LHS, assign to row 2
    A = zeros(3, 4)
    B = [10.0, 20.0, 30.0, 40.0]
    @t A[2, j] = B[j]
    @test A[2, :] == B
    @test A[1, :] == zeros(4)
    @test A[3, :] == zeros(4)

    # integer literal on LHS and RHS
    A = zeros(3, 4)
    B = rand(3, 4)
    @t A[1, j] = B[3, j]
    @test A[1, :] == B[3, :]
    @test A[2, :] == zeros(4)
end

@testset "Fixed indices: \$var" begin
    # $var on LHS and RHS
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
    @test A[1, :] == ones(4)  # unchanged

    # $var with *=
    A = 2.0 * ones(3, 4)
    B = [1.0, 2.0, 3.0, 4.0]
    idx = 3
    @t A[$idx, j] *= B[j]
    @test A[3, :] ≈ 2.0 .* B
    @test A[1, :] == 2.0 * ones(4)  # unchanged
end

@testset "Fixed indices: \$expr arithmetic" begin
    # $a+1 is a fixed index with runtime value a+1
    A = zeros(5, 3)
    B = rand(5, 3)
    idx = 2
    @t A[$idx+1, j] = B[$idx+1, j]
    @test A[3, :] == B[3, :]
    @test A[1, :] == zeros(3)
end

@testset "Fixed indices: combined with broadcasting" begin
    # fixed + dimension alignment
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
    # $var on RHS with reduction
    A = rand(3, 4, 5)
    idx = 2
    @t (+) result[k] := A[$idx, j, k]
    @test result ≈ vec(sum(A[2, :, :], dims=1))
end

@testset "Fixed indices: errors" begin
    # := with fixed on LHS → error
    @test_throws LoadError begin
        eval(quote
            A = rand(3, 4)
            idx = 1
            @t A[$idx, j] := rand(4)
        end)
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: FAIL — fixed index handling not implemented

- [ ] **Step 3: Implement fixed index support**

Modify `src/LoopToVecs.jl`. The changes are in 4 areas:

**3a. Modify `_collect_rhs_inds` to use `_parse_index`:**
```julia
function _collect_rhs_inds(ex, acc::Vector{Symbol}=Symbol[])
    ex isa Expr || return acc
    if ex.head == :ref && ex.args[1] isa Symbol
        for a in ex.args[2:end]
            pi = _parse_index(a)
            if pi[1] == :loop || pi[1] == :shifted
                sym = pi[2]::Symbol
                (sym ∉ acc) && push!(acc, sym)
            end
            # :fixed indices are skipped — not loop variables
        end
    end
    for a in ex.args
        _collect_rhs_inds(a, acc)
    end
    acc
end
```

**3b. Add `_rewrite_ref_ext` — extended version of `_rewrite_ref`:**

This replaces the old `_rewrite_ref` with one that handles fixed indices via `view()`:

```julia
# extended rewrite: handles fixed, shifted, and loop indices
function _rewrite_ref_ext(A::Symbol, raw_indices, canon::Vector{Symbol};
                          shifted_ranges::Dict{Symbol,Tuple}=Dict{Symbol,Tuple}())
    parsed = [_parse_index(idx) for idx in raw_indices]

    # separate fixed from non-fixed
    has_fixed = any(p[1] == :fixed for p in parsed)
    has_shifted = any(p[1] == :shifted for p in parsed)

    # build view args: fixed positions get their value, others get ':'
    # also track which original dims are loop/shifted (remaining after view)
    view_args = Any[]
    remaining = Tuple{Int,Symbol}[]  # (orig_dim, loop_symbol)
    for (d, p) in enumerate(parsed)
        if p[1] == :fixed
            push!(view_args, p[2])
        elseif p[1] == :shifted
            sym, offset = p[2]::Symbol, p[3]::Int
            if haskey(shifted_ranges, sym)
                lo_expr, hi_expr = shifted_ranges[sym]
                push!(view_args, :(($lo_expr + $offset):($hi_expr + $offset)))
            else
                push!(view_args, :(:))
            end
            push!(remaining, (d, sym))
        else  # :loop
            push!(view_args, :(:))
            push!(remaining, (d, p[2]::Symbol))
        end
    end

    # build base expression
    if has_fixed || has_shifted
        base = :(view($A, $(view_args...)))
    else
        base = A
    end

    # now apply the existing permute + reshape logic on the remaining dims
    remaining_syms = Symbol[s for (_, s) in remaining]

    # handle repeated symbols (diagonal) — for now, just unique
    unique_remaining = _unique_syms(remaining_syms)

    # W = indices of this array in canonical order
    W = [idx for idx in canon if idx in unique_remaining]
    perm = [findfirst(==(unique_remaining[d]), W) for d in 1:length(unique_remaining)]
    needperm = any(p != d for (p, d) in zip(perm, 1:length(unique_remaining)))
    perm_expr = _tuple(perm)

    if needperm
        base = :(PermutedDimsArray($base, $perm_expr))
    end

    # shape: for each canonical index, use size(A, orig_dim) or 1
    orig_dim_for_sym = Dict{Symbol,Int}()
    for (d, s) in remaining
        haskey(orig_dim_for_sym, s) || (orig_dim_for_sym[s] = d)
    end

    shape = Any[]
    for s in canon
        if haskey(orig_dim_for_sym, s)
            if has_shifted && haskey(shifted_ranges, s)
                lo_expr, hi_expr = shifted_ranges[s]
                push!(shape, :($hi_expr - $lo_expr + 1))
            else
                push!(shape, :(size($A, $(orig_dim_for_sym[s]))))
            end
        else
            push!(shape, 1)
        end
    end
    shape_expr = _tuple(shape)

    :(reshape($base, $shape_expr))
end
```

**3c. Update `_rewrite_rhs` to use `_rewrite_ref_ext`:**
```julia
function _rewrite_rhs(ex, canon::Vector{Symbol};
                      shifted_ranges::Dict{Symbol,Tuple}=Dict{Symbol,Tuple}())
    if !(ex isa Expr)
        return ex
    elseif ex.head == :ref && ex.args[1] isa Symbol
        A = ex.args[1]::Symbol
        raw_indices = ex.args[2:end]
        return _rewrite_ref_ext(A, raw_indices, canon; shifted_ranges=shifted_ranges)
    elseif ex.head == :call
        f, args = ex.args[1], ex.args[2:end]
        args2 = map(a -> _rewrite_rhs(a, canon; shifted_ranges=shifted_ranges), args)
        ex2 = Expr(:call, f, args2...)
        return _broadcastify(ex2)
    else
        return Expr(ex.head,
            map(a -> _rewrite_rhs(a, canon; shifted_ranges=shifted_ranges), ex.args)...)
    end
end
```

**3d. Modify the macro to handle fixed indices on LHS:**

Replace the LHS parsing section with:
```julia
    # LHS: scalar symbol, or array ref with loop/fixed/shifted indices
    is_scalar_lhs = lhs isa Symbol
    local L::Symbol
    Linds = Symbol[]
    lhs_fixed = Tuple{Int,Any}[]  # (position, value) for fixed LHS indices
    has_lhs_fixed = false

    if is_scalar_lhs
        L = lhs::Symbol
    else
        (lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol) ||
            error("LHS must be a variable (scalar) or an array reference like A[i,j,...]")
        L = lhs.args[1]::Symbol
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :loop
                push!(Linds, pi[2]::Symbol)
            elseif pi[1] == :fixed
                push!(lhs_fixed, (d, pi[2]))
                has_lhs_fixed = true
            else
                error("Shifted indices on LHS are not yet supported")
            end
        end
    end

    if has_lhs_fixed && is_newvar
        error("Cannot use `:=` with fixed indices on LHS (use `=` or `+=`)")
    end
```

For the assignment, build the LHS target expression:
```julia
    # build LHS target (apply view for fixed indices)
    lhs_target = L
    if has_lhs_fixed && !is_scalar_lhs
        lhs_view_args = Any[]
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :fixed
                push!(lhs_view_args, pi[2])
            else
                push!(lhs_view_args, :(:))
            end
        end
        lhs_target = :(view($L, $(lhs_view_args...)))
    end
```

Then use `lhs_target` instead of `L` in the assignment generation:
```julia
    out = if is_mulassign
        if is_scalar_lhs
            :( $L = $L * $rhs_final )
        else
            Expr(:(.=), lhs_target, :(Base.broadcast(*, $lhs_target, $rhs_final)))
        end
    elseif is_addassign
        if is_scalar_lhs
            :( $L = $L + $rhs_final )
        else
            Expr(:(.=), lhs_target, :(Base.broadcast(+, $lhs_target, $rhs_final)))
        end
    elseif is_inplace
        if is_scalar_lhs
            :( $L = $rhs_final )
        else
            Expr(:(.=), lhs_target, rhs_final)
        end
    else
        :( $L = $rhs_final )
    end
```

Keep the old `_rewrite_ref` for backward compatibility with existing tests, or update `test/test_rewrite.jl` to call `_rewrite_ref_ext` with the new signature.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS

- [ ] **Step 5: Update `test/runtests.jl` and commit**

Add `include("test_fixed_indices.jl")` to `test/runtests.jl`.

```bash
cd D:/macro_local/LoopToVecs.jl
git add src/LoopToVecs.jl test/test_fixed_indices.jl test/runtests.jl
git commit -m "feat: add fixed index support (\$expr and integer literals)"
```

---

## Task 3: Bare loop indices in expressions

**Files:**
- Modify: `src/LoopToVecs.jl`
- Create: `test/test_bare_indices.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write bare loop index tests**

Create `test/test_bare_indices.jl`:
```julia
# test/test_bare_indices.jl
using Test
using LoopToVecs

@testset "Bare loop indices: simple comparisons" begin
    # (i <= 3) — boolean mask from loop index
    A = zeros(5, 3)
    @t A[i, j] = 1.0 * (i <= 3)
    @test A[1:3, :] == ones(3, 3)
    @test A[4:5, :] == zeros(2, 3)

    # (j == 2)
    A = zeros(5, 3)
    @t A[i, j] = 1.0 * (j == 2)
    @test A[:, 2] == ones(5)
    @test A[:, 1] == zeros(5)
    @test A[:, 3] == zeros(5)
end

@testset "Bare loop indices: two-index comparison" begin
    # (i != j)
    A = zeros(4, 4)
    @t A[i, j] = 1.0 * (i != j)
    for ii in 1:4, jj in 1:4
        @test A[ii, jj] == (ii != jj ? 1.0 : 0.0)
    end

    # (n != np) — different symbol names
    A = zeros(3, 3)
    @t A[n, np] = 5.0 * (n != np)
    for n in 1:3, np in 1:3
        @test A[n, np] == (n != np ? 5.0 : 0.0)
    end
end

@testset "Bare loop indices: combined with array refs" begin
    # ss_as[a,s] = ss_bar*(a<aR)*(s==2) — the motivating example
    num_a, num_s = 5, 2
    ss_as = zeros(num_a, num_s)
    ss_bar = 0.3
    aR = 3
    @t ss_as[a, s] = ss_bar * (a < aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test ss_as[a, s] ≈ ss_bar * (a < aR) * (s == 2)
    end

    # mixing with array reference
    B = rand(5, 2)
    A = zeros(5, 2)
    aR = 3
    @t A[a, s] = B[a, s] * (a >= aR)
    for a in 1:5, s in 1:2
        @test A[a, s] ≈ B[a, s] * (a >= aR)
    end
end

@testset "Bare loop indices: with += and reduction" begin
    # += with bare index comparison
    A = zeros(3, 3)
    @t A[n, i] += 1.0 * (n != i)
    for n in 1:3, i in 1:3
        @test A[n, i] == (n != i ? 1.0 : 0.0)
    end

    # reduction with bare index
    num_a, num_s = 5, 2
    B = rand(num_a, num_s)
    ss_bar = 0.1
    aR = 3
    @t (+) total := B[a, s] * ss_bar * (a < aR) * (s == 2)
    expected = sum(B[a, s] * ss_bar * (a < aR) * (s == 2) for a in 1:num_a, s in 1:num_s)
    @test total ≈ expected
end

@testset "Bare loop indices: no array refs on RHS" begin
    # RHS has only bare indices and scalars, range from LHS
    A = zeros(3, 3)
    @t A[i, j] = 1.0 * (i == j)
    @test A ≈ [1 0 0; 0 1 0; 0 0 1]
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: FAIL — bare loop indices not replaced with range vectors

- [ ] **Step 3: Implement bare loop index support**

Add two new functions to `src/LoopToVecs.jl`:

**3a. `_collect_index_sources` — maps each loop index to an (array, dim) for range lookup:**
```julia
# collect mapping: loop_symbol → (array_name, dimension) for axes() lookup
function _collect_index_sources(lhs, rhs, Linds::Vector{Symbol})
    sources = Dict{Symbol,Tuple{Symbol,Int}}()
    # from LHS first
    if lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol
        arr = lhs.args[1]::Symbol
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if (pi[1] == :loop || pi[1] == :shifted) && !haskey(sources, pi[2]::Symbol)
                sources[pi[2]::Symbol] = (arr, d)
            end
        end
    end
    # from RHS
    _collect_index_sources_rhs!(sources, rhs)
    return sources
end

function _collect_index_sources_rhs!(sources::Dict{Symbol,Tuple{Symbol,Int}}, ex)
    ex isa Expr || return
    if ex.head == :ref && ex.args[1] isa Symbol
        arr = ex.args[1]::Symbol
        for (d, idx) in enumerate(ex.args[2:end])
            pi = _parse_index(idx)
            if (pi[1] == :loop || pi[1] == :shifted) && !haskey(sources, pi[2]::Symbol)
                sources[pi[2]::Symbol] = (arr, d)
            end
        end
    end
    for a in ex.args
        _collect_index_sources_rhs!(sources, a)
    end
end
```

**3b. `_replace_bare_indices` — replaces bare loop index symbols with reshaped range vectors:**

Uses structural AST walking: when encountering a `:ref` node, skip its subscript arguments entirely (they are array indices, not bare occurrences). Only symbols found outside `:ref` subscripts are replaced.

```julia
# replace bare loop index symbols in RHS with reshaped range vectors
# Structurally skips :ref subscript positions (those are array indices, not bare)
function _replace_bare_indices(ex, loop_syms::Vector{Symbol}, canon::Vector{Symbol},
                               sources::Dict{Symbol,Tuple{Symbol,Int}};
                               shifted_ranges::Dict{Symbol,Tuple}=Dict{Symbol,Tuple}())
    if ex isa Expr && ex.head == :ref
        # :ref node — do NOT replace symbols in subscript positions
        # leave the entire :ref untouched (will be handled by _rewrite_rhs)
        return ex
    elseif ex isa Symbol && ex in loop_syms
        # bare loop index — replace with reshaped range vector
        if haskey(shifted_ranges, ex)
            lo_expr, hi_expr = shifted_ranges[ex]
            range_expr = :($lo_expr:$hi_expr)
        else
            arr, dim = sources[ex]
            range_expr = :(axes($arr, $dim))
        end
        canon_pos = findfirst(==(ex), canon)
        shape = Any[i == canon_pos ? :(:) : 1 for i in 1:length(canon)]
        return :(reshape($range_expr, $(Expr(:tuple, shape...))))
    elseif ex isa Expr
        new_args = Any[_replace_bare_indices(a, loop_syms, canon, sources;
                       shifted_ranges=shifted_ranges) for a in ex.args]
        return Expr(ex.head, new_args...)
    else
        return ex
    end
end
```

**3c. Integrate into the macro:**

After the RHS rewriting step, apply bare index replacement:
```julia
    # collect index sources for bare index replacement
    index_sources = _collect_index_sources(lhs, rhs, Linds)

    # rewrite RHS
    rhs_rw = _rewrite_rhs(rhs, canon)

    # replace bare loop indices with range vectors
    ref_positions = _ref_index_positions(rhs)
    rhs_rw = _replace_bare_indices(rhs_rw, collect(keys(index_sources)),
                                    canon, index_sources;
                                    ref_positions=_ref_index_positions(rhs_rw))

    rhs_b = _broadcastify(rhs_rw)
```

**Key:** Run `_replace_bare_indices` BEFORE `_rewrite_rhs`. At that point, `:ref` nodes still exist in the AST, so the structural skip works — bare symbols are replaced while `:ref` subscripts are left untouched. Then `_rewrite_rhs` handles the array refs.

```julia
    # replace bare loop indices BEFORE rewriting refs
    loop_syms_set = vcat(Linds, red_inds)
    index_sources = _collect_index_sources(lhs, rhs, Linds)
    rhs_bare = _replace_bare_indices(rhs, loop_syms_set, canon, index_sources)

    # rewrite RHS array refs
    rhs_rw = _rewrite_rhs(rhs_bare, canon)
    rhs_b = _broadcastify(rhs_rw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS

- [ ] **Step 5: Update `test/runtests.jl` and commit**

Add `include("test_bare_indices.jl")` to `test/runtests.jl`.

```bash
cd D:/macro_local/LoopToVecs.jl
git add src/LoopToVecs.jl test/test_bare_indices.jl test/runtests.jl
git commit -m "feat: add bare loop index support in expressions"
```

---

## Task 4: Shifted indices (`a+1`, `a-1`)

**Files:**
- Modify: `src/LoopToVecs.jl`
- Create: `test/test_shifted_indices.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write shifted index tests**

Create `test/test_shifted_indices.jl`:
```julia
# test/test_shifted_indices.jl
using Test
using LoopToVecs

@testset "Shifted indices: a+1 on RHS with :=" begin
    # eq[a] := N[a+1] - N[a]
    N = [1.0, 3.0, 6.0, 10.0, 15.0]
    @t eq[a] := N[a+1] - N[a]
    @test length(eq) == 4
    for a in 1:4
        @test eq[a] ≈ N[a+1] - N[a]
    end
end

@testset "Shifted indices: a+1 with scalar multiply" begin
    # eq1[a] := N[a+1]*(1+gN) - zeta_a[a]*N[a]
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

@testset "Shifted indices: := on LHS with shift → error" begin
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
    # a ranges 1:4 (from N[a+1])
    @test eq[1] ≈ 0.0            # a=1: N[2]*(1>1) = 2*0 = 0
    @test eq[2] ≈ N[3]           # a=2: N[3]*(2>1) = 3*1 = 3
    @test eq[3] ≈ N[4]           # a=3: 4
    @test eq[4] ≈ N[5]           # a=4: 5
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: FAIL

- [ ] **Step 3: Implement shifted index support**

**3a. Add range inference function:**
```julia
# Compute the intersected range for a shifted index.
# constraints: Vector of (array_name::Symbol, dim::Int, offset::Int)
# Returns (lo_expr, hi_expr) — AST expressions for the range bounds
function _compute_shifted_range(constraints::Vector{Tuple{Symbol,Int,Int}})
    # For each (A, d, k): valid a satisfies 1 <= a+k <= size(A,d)
    # i.e. max(1, 1-k) <= a <= size(A,d) - k
    lo_parts = Int[]
    hi_parts = Any[]
    for (A, d, k) in constraints
        push!(lo_parts, max(1, 1 - k))
        push!(hi_parts, k == 0 ? :(size($A, $d)) : :(size($A, $d) - $k))
    end
    lo = maximum(lo_parts)
    hi = length(hi_parts) == 1 ? hi_parts[1] : Expr(:call, :min, hi_parts...)
    return (lo, hi)
end
```

**3b. Collect shifted index constraints from all array refs:**
```julia
function _collect_shifted_constraints(ex, acc::Dict{Symbol,Vector{Tuple{Symbol,Int,Int}}})
    ex isa Expr || return acc
    if ex.head == :ref && ex.args[1] isa Symbol
        A = ex.args[1]::Symbol
        for (d, idx) in enumerate(ex.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :loop
                sym = pi[2]::Symbol
                constraints = get!(acc, sym, Tuple{Symbol,Int,Int}[])
                push!(constraints, (A, d, 0))
            elseif pi[1] == :shifted
                sym, offset = pi[2]::Symbol, pi[3]::Int
                constraints = get!(acc, sym, Tuple{Symbol,Int,Int}[])
                push!(constraints, (A, d, offset))
            end
        end
    end
    for a in ex.args
        _collect_shifted_constraints(a, acc)
    end
    acc
end
```

**3c. Integrate into the macro:**

After collecting RHS indices and computing canon:
```julia
    # collect shift constraints from both LHS and RHS
    all_constraints = Dict{Symbol,Vector{Tuple{Symbol,Int,Int}}}()
    if !is_scalar_lhs
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :loop
                cs = get!(all_constraints, pi[2]::Symbol, Tuple{Symbol,Int,Int}[])
                push!(cs, (L, d, 0))
            elseif pi[1] == :shifted
                cs = get!(all_constraints, pi[2]::Symbol, Tuple{Symbol,Int,Int}[])
                push!(cs, (L, d, pi[3]::Int))
            end
        end
    end
    _collect_shifted_constraints(rhs, all_constraints)

    # identify which indices actually have shifts
    shifted_syms = Symbol[sym for (sym, cs) in all_constraints
                          if any(c[3] != 0 for c in cs)]

    # compute shifted ranges
    shifted_ranges = Dict{Symbol,Tuple}()
    for sym in shifted_syms
        shifted_ranges[sym] = _compute_shifted_range(all_constraints[sym])
    end

    # detect shifted index on LHS
    has_lhs_shifted = false
    lhs_shifted_view_args = Any[]
    if !is_scalar_lhs
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :shifted
                has_lhs_shifted = true
            end
        end
    end

    if has_lhs_shifted && is_newvar
        error("Cannot use `:=` with shifted indices on LHS")
    end
```

Pass `shifted_ranges` to `_rewrite_rhs`:
```julia
    rhs_rw = _rewrite_rhs(rhs_bare, canon; shifted_ranges=shifted_ranges)
```

For the LHS target with shifted indices, build the view with computed ranges:
```julia
    if has_lhs_shifted && !is_scalar_lhs
        lhs_view_args = Any[]
        for (d, idx) in enumerate(lhs.args[2:end])
            pi = _parse_index(idx)
            if pi[1] == :fixed
                push!(lhs_view_args, pi[2])
            elseif pi[1] == :shifted
                sym, offset = pi[2]::Symbol, pi[3]::Int
                lo_expr, hi_expr = shifted_ranges[sym]
                push!(lhs_view_args, :(($lo_expr + $offset):($hi_expr + $offset)))
            else
                if haskey(shifted_ranges, pi[2])
                    lo_expr, hi_expr = shifted_ranges[pi[2]]
                    push!(lhs_view_args, :($lo_expr:$hi_expr))
                else
                    push!(lhs_view_args, :(:))
                end
            end
        end
        lhs_target = :(view($L, $(lhs_view_args...)))
    end
```

For bare indices with shifted ranges, update `_collect_index_sources` to handle the shifted case. The axes for a bare shifted index should use the intersected range, not `axes(A, d)`. This can be handled by checking `shifted_ranges` in `_replace_bare_indices`:
```julia
# In _replace_bare_indices, when the index has a shifted range:
    if haskey(shifted_ranges, ex)
        lo_expr, hi_expr = shifted_ranges[ex]
        range_expr = :($lo_expr:$hi_expr)
    else
        arr, dim = sources[ex]
        range_expr = :(axes($arr, $dim))
    end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS

- [ ] **Step 5: Update `test/runtests.jl` and commit**

```bash
cd D:/macro_local/LoopToVecs.jl
git add src/LoopToVecs.jl test/test_shifted_indices.jl test/runtests.jl
git commit -m "feat: add shifted index support (a+1, a-1)"
```

---

## Task 5: Diagonal/repeated indices

**Files:**
- Modify: `src/LoopToVecs.jl`
- Create: `test/test_diagonal_indices.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write diagonal index tests**

Create `test/test_diagonal_indices.jl`:
```julia
# test/test_diagonal_indices.jl
using Test
using LoopToVecs
using LinearAlgebra

@testset "Diagonal indices: basic 2-repeat" begin
    # set diagonal to 1.0
    A = zeros(3, 3)
    @t A[n, n] = 1.0
    @test A ≈ Matrix(I, 3, 3)

    # copy diagonal
    A = zeros(4, 4)
    B = rand(4, 4)
    @t A[n, n] = B[n, n]
    @test diag(A) ≈ diag(B)
    @test A[1, 2] == 0.0  # off-diagonal untouched
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
    @test A[1, 2] == 2.0  # off-diagonal unchanged
end

@testset "Diagonal indices: mixed with non-repeated dim" begin
    # A[n,n,j] = B[n,j]  — diagonal on first two dims, broadcast on third
    A = zeros(3, 3, 4)
    B = rand(3, 4)
    @t A[n, n, j] = B[n, j]
    for n in 1:3, j in 1:4
        @test A[n, n, j] ≈ B[n, j]
    end
    @test A[1, 2, 1] == 0.0  # off-diagonal
end

@testset "Diagonal indices: non-contiguous A[n,j,n]" begin
    A = zeros(3, 4, 3)
    B = rand(3, 4)
    @t A[n, j, n] = B[n, j]
    for n in 1:3, j in 1:4
        @test A[n, j, n] ≈ B[n, j]
    end
    @test A[1, 1, 2] == 0.0  # off-diagonal
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
    # A[$idx, n, n] = 1.0
    A = zeros(5, 3, 3)
    idx = 2
    @t A[$idx, n, n] = 1.0
    for n in 1:3
        @test A[2, n, n] == 1.0
    end
    @test A[1, 1, 1] == 0.0  # other fixed position
    @test A[2, 1, 2] == 0.0  # off-diagonal
end

@testset "Diagonal indices: RHS diagonal extraction" begin
    # C[n] := A[n,n] — extract diagonal
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @t C[n] := A[n, n]
    @test C ≈ [1.0, 5.0, 9.0]
end

@testset "Diagonal indices: RHS diagonal with broadcasting" begin
    # A[i,j] = B[i,i] — broadcast diagonal to all columns
    B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @t A[i, j] := B[i, i]
    # each column should be the diagonal of B
    for j in 1:3
        @test A[:, j] ≈ diag(B)
    end
end

@testset "Diagonal indices: errors" begin
    # := with diagonal on LHS → error
    @test_throws LoadError begin
        eval(quote
            A = zeros(3, 3)
            @t A[n, n] := 1.0
        end)
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: FAIL

- [ ] **Step 3: Implement diagonal index support**

This is the most complex feature. The key changes are:

**3a. Detect diagonal indices in LHS and array refs:**

In the macro, after parsing LHS indices:
```julia
    # detect diagonal: loop symbols appearing more than once on LHS
    lhs_sym_counts = Dict{Symbol,Int}()
    for s in Linds  # Linds includes duplicates at this point
        lhs_sym_counts[s] = get(lhs_sym_counts, s, 0) + 1
    end
    diag_syms = Symbol[s for (s, c) in lhs_sym_counts if c > 1]
    has_diagonal = !isempty(diag_syms)
    Linds_unique = _unique_syms(Linds)  # de-duplicated for canon

    if has_diagonal && is_newvar
        error("Cannot use `:=` with repeated indices on LHS (diagonal)")
    end
```

Use `Linds_unique` instead of `Linds` for building `canon`.

**3b. Build LHS diagonal target:**

For each diagonal symbol, build `CartesianIndex` and permutation:
```julia
function _build_diagonal_lhs(L::Symbol, lhs_parsed, diag_syms, Linds_unique)
    # Find positions of each diagonal symbol
    # Build permutation to bring diagonal positions to front
    # Build CartesianIndex expression
    # Build indexing expression: PermutedDimsArray(L, perm)[ci, :, :, ...]

    n_dims = length(lhs_parsed)

    # all positions that are diagonal
    diag_positions = Int[]
    non_diag_positions = Int[]
    for (d, pi) in enumerate(lhs_parsed)
        if pi[1] == :loop && pi[2] in diag_syms
            push!(diag_positions, d)
        elseif pi[1] != :fixed  # skip fixed, they're handled by view
            push!(non_diag_positions, d)
        end
    end

    # after view removes fixed positions, renumber
    non_fixed = [(d, pi) for (d, pi) in enumerate(lhs_parsed) if pi[1] != :fixed]
    new_diag_pos = Int[]
    new_non_diag_pos = Int[]
    for (new_d, (orig_d, pi)) in enumerate(non_fixed)
        if orig_d in diag_positions
            push!(new_diag_pos, new_d)
        else
            push!(new_non_diag_pos, new_d)
        end
    end

    # permutation: diag positions first, then non-diag
    perm = vcat(new_diag_pos, new_non_diag_pos)
    need_perm = perm != collect(1:length(non_fixed))

    # determine array source for range
    diag_sym = diag_syms[1]  # for now, handle one diagonal symbol
    # find the first LHS position for this symbol
    first_diag_orig = findfirst(d -> lhs_parsed[d][1] == :loop && lhs_parsed[d][2] == diag_sym,
                                 1:n_dims)
    n_repeats = count(d -> d in diag_positions, 1:n_dims)

    range_expr = :(axes($L, $first_diag_orig))
    ci_args = [range_expr for _ in 1:n_repeats]
    ci_expr = :(CartesianIndex.($(ci_args...)))

    # build indexing: perm_view[ci, :, :, ...]
    n_colons = length(new_non_diag_pos)
    idx_args = Any[ci_expr]
    for _ in 1:n_colons
        push!(idx_args, :(:))
    end

    return (perm=need_perm ? perm : nothing,
            ci_expr=ci_expr,
            idx_args=idx_args,
            n_repeats=n_repeats)
end
```

**3c. Handle diagonal in `_rewrite_ref_ext`:**

Add diagonal detection inside `_rewrite_ref_ext`. After building the view (for fixed/shifted), check if the remaining symbols have repeats:

```julia
    # after view, check for diagonal in remaining symbols
    sym_counts = Dict{Symbol,Int}()
    for s in remaining_syms
        sym_counts[s] = get(sym_counts, s, 0) + 1
    end
    diag_in_ref = Symbol[s for (s, c) in sym_counts if c > 1]

    if !isempty(diag_in_ref)
        # handle diagonal extraction on RHS
        dsym = diag_in_ref[1]
        diag_pos_in_remaining = findall(s -> s == dsym, remaining_syms)
        non_diag_pos = findall(s -> s != dsym, remaining_syms)

        # permute to bring diagonal positions to front
        perm_order = vcat(diag_pos_in_remaining, non_diag_pos)
        need_perm = perm_order != collect(1:length(remaining_syms))
        if need_perm
            base = :(PermutedDimsArray($base, $(_tuple(perm_order))))
        end

        # build CartesianIndex
        first_orig = remaining[diag_pos_in_remaining[1]][1]
        range_expr = :(axes($A, $first_orig))
        n_rep = length(diag_pos_in_remaining)
        ci = :(CartesianIndex.($([range_expr for _ in 1:n_rep]...)))
        n_colons = length(non_diag_pos)
        if n_colons > 0
            base = :($base[$ci, $([:(Colon()) for _ in 1:n_colons]...)])
        else
            base = :($base[$ci])
        end

        # effective syms after diagonal collapse
        effective_syms_after_diag = vcat([dsym], [remaining_syms[p] for p in non_diag_pos])
        remaining_syms = effective_syms_after_diag
        unique_remaining = _unique_syms(remaining_syms)
    end
```

Then continue with the existing permute+reshape logic using `unique_remaining` and `canon`.

**3d. Update the assignment generation to handle diagonal LHS:**

When `has_diagonal` is true, build the LHS target using `_build_diagonal_lhs`:
```julia
    if has_diagonal && !is_scalar_lhs
        diag_info = _build_diagonal_lhs(L, lhs_parsed, diag_syms, Linds_unique)
        if has_lhs_fixed
            base_target = :(view($L, $(lhs_view_args...)))
        else
            base_target = L
        end
        if diag_info.perm !== nothing
            base_target = :(PermutedDimsArray($base_target, $(_tuple(diag_info.perm))))
        end
        lhs_target = :($base_target[$(diag_info.idx_args...)])
    end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS

- [ ] **Step 5: Update `test/runtests.jl` and commit**

```bash
cd D:/macro_local/LoopToVecs.jl
git add src/LoopToVecs.jl test/test_diagonal_indices.jl test/runtests.jl
git commit -m "feat: add diagonal/repeated index support"
```

---

## Task 6: Integration tests from code_samples

**Files:**
- Create: `test/test_integration.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write integration tests**

Create `test/test_integration.jl` with patterns from the actual code_samples, adapted with explicit reductions:

```julia
# test/test_integration.jl
using Test
using LoopToVecs

@testset "Integration: setup.jl patterns" begin
    num_regions, num_sectors, num_s = 3, 2, 2
    num_x = num_regions * num_sectors * num_s

    # tau_bar pattern: @t A[n,j,s,np,jp,sp] += 5.0*(n!=np)
    tau_bar = zeros(num_regions, num_sectors, num_s, num_regions, num_sectors, num_s)
    @t tau_bar[n, j, s, np, jp, sp] += 5.0 * (n != np)
    for n in 1:num_regions, np in 1:num_regions, j in 1:num_sectors, s in 1:num_s,
        jp in 1:num_sectors, sp in 1:num_s
        @test tau_bar[n, j, s, np, jp, sp] ≈ (n != np ? 5.0 : 0.0)
    end

    # ln_d pattern: @t A[n,i,j] += 1.0*(n!=i)
    ln_d = zeros(num_regions, num_regions, num_sectors)
    @t ln_d[n, i, j] += 1.0 * (n != i)
    for n in 1:num_regions, i in 1:num_regions, j in 1:num_sectors
        @test ln_d[n, i, j] ≈ (n != i ? 1.0 : 0.0)
    end
end

@testset "Integration: solve_decision.jl patterns" begin
    num_a, num_s = 5, 2
    aR = 3
    ss_bar = 0.3
    Tss_bar = 0.1

    # @t ss_as[a,s] = ss_bar*(a<aR)*(s==2)
    ss_as = zeros(num_a, num_s)
    @t ss_as[a, s] = ss_bar * (a < aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test ss_as[a, s] ≈ ss_bar * (a < aR) * (s == 2)
    end

    # @t Tss_as[a,s] = Tss_bar*(a>=aR)*(s==2)
    Tss_as = zeros(num_a, num_s)
    @t Tss_as[a, s] = Tss_bar * (a >= aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test Tss_as[a, s] ≈ Tss_bar * (a >= aR) * (s == 2)
    end

    # $var indexing inside a "loop"
    num_x = 6
    psi1_ax = zeros(num_a, num_x)
    y_ax = rand(num_a, num_x)
    eta = 1.0

    # @t psi1_ax[$num_a, x] = 1/eta*exp(-eta*y_ax[$num_a, x])
    @t psi1_ax[$num_a, x] = 1 / eta * exp(-eta * y_ax[$num_a, x])
    @test psi1_ax[num_a, :] ≈ (1 / eta) .* exp.(-eta .* y_ax[num_a, :])
    @test psi1_ax[1, :] == zeros(num_x)  # other rows untouched
end

@testset "Integration: solve_aggregates.jl patterns" begin
    num_a, num_x = 4, 3

    # @t N_ax[1,x] = N1_x[x]
    N_ax = zeros(num_a, num_x)
    N1_x = rand(num_x)
    @t N_ax[1, x] = N1_x[x]
    @test N_ax[1, :] == N1_x
    @test N_ax[2, :] == zeros(num_x)

    # @t Tau_ax[$num_a, a] = 0.0
    Tau_ax = rand(num_a, num_a)
    old_row1 = copy(Tau_ax[1, :])
    @t Tau_ax[$num_a, a] = 0.0
    @test Tau_ax[num_a, :] == zeros(num_a)
    @test Tau_ax[1, :] == old_row1
end

@testset "Integration: solve_gN.jl patterns" begin
    # @t eq1[a] := N[a+1]*(1+gN) - zeta_a[a]*N[a]
    num_a = 5
    N = rand(num_a)
    zeta_a = rand(num_a - 1)
    gN = 0.02

    @t eq1[a] := N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    @test length(eq1) == num_a - 1
    for a in 1:(num_a-1)
        @test eq1[a] ≈ N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    end

    # scalar reduction: @t (+) eq3 += N[a]
    eq3 = -1.0
    @t (+) eq3 += N[a]
    @test eq3 ≈ -1.0 + sum(N)
end

@testset "Integration: calibration.jl patterns" begin
    num_regions = 3

    # @t tau_bar_ni[n,i] += 5.0*(n!=i)
    tau_bar_ni = zeros(num_regions, num_regions)
    @t tau_bar_ni[n, i] += 5.0 * (n != i)
    for n in 1:num_regions, i in 1:num_regions
        @test tau_bar_ni[n, i] == (n != i ? 5.0 : 0.0)
    end

    # @t resid_mu[n,n] = tau_bar_ni[n,n]
    resid_mu = zeros(num_regions, num_regions)
    @t resid_mu[n, n] = tau_bar_ni[n, n]
    @test diag(resid_mu) ≈ diag(tau_bar_ni)
    @test resid_mu[1, 2] == 0.0
end

@testset "Integration: solve_eq.jl patterns" begin
    num_regions, num_sectors = 3, 2
    num_a, num_s = 5, 2

    # @t (+) Tss_numer := ss_bar*N_a_njs[a,n,j,s]*W_nj[n,j]*e_aj[a,j]*(a<aR)*(s==2)
    N_a_njs = rand(num_a, num_regions, num_sectors, num_s)
    W_nj = rand(num_regions, num_sectors)
    e_aj = rand(num_a, num_sectors)
    ss_bar = 0.1
    aR = 3
    @t (+) Tss_numer := ss_bar * N_a_njs[a, n, j, s] * W_nj[n, j] * e_aj[a, j] * (a < aR) * (s == 2)
    expected = sum(
        ss_bar * N_a_njs[a, n, j, s] * W_nj[n, j] * e_aj[a, j] * (a < aR) * (s == 2)
        for a in 1:num_a, n in 1:num_regions, j in 1:num_sectors, s in 1:num_s
    )
    @test Tss_numer ≈ expected

    # *= pattern
    implied_W = rand(num_regions, num_sectors)
    GDP = 10.0
    orig = copy(implied_W)
    @t implied_W[n, j] *= 1 / GDP
    @test implied_W ≈ orig ./ GDP
end

@testset "Integration: calc_stats.jl patterns (high-dim reduction)" begin
    num_a, num_regions, num_sectors, num_s = 3, 2, 2, 2

    N_a_njs = rand(num_a, num_regions, num_sectors, num_s)

    # @t (+) denom_mu_n[n] := N_a_njs[a,n,j,s]  — reduce over a,j,s
    @t (+) denom_mu_n[n] := N_a_njs[a, n, j, s]
    for n in 1:num_regions
        @test denom_mu_n[n] ≈ sum(N_a_njs[:, n, :, :])
    end
end

@testset "Integration: code_sample adaptation notes" begin
    # Patterns from code_samples that use Tullio's implicit reduction
    # must be adapted with explicit (+) for LoopToVecs.
    #
    # solve_aggregates.jl for-loop patterns like:
    #   for a = 1:num_a-1
    #       @t (+) N_ax[a+1,xp] = zeta_a[a]*N_ax[a,x]*mu_ax_xp[a,x,xp]/(1+gN)
    #   end
    # require $a prefix on the external loop variable:
    #   for a = 1:num_a-1
    #       @t (+) N_ax[$a+1,xp] = zeta_a[$a]*N_ax[$a,x]*mu_ax_xp[$a,x,xp]/(1+gN)
    #   end

    # Test $a inside a for loop
    num_a, num_x = 4, 3
    N_ax = zeros(num_a, num_x)
    N_ax[1, :] .= 1.0
    zeta = [0.9, 0.8, 0.7]
    mu = ones(num_a, num_x, num_x) ./ num_x  # uniform transitions

    for a = 1:num_a-1
        @t (+) N_ax[$a+1, xp] = zeta[$a] * N_ax[$a, x] * mu[$a, x, xp]
    end
    # verify sequential accumulation worked
    @test N_ax[2, :] ≈ fill(zeta[1] / num_x, num_x)
end
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS

- [ ] **Step 3: Update `test/runtests.jl` and commit**

Add `include("test_integration.jl")` to `test/runtests.jl`.

```bash
cd D:/macro_local/LoopToVecs.jl
git add test/test_integration.jl test/runtests.jl
git commit -m "test: add integration tests from code_samples patterns"
```

---

## Task 7: Final cleanup and full test run

- [ ] **Step 1: Run the complete test suite**

Run: `cd D:/macro_local/LoopToVecs.jl && julia --project=. -e "using Pkg; Pkg.test()"`
Expected: ALL PASS — all existing tests + all new feature tests + integration tests

- [ ] **Step 2: Verify no regressions**

All existing tests in `test_helpers.jl`, `test_rewrite.jl`, `test_macro_broadcast.jl`, `test_macro_reduction.jl` should still pass unchanged.

- [ ] **Step 3: Review and update `test/test_rewrite.jl`**

The existing `test_rewrite.jl` calls `LoopToVecs._rewrite_ref` with the old signature `(A, IA, canon)`. If the function was replaced or signature changed, update the tests to use `_rewrite_ref_ext` or adapt the old function to delegate.

- [ ] **Step 4: Final commit**

```bash
cd D:/macro_local/LoopToVecs.jl
git add -A
git commit -m "chore: final cleanup after macro extensions"
```
