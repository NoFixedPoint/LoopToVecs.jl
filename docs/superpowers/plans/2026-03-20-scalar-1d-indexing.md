# Scalar 1D Indexing Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When all indices of an array reference are fixed, return the scalar view directly instead of wrapping it in `reshape()`.

**Architecture:** Add an early return in `_rewrite_ref_ext()` after computing `remaining`. When `remaining` is empty (all indices fixed), `base` is already a scalar — return it without reshape.

**Tech Stack:** Julia, LoopToVecs.jl macro system

**Spec:** `docs/superpowers/specs/2026-03-20-scalar-1d-indexing-design.md`

---

### Task 1: Add test for all-fixed 1D array reference

**Files:**
- Modify: `test/test_fixed_indices.jl`

- [ ] **Step 1: Write the failing test**

Add a new testset at the end of `test/test_fixed_indices.jl` (before the errors testset), testing that a 1D array with a single `$var` index broadcasts correctly as a scalar on the RHS:

```julia
@testset "Fixed indices: all-fixed 1D array ref as scalar on RHS" begin
    # 1D array with single $var index should produce a scalar, not a reshaped view
    zeta = [10.0, 20.0, 30.0]
    B = rand(4)
    idx = 2
    @t result[j] := zeta[$idx] * B[j]
    @test result ≈ 20.0 .* B

    # multiple all-fixed 1D refs in one expression
    alpha = [1.0, 2.0, 3.0]
    beta_arr = [4.0, 5.0, 6.0]
    C = rand(5)
    a = 1; b = 3
    @t out[x] := alpha[$a] * C[x] + beta_arr[$b]
    @test out ≈ 1.0 .* C .+ 6.0

    # integer-literal all-fixed 1D ref
    zeta2 = [10.0, 20.0, 30.0]
    D = rand(4)
    @t res2[j] := zeta2[2] * D[j]
    @test res2 ≈ 20.0 .* D
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project -e "using Pkg; Pkg.test()"`
Expected: Tests pass in vanilla Julia (reshape on a 0-dim view works). This confirms the test logic is correct. The fix is still needed for Reactant.jl compatibility, but we verify the test is well-formed.

- [ ] **Step 3: Commit**

```bash
git add test/test_fixed_indices.jl
git commit -m "test: add all-fixed 1D array ref test cases"
```

---

### Task 2: Add early return in `_rewrite_ref_ext`

**Files:**
- Modify: `src/LoopToVecs.jl:170-171`

- [ ] **Step 1: Add early return after `remaining_syms` computation**

In `src/LoopToVecs.jl`, in `_rewrite_ref_ext()`, insert after line 170 (`remaining_syms = Symbol[s for (_, s) in remaining]`):

```julia
    # all indices were fixed → base is already a scalar, no reshape needed
    if isempty(remaining)
        return base
    end
```

- [ ] **Step 2: Run tests to verify everything passes**

Run: `julia --project -e "using Pkg; Pkg.test()"`
Expected: All tests PASS.

- [ ] **Step 3: Verify macro expansion produces no reshape for all-fixed refs**

Run: `julia --project -e "using LoopToVecs; println(@macroexpand @t result[j] := A[\$1] * B[j])"`
Expected: Output should contain `view(A, 1)` but NOT `reshape(view(A, 1), ...)`.

- [ ] **Step 4: Commit**

```bash
git add src/LoopToVecs.jl
git commit -m "fix: return scalar for all-fixed index array refs

Avoids reshape() on 0-dim views, which fails in Reactant.jl
when TracedNumber cannot be reshaped."
```

---

### Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove the resolved issue from CLAUDE.md**

Replace the current content of `CLAUDE.md` with an empty file or remove the line about 1D indexing, since the issue is now fixed.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: remove resolved 1D indexing issue from CLAUDE.md"
```
