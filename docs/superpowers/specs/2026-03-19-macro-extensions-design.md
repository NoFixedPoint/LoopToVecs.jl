# LoopToVecs.jl Macro Extensions Design

## Overview

Extend the `@t` macro to handle 6 additional patterns found in `code_samples/`, which currently use Tullio's `@t` macro. The existing reshape+broadcast architecture is preserved and extended with view-slicing and CartesianIndex strategies for cases that don't fit pure broadcast.

**Explicit-only policy:** No implicit reductions. All reductions require an explicit operator `(+)`, `(*)`, `(max)`, `(min)`.

**`$` required for external variables:** Any variable in index position that is not a `@t`-managed loop index must be prefixed with `$` (or be an integer literal). Plain symbols in index positions are always treated as loop indices.

## Features

### Feature 1 & 4 (unified): Fixed/Scalar Indices (`$expr` and integer literals)

**Syntax:** `$expr` or integer literal in index position.

**Semantics:** The expression is a fixed scalar value, not a loop index. The macro slices the array at that position using `view`, producing a lower-dimensional array, then applies normal reshape+broadcast to the remaining dimensions.

**Examples:**
```julia
@t psi1_ax[$num_a, x] = 1/eta*exp(-eta*y_ax[$num_a, x])
@t N_ax[1, x] = N1_x[x]
@t Tau_ax[$num_a, a] = 0.0
```

**`$` with arithmetic:** Julia parses `$a+1` as `($a)+1`, producing `Expr(:call, :+, Expr(:$, :a), 1)`. When the base of an apparent "shifted" expression is a `$`-interpolated symbol, the entire arithmetic expression is evaluated at runtime as a **fixed index** — not a shifted loop index. So `A[$a+1, x]` means "slice A at position `a+1` (a runtime value), vectorize over `x`."

**Implementation:**
- Parser recognizes `Expr(:$, expr)` and `::Integer` in index positions
- Parser also recognizes `Expr(:call, :+/-, Expr(:$, sym), int)` — this is a fixed index with value `sym + int`, NOT a shifted loop index
- These positions are excluded from loop index collection
- Array refs with fixed indices are rewritten as `view(A, fixed_val, :, :, ...)` where fixed positions get the scalar value and loop-index positions get `:`
- The resulting view is then processed through normal reshape+broadcast for remaining dimensions

**LHS constraints:**
- `:=` with fixed indices on LHS is an error (can't create a new array from a slice assignment)
- `=`, `+=`, `*=` target the view

### Feature 2: Bare Loop Indices in Expressions

**Syntax:** Loop index symbols appearing outside array subscripts — in comparisons, arithmetic, etc.

**Examples:**
```julia
@t ss_as[a,s] = ss_bar*(a<aR) * (s==2)
@t tau_bar_njs_npjpsp[n,j,s,np,jp,sp] += 5.0*(n!=np)
```

**Semantics:** Replace bare loop index with a range vector reshaped to broadcast in its canonical position. Bare loop indices evaluate to 1-based integer positions (matching Julia's default array indexing).

**Implementation:**
- After collecting loop indices, scan the RHS for bare occurrences of those symbols (i.e., not inside array subscript positions)
- Replace each bare loop index `a` with `axes(ref_array, pos)` where `ref_array` is any array that uses `a` as an index, and `pos` is the position of `a` in that array
- Reshape this range vector to have `1`s in all non-canonical positions so it broadcasts correctly
- Example: if canon is `[a, s]`, then bare `a` becomes `reshape(axes(ss_as, 1), (:, 1))` and bare `s` becomes `reshape(axes(ss_as, 2), (1, :))`

### Feature 3: `*=` Compound Assignment

**Syntax:** `@t A[i,j] *= expr`

**Semantics:**
- Scalar LHS: `L = L * rhs_final`
- Array LHS: `L .= Base.broadcast(*, L, rhs_final)`
- Diagonal LHS: `L[ci, ...] .= Base.broadcast(*, L[ci, ...], rhs_final)`
- Fixed-index LHS: `view(L, ...) .= Base.broadcast(*, view(L, ...), rhs_final)`

**Implementation:** Add `*=` alongside `+=` in the assignment operator parsing. Julia parses `*=` as `Expr(:*=, lhs, rhs)`.

**Constraint:** `*=` combined with a reduction operator (e.g., `@t (+) A[i] *= B[i,j]`) is an error.

### Feature 5: Shifted Indices (`a+1`, `a-1`)

**Syntax:** `a+k` or `a-k` where `a` is a `@t`-managed loop index (plain symbol, NOT `$`-prefixed) and `k` is an integer literal.

**Distinguishing from fixed `$a+1`:** If the base symbol is `$`-interpolated, it is a fixed index (Feature 1). If the base symbol is a plain symbol that also appears as a bare loop index elsewhere in the expression, it is a shifted loop index (Feature 5).

**Examples:**
```julia
@t eq1[a] := N[a+1]*(1+gN) - zeta_a[a]*N[a]
```

**Semantics:** The loop index `a` still exists, but its valid range is the intersection of all constraints:
- `A[a]` → `a ∈ 1:size(A, pos)`
- `A[a+k]` → `a ∈ 1:size(A, pos)-k`
- `A[a-k]` → `a ∈ 1+k:size(A, pos)`
- Intersection of all ranges for `a` gives the final range

**Important: No sequential dependencies.** Shifted indices vectorize across the full intersected range simultaneously. For recurrences where iteration `a` depends on iteration `a-1`, use an explicit for-loop with `$a` fixed indices instead.

**Implementation:**
- Parser detects `Expr(:call, :+, sym, int)` or `Expr(:call, :-, sym, int)` in index positions where `sym` is a plain Symbol (not `$`-wrapped)
- The base symbol is collected as a loop index
- For each array reference, compute the valid range constraint for each shifted index
- Compute the intersection across all arrays
- Rewrite array references using `view` with the appropriate range:
  - `N[a]` with `a ∈ 1:L-1` → `view(N, 1:L-1)`
  - `N[a+1]` with `a ∈ 1:L-1` → `view(N, 2:L)`
- After slicing, apply reshape+broadcast for alignment with other dimensions

**LHS constraints:**
- `:=` with shifted indices **on the LHS** is an error
- `:=` is allowed when only the RHS has shifts — the new array is sized according to the intersected range length
- `=`, `+=`, `*=` target the view with the computed range

### Feature 6: Repeated/Diagonal Indices

**Syntax:** Same loop index appearing multiple times on LHS.

**Examples:**
```julia
@t mu_ax_xp[$num_a, x, x] = 1.0
@t resid_mu[n,n] = tau_bar_ni[n,n]
```

**Semantics:** Operates on the "diagonal" — the subset of elements where repeated index positions are equal. Fully vectorized using `PermutedDimsArray` + `CartesianIndex`.

**Non-contiguous support:** Repeated positions need NOT be contiguous. When they are not (e.g., `A[n, j, n]`), a `PermutedDimsArray` permutation brings repeated positions to the front before applying `CartesianIndex`. Since `PermutedDimsArray` is a zero-cost view, modifications go through to the original array.

**Implementation:**
- Detect when a loop index appears more than once on the LHS
- For a repeated index `n` appearing at k positions `[p1, p2, ..., pk]` (possibly non-contiguous) on the LHS:
  - Compute a permutation that moves all repeated positions to the front, preserving order of non-repeated positions after them
  - Apply `PermutedDimsArray(A, perm)` to get a view where repeated positions are contiguous and leading
  - Generate `ci = CartesianIndex.(range, range, ...)` with k entries
  - On LHS: `PermutedDimsArray(A, perm)[ci, :, :]` where `:` fills non-repeated positions
  - On RHS arrays with same pattern: `PermutedDimsArray(B, perm)[ci, :]` to extract diagonal values
  - If positions are already contiguous and leading, skip the permutation step
- After extraction, the repeated positions collapse to a single dimension
- The collapsed dimension participates in canonical order normally
- Non-repeated dimensions use normal reshape+broadcast
- Supports any number of repetitions (2, 3, ...)

**LHS constraints:**
- `:=` with repeated indices on LHS is an error (can't create a new array from diagonal assignment)
- `=`, `+=`, `*=` use CartesianIndex assignment

## Architecture: Multi-Phase Pipeline

The macro is refactored from one monolithic function into distinct phases:

### Phase 1: `_parse_assignment(args...)`
- Extract reduction operator, assignment operator (`:=`, `=`, `+=`, `*=`), LHS, RHS
- Classify each index position: `:loop`, `:fixed`, `:shifted`, with associated metadata

### Phase 2: `_classify_indices(lhs_info, rhs_info)`
- Collect all loop indices (first-appearance order)
- Identify fixed indices, shifted indices, diagonal indices
- Compute reduction indices (RHS-only loop indices)
- Build canonical order
- For shifted indices: compute range constraints and intersect
- For bare loop indices in RHS expressions: identify locations for replacement

### Phase 3: `_rewrite_rhs(rhs, canon, index_info)`
- Rewrite array references:
  - Normal: reshape + PermutedDimsArray (as today)
  - Fixed positions: view first, then reshape remaining
  - Shifted positions: view with range, then reshape remaining
  - Diagonal on RHS: CartesianIndex extraction, then reshape
- Replace bare loop indices with reshaped range vectors
- Broadcastify operators (with `view` and `CartesianIndex` added to `_NO_BCAST_FUNS`)

### Phase 4: `_apply_reduction(rhs, redsym, canon, lhs_inds, red_inds)`
- Same as today: `sum`/`prod`/`maximum`/`minimum` with `dims=...`

### Phase 5: `_generate_assignment(L, lhs_info, rhs_final, op)`
- Normal: as today
- Fixed LHS: assign into `view(L, ...)`
- Shifted LHS: assign into `view(L, range, ...)`
- Diagonal LHS: assign into `L[ci, ...]`
- Handles `:=`, `=`, `+=`, `*=`

## Testing Plan

Extensive tests for each feature, organized by feature:

1. **Fixed indices:** `$var`, `$expr`, integer literals, on LHS, on RHS, combined with broadcasting, combined with reduction, `$a+1` arithmetic
2. **Bare loop indices:** comparisons (`<`, `==`, `!=`, `>=`), arithmetic on indices, combined with array references, multi-index comparisons like `(n!=np)`
3. **`*=` operator:** scalar, array, combined with other features; error when combined with reduction
4. **Shifted indices:** `a+1`, `a-1`, multiple shifted refs, shifted on LHS (error), shifted on RHS with `:=`, range inference correctness, verify no sequential dependency issues
5. **Diagonal indices:** 2-repeat, 3-repeat, mixed with non-repeated dims, mixed with fixed indices, on both LHS and RHS, non-contiguous repeated positions (e.g., `A[n,j,n]`)
6. **Integration tests:** Patterns directly from `code_samples/` (adapted with explicit `(+)` where needed)
7. **Error tests:** `:=` with fixed/shifted/diagonal LHS errors, missing reduction op errors, `*=` with reduction error

## Code Sample Adaptation Note

The `code_samples/` were written for Tullio which supports implicit summation. When testing these patterns with LoopToVecs, explicit reduction operators `(+)` must be added where Tullio would implicitly sum. For example:
- `@t numer_mu_ni[n,i] := ...` with extra RHS indices → needs `@t (+) numer_mu_ni[n,i] := ...`
- `@t GDP := gamma_jL[j]*X_nj[n,j]` → needs `@t (+) GDP := ...`
