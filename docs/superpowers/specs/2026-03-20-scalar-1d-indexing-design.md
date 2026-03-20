# Fix: Return scalar for all-fixed index array references

**Date:** 2026-03-20

## Problem

When a 1D array is indexed with all-fixed indices (e.g., `zeta_a[$a]`, `psi2_a[$ap]`), `_rewrite_ref_ext` produces `reshape(view(zeta_a, a), (1,))`. In Reactant.jl, the `view()` of a single element yields a `TracedNumber`, and calling `reshape` on a `TracedNumber` errors.

### Concrete example

```julia
@t psi3_ax[$a,x] = (
    log( zeta_a[$a]*beta*psi1_ax[$ap,x]*eta*psi2_a[$ap] )
    / eta / ( 1 + psi2_a[$ap] )
)
```

Here `zeta_a[$a]` and `psi2_a[$ap]` are 1D arrays with a single fixed index. The generated code wraps these in `reshape(view(...), (1,))`, which fails under Reactant.

## Solution

**Approach A: Early return in `_rewrite_ref_ext` when `remaining` is empty.**

After computing `remaining` (the list of non-fixed dimension/symbol pairs) at line 170, if `remaining` is empty, return `base` immediately. The scalar already broadcasts correctly with all other operands — no permutation or reshape is needed.

### Change location

`src/LoopToVecs.jl`, function `_rewrite_ref_ext()`, after line 170.

### Code change

```julia
remaining_syms = Symbol[s for (_, s) in remaining]

# all indices were fixed → base is already a scalar, no reshape needed
if isempty(remaining)
    return base
end
```

## Testing

Add a test verifying that all-fixed 1D array refs produce scalar values, not reshaped views.

## Scope

This is a ~3-line change. No-op reshapes (e.g., 1D view reshaped to 1D) are not in scope — they work fine in Reactant.
