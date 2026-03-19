module LoopToVecs

export @t

# preserve-first-appearance unique
_unique_syms(v::Vector{Symbol}) = begin
    seen = Set{Symbol}(); out = Symbol[]
    for x in v
        if !(x in seen)
            push!(seen, x); push!(out, x)
        end
    end
    out
end

# collect RHS indices in first-appearance order (only A[i,j] with A::Symbol)
function _collect_rhs_inds(ex, acc::Vector{Symbol}=Symbol[])
    ex isa Expr || return acc
    if ex.head == :ref && ex.args[1] isa Symbol
        for a in ex.args[2:end]
            pi = _parse_index(a)
            if pi[1] == :loop || pi[1] == :shifted
                sym = pi[2]::Symbol
                (sym ∉ acc) && push!(acc, sym)
            end
        end
    end
    for a in ex.args
        _collect_rhs_inds(a, acc)
    end
    acc
end

# tuple literal
_tuple(xs) = Expr(:tuple, xs...)

# should we convert this call to broadcast(f, ...)?
const _NO_BCAST_FUNS = Set([:reshape, :dropdims, :sum, :maximum, :minimum, :prod,
                            :PermutedDimsArray, :tuple, :size, :axes, :length,
                            :view, :CartesianIndex])
_should_broadcast(f) = (f isa Symbol) && !(f in _NO_BCAST_FUNS)

# turn calls & operators into Base.broadcast(f, args...)
function _broadcastify(ex)
    ex isa Expr || return ex
    if ex.head == :call
        f, args = ex.args[1], ex.args[2:end]
        args2 = map(_broadcastify, args)
        return _should_broadcast(f) ? :(
            Base.broadcast($f, $(args2...))
        ) : Expr(:call, f, args2...)
    elseif ex.head == :ref
        return ex # handled elsewhere
    else
        return Expr(ex.head, map(_broadcastify, ex.args)...)
    end
end

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

# rewrite a single A[i,j,...] into reshape(permuted(A), fullshape)
# (kept for backward compatibility with existing unit tests)
function _rewrite_ref(A::Symbol, IA::Vector{Symbol}, canon::Vector{Symbol})
    # W = indices of this array in canonical order
    W = [idx for idx in canon if idx in IA]
    # permutation mapping old positions -> new positions following canon
    perm = [findfirst(==(IA[d]), W) for d in 1:length(IA)]
    needperm = any(p != d for (p,d) in zip(perm, 1:length(IA)))
    perm_expr = _tuple(perm)

    # shape: for each canonical index, use size(A, pos_in_A) or 1 if absent
    pos_in_A = Dict{Symbol,Int}((s => i) for (i,s) in enumerate(IA))
    shape = Any[ haskey(pos_in_A, s) ? :(size($A, $(pos_in_A[s]))) : 1 for s in canon ]
    shape_expr = _tuple(shape)

    base = needperm ? :(PermutedDimsArray($A, $perm_expr)) : A
    :( reshape($base, $shape_expr) )
end

# extended rewrite: handles fixed ($expr/integer) indices via view()
function _rewrite_ref_ext(A::Symbol, raw_indices, canon::Vector{Symbol};
                          shifted_ranges::Dict{Symbol,Tuple}=Dict{Symbol,Tuple}())
    parsed = [_parse_index(idx) for idx in raw_indices]

    has_fixed = any(p[1] == :fixed for p in parsed)
    has_shifted = any(p[1] == :shifted for p in parsed)

    # build view args and track remaining (non-fixed) dims
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

    # permute + reshape on remaining loop dims
    remaining_syms = Symbol[s for (_, s) in remaining]
    unique_remaining = _unique_syms(remaining_syms)

    W = [idx for idx in canon if idx in unique_remaining]
    perm = [findfirst(==(unique_remaining[d]), W) for d in 1:length(unique_remaining)]
    needperm = any(p != d for (p, d) in zip(perm, 1:length(unique_remaining)))
    perm_expr = _tuple(perm)

    if needperm
        base = :(PermutedDimsArray($base, $perm_expr))
    end

    # shape: for each canon index, use size from original array or 1
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

# rewrite all array refs on RHS using the canonical order
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

# collect index sources: maps each loop index symbol to (array, dim) for range lookup
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

# replace bare loop index symbols (outside :ref nodes) with reshaped range vectors
function _replace_bare_indices(ex, loop_syms::Vector{Symbol}, canon::Vector{Symbol},
                               sources::Dict{Symbol,Tuple{Symbol,Int}};
                               shifted_ranges::Dict{Symbol,Tuple}=Dict{Symbol,Tuple}())
    if ex isa Expr && ex.head == :ref
        # :ref node — do NOT replace symbols in subscript positions
        return ex
    elseif ex isa Symbol && ex in loop_syms && (haskey(sources, ex) || haskey(shifted_ranges, ex))
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

# map (+,* ,max,min) -> (sum,prod,maximum,minimum)
const _REDMAP = Dict{Symbol,Symbol}(:+ => :sum, :* => :prod, :max => :maximum, :min => :minimum)

# dims kwarg: either an Int, or a tuple of Ints
_dims_arg(pos::Vector{Int}) = length(pos) == 1 ? pos[1] : _tuple(pos)

macro t(args...)
    # parse (optional) reducer, and the assignment expression
    redsym = nothing
    assn   = nothing
    if length(args) == 1
        assn = args[1]
    elseif length(args) == 2
        red = args[1]; assn = args[2]
        redsym = (red isa Symbol) ? red :
                 (red isa Expr && red.head==:call && length(red.args)==1 && red.args[1] isa Symbol) ?
                 (red.args[1]::Symbol) : error("Unrecognized reduction: $red")
        haskey(_REDMAP, redsym) || error("Reduction must be one of: +, *, max, min")
    else
        error("@t takes either `@t expr` or `@t (op) expr`")
    end

    ex = assn
    ex isa Expr || error("Expected an assignment expression after @t")
    is_addassign = (ex.head == :+=)
    is_mulassign = (ex.head == :*=)
    is_inplace = (ex.head == :(=))
    is_newvar = (ex.head == :(:=))
    is_newvar || is_inplace || is_addassign || is_mulassign ||
        error("Use `:=`, `=`, `+=`, or `*=` with @t")

    lhs = ex.args[1]
    rhs = ex.args[2]

    # LHS: scalar symbol, or array ref with loop/fixed indices
    is_scalar_lhs = lhs isa Symbol
    local L::Symbol
    Linds = Symbol[]
    lhs_fixed = Tuple{Int,Any}[]
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

    # RHS indices and reduction indices
    rhs_inds = _unique_syms(_collect_rhs_inds(rhs))
    red_inds = [s for s in rhs_inds if s ∉ Linds]
    if !isempty(red_inds) && redsym === nothing
        error("Found reduction indices $(red_inds) but no reduction op was given (e.g. @t (+) ...)")
    end
    if is_mulassign && redsym !== nothing
        error("Cannot combine `*=` with a reduction operator")
    end
    canon = vcat(Linds, red_inds)

    # collect index sources for bare index replacement
    index_sources = _collect_index_sources(lhs, rhs, Linds)

    # replace bare loop indices BEFORE rewriting refs
    loop_syms_set = vcat(Linds, red_inds)
    rhs_bare = _replace_bare_indices(rhs, loop_syms_set, canon, index_sources)

    # rewrite RHS array refs (on rhs_bare instead of rhs)
    rhs_rw = _rewrite_rhs(rhs_bare, canon)
    rhs_b  = _broadcastify(rhs_rw)

    # build the RHS (reduction or not)
    rhs_final = rhs_b
    if redsym !== nothing
        redfun = _REDMAP[redsym]
        if is_scalar_lhs
            # scalar reduction: reduce over all dims, no dims=... needed
            rhs_final = :( $redfun($rhs_b) )
        else
            red_pos = [findfirst(==(s), canon) for s in red_inds]
            dims_arg = _dims_arg(red_pos)
            rhs_final = :( dropdims($redfun($rhs_b; dims=$dims_arg); dims=$dims_arg) )
        end
    end

    # build LHS target (view for fixed LHS positions)
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

    # assignment
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

    return esc(out)
end

end # module
