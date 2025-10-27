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
            a isa Symbol && (a ∉ acc) && push!(acc, a)
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
                            :PermutedDimsArray, :tuple, :size, :axes, :length])
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

# rewrite a single A[i,j,...] into reshape(permuted(A), fullshape)
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

# rewrite all array refs on RHS using the canonical order
function _rewrite_rhs(ex, canon::Vector{Symbol})
    if !(ex isa Expr)
        return ex
    elseif ex.head == :ref && ex.args[1] isa Symbol
        A = ex.args[1]::Symbol
        IA = Symbol[ s for s in ex.args[2:end] if s isa Symbol ]
        return _rewrite_ref(A, IA, canon)
    elseif ex.head == :call
        # recurse, then broadcastify (leaving protected calls intact)
        f, args = ex.args[1], ex.args[2:end]
        args2 = map(a -> _rewrite_rhs(a, canon), args)
        ex2 = Expr(:call, f, args2...)
        return _broadcastify(ex2)
    else
        return Expr(ex.head, map(a -> _rewrite_rhs(a, canon), ex.args)...)
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
    ex.head == :(=) || is_addassign || error("Use `=` or `+=` with @t")

    lhs = ex.args[1]
    rhs = ex.args[2]

    # LHS: either scalar symbol, or array ref with plain-symbol indices
    is_scalar_lhs = lhs isa Symbol
    local L::Symbol
    Linds = Symbol[]
    if is_scalar_lhs
        L = lhs::Symbol
    else
        (lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol) ||
            error("LHS must be a variable (scalar) or an array reference like A[i,j,...]")
        any(!(lhs.args[i] isa Symbol) for i in 2:length(lhs.args)) &&
            error("LHS indices must be plain symbols, like A[i,j,...]")
        L = lhs.args[1]::Symbol
        Linds = Symbol[lhs.args[i] for i in 2:length(lhs.args)]
    end

    # RHS indices and reduction indices
    rhs_inds = _unique_syms(_collect_rhs_inds(rhs))
    red_inds = [s for s in rhs_inds if s ∉ Linds]
    if !isempty(red_inds) && redsym === nothing
        error("Found reduction indices $(red_inds) but no reduction op was given (e.g. @t (+) ...)")
    end
    canon = vcat(Linds, red_inds)

    # rewrite RHS into canonical order, then broadcastify
    rhs_rw = _rewrite_rhs(rhs, canon)
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

    # assignment
    out = if is_addassign
        if is_scalar_lhs
            :( $L = $L + $rhs_final )
        else
            :( $L = Base.broadcast(+, $L, $rhs_final) )
        end
    else
        :( $L = $rhs_final )
    end

    return esc(out)
end

end # module