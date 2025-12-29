module AD

using LinearAlgebra
import Base: +, -, *, /, ^, exp, log, sin, cos, tan, abs, sqrt, show, hash, isequal, isless, promote_rule, convert, promote_shape, tanh, permutedims, reshape            
import Base.Broadcast: broadcasted, broadcast_shape 

export Node, topological_sort, backward!, reset_gradients!, forward!, sigma, relu, tanh_activation, softmax, linear

using Base.Threads

abstract type AbstractNode end

@inline to_tensor(x::Number) = reshape([Float32(x)], 1, 1)   
@inline to_tensor(x::AbstractArray{<:Real}) = convert(Array{Float32}, x)    

@inline zero_like(x::AbstractArray) = zeros(Float32, size(x))  
@inline local_one(x::AbstractArray) = ones(Float32, size(x)) 

@inline alloc_result(a::AbstractArray, b::AbstractArray) =
    zeros(Float32, broadcast_shape(size(a), size(b))...)

@inline function reduce_grad(g::Array{Float32}, orig_shape::Tuple) 
    for d in 1:ndims(g)
        o = d <= length(orig_shape) ? orig_shape[d] : 1
        if o == 1 && size(g, d) > 1
            g = sum(g, dims=d)        
        end
    end
    return g
end

mutable struct Node <: AbstractNode
    value::AbstractArray{Float32}                                   
    grad ::AbstractArray{Float32}                               
    parents::Vector{Tuple{AbstractNode, Function}}                
    name::String
    compute::Union{Nothing, Function}
end

@inline Node(x) = begin                                    
    a = to_tensor(x)
    Node(a, zero_like(a), Tuple{AbstractNode, Function}[], "", nothing)
end

promote_rule(::Type{Node}, ::Type{Real}) = Node
promote_rule(::Type{Real}, ::Type{Node}) = Node
Base.convert(::Type{Node}, x::Real)      = Node(x)

Base.length(n::Node)   = length(n.value)
Base.iterate(n::Node)  = nothing

function Base.show(io::IO, x::Node)
    print(io,
          "Node(value=$(x.value), grad=$(x.grad), #parents=$(length(x.parents)), name=$(x.name))")
end

hash(n::Node, h::UInt) = hash(objectid(n), h)
isequal(a::Node, b::Node) = a === b
@inline isless(a::Node, b::Node) = objectid(a) < objectid(b)    

Base.Broadcast.broadcastable(x::Node) = Ref(x)
@inline is_scalar(n::Node) = size(n.value) == (1, 1)

function +(x::Node, y::Node)
    result = alloc_result(x.value, y.value)                         
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .+ y.value; result)
    push!(out.parents, (x, g -> g))
    push!(out.parents, (y, g -> g))
    return out
end

function broadcasted(::typeof(+), x::Node, y::Node)
    result = alloc_result(x.value, y.value)                       
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .+ y.value; result)
    push!(out.parents, (x, g -> reduce_grad(g, size(x.value))))
    push!(out.parents, (y, g -> reduce_grad(g, size(y.value))))
    return out
end

function -(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= -x.value; result)
    push!(out.parents, (x, g -> -g))
    return out
end
-(x::Node, y::Node) = x + (-y)

function broadcasted(::typeof(-), x::Node, y::Node)
    result = alloc_result(x.value, y.value)                           
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .- y.value; result)
    push!(out.parents, (x, g -> reduce_grad(g, size(x.value))))
    push!(out.parents, (y, g -> reduce_grad(-g, size(y.value))))
    return out
end

function *(a::Real, n::Node)
    result = zeros(Float32, size(n.value))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= a .* n.value; result)
    push!(out.parents, (n, g -> a .* g))
    return out
end

function *(x::Node, y::Node)
    result = zeros(Float32, size(x.value, 1), size(y.value, 2))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (mul!(result, x.value, y.value); result)
    push!(out.parents, (x, g -> g * transpose(y.value)))
    push!(out.parents, (y, g -> transpose(x.value) * g))
    return out
end

function broadcasted(::typeof(*), x::Node, y::Node)
    result = alloc_result(x.value, y.value)
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .* y.value; result)
    push!(out.parents, (x, g -> reduce_grad(g .* y.value, size(x.value))))
    push!(out.parents, (y, g -> reduce_grad(g .* x.value, size(y.value))))
    return out
end

function /(x::Node, y::Node)
    result = alloc_result(x.value, y.value) 
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value ./ y.value; result)
    push!(out.parents, (x, g -> g ./ y.value))
    push!(out.parents, (y, g -> -(x.value ./ (y.value .^ 2)) .* g))
    return out
end

function broadcasted(::typeof(/), x::Node, y::Node)
    result = alloc_result(x.value, y.value)        
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value ./ y.value; result)
    push!(out.parents, (x, g -> reduce_grad(g ./ y.value, size(x.value))))
    push!(out.parents, (y, g -> reduce_grad(-(x.value ./ (y.value .^ 2)) .* g, size(y.value))))
    return out
end

@inline function _pow_deriv(x::Float32, n::Float32)
    if x >= 0
        dx = n * x^(n - 1.0f0)
        dn = x^n * log(x)
    else
        dx = n * x^(n - 1.0f0)
        d  = abs(x)^n * log(abs(x))
        dn = isodd(Int(round(n))) ? -d : d
    end
    return dx, dn
end

function ^(x::Node, n::Node)
    result = alloc_result(x.value, n.value)            
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .^ n.value; result)
    push!(out.parents, (x, g -> begin
        deriv = similar(x.value)
        @inbounds @fastmath for i in eachindex(x.value, n.value)
            dx, _ = _pow_deriv(x.value[i], n.value[i])
            deriv[i] = dx
        end
        g .* deriv
    end))
    push!(out.parents, (n, g -> begin
        deriv = similar(n.value)
        @inbounds @fastmath for i in eachindex(x.value, n.value)
            _, dy = _pow_deriv(x.value[i], n.value[i])
            deriv[i] = dy
        end
        g .* deriv
    end))
    return out
end

function ^(x::Node, n::Number)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .^ n; result)
    push!(out.parents, (x, g -> begin
        deriv = similar(x.value)
        @inbounds @fastmath for i in eachindex(x.value)
            dx, _ = _pow_deriv(x.value[i], n)
            deriv[i] = dx
        end
        g .* deriv
    end))
    return out
end

function broadcasted(::typeof(^), x::Node, y::Node)
    result = alloc_result(x.value, y.value)               
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .^ y.value; result)
    push!(out.parents, (x, g -> begin
        deriv = similar(x.value)
        @inbounds @fastmath for i in eachindex(x.value, y.value)
            dx, _ = _pow_deriv(x.value[i], y.value[i])
            deriv[i] = dx
        end
        reduce_grad(g .* deriv, size(x.value))
    end))
    push!(out.parents, (y, g -> begin
        deriv = similar(y.value)
        @inbounds @fastmath for i in eachindex(x.value, y.value)
            _, dy = _pow_deriv(x.value[i], y.value[i])
            deriv[i] = dy
        end
        reduce_grad(g .* deriv, size(y.value))
    end))
    return out
end

function broadcasted(::typeof(^), x::Node, n::Number)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= x.value .^ n; result)
    push!(out.parents, (x, g -> begin
        deriv = similar(x.value)
        @inbounds @fastmath for i in eachindex(x.value)
            dx, _ = _pow_deriv(x.value[i], n)
            deriv[i] = dx
        end
        reduce_grad(g .* deriv, size(x.value))
    end))
    return out
end

function broadcasted(::typeof(^), n::Number, x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{AbstractNode,Function}[], "", nothing)
    out.compute = () -> (result .= n .^ x.value; result)
    push!(out.parents, (x, g ->
        reduce_grad(g .* (result .* log(n)), size(x.value))))
    return out
end

function exp(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= exp.(x.value); result)
    push!(out.parents, (x, g -> g .* exp.(x.value)))
    return out
end

function broadcasted(::typeof(exp), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= exp.(x.value); result)   
    push!(out.parents, (x, g -> reduce_grad(g .* exp.(x.value), size(x.value))))
    return out
end

function log(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)   
    out.compute = () -> (result .= log.(x.value); result)
    push!(out.parents, (x, g -> g .* (1.0f0 ./ x.value)))
    return out
end

function broadcasted(::typeof(log), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= log.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (1.0f0 ./ x.value), size(x.value))))
    return out
end

function sin(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= sin.(x.value); result)
    push!(out.parents, (x, g -> g .* cos.(x.value)))
    return out
end

function broadcasted(::typeof(sin), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= sin.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* cos.(x.value), size(x.value))))
    return out
end

function cos(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= cos.(x.value); result)
    push!(out.parents, (x, g -> g .* (-sin.(x.value))))
    return out
end

function broadcasted(::typeof(cos), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= cos.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (-sin.(x.value)), size(x.value))))
    return out
end

function tan(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= tan.(x.value); result)
    push!(out.parents, (x, g -> g .* (1.0f0 .+ tan.(x.value).^2)))
    return out
end

function broadcasted(::typeof(tan), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= tan.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (1.0f0 .+ tan.(x.value).^2), size(x.value))))
    return out
end

function sqrt(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= sqrt.(x.value); result)
    push!(out.parents, (x, g -> g .* (0.5f0 ./ sqrt.(x.value))))
    return out
end

function broadcasted(::typeof(sqrt), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= sqrt.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (0.5f0 ./ sqrt.(x.value)), size(x.value))))
    return out
end

abs(x::Node) = broadcasted(abs, x)

function broadcasted(::typeof(abs), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= abs.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* sign.(x.value), size(x.value))))
    return out
end

function sigma(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= 1.0f0 ./(1.0f0 .+ exp.(-x.value)); result)
    push!(out.parents, (x, g -> begin
        s = 1.0f0 ./(1.0f0 .+ exp.(-x.value))
        reduce_grad(g .* (s .* (1.0f0 .- s)), size(x.value))
    end))
    return out
end

function broadcasted(::typeof(sigma), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= 1.0f0 ./(1.0f0 .+ exp.(-x.value)); result)
    push!(out.parents, (x, g -> begin
        s = 1.0f0 ./(1.0f0 .+ exp.(-x.value))
        reduce_grad(g .* (s .* (1.0f0 .- s)), size(x.value))
    end))
    return out
end

function relu(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= ifelse.(x.value .> 0, x.value, 0.0f0); result)
    push!(out.parents, (x, g -> g .* ifelse.(x.value .> 0, 1.0f0, 0.0f0)))
    return out
end

function broadcasted(::typeof(relu), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= ifelse.(x.value .> 0.0f0, x.value, 0.0f0); result)
    push!(out.parents, (x, g -> reduce_grad(g .* ifelse.(x.value .> 0f0, 1.0f0, 0.0f0), size(x.value))))
    return out
end

tanh(x::Node) = tanh_activation(x)

function tanh_activation(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= tanh.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (1.0f0 .- tanh.(x.value).^2), size(x.value))))
    return out
end

function broadcasted(::typeof(tanh_activation), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= tanh.(x.value); result)
    push!(out.parents, (x, g -> reduce_grad(g .* (1.0f0 .- tanh.(x.value).^2), size(x.value))))
    return out
end

function linear(x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= x.value; result)
    push!(out.parents, (x, g -> g))
    return out
end

function broadcasted(::typeof(linear), x::Node)
    result = zeros(Float32, size(x.value))
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> (result .= x.value; result)
    push!(out.parents, (x, g -> reduce_grad(g, size(x.value))))
    return out
end

function softmax(x::Node)
    result = zeros(Float32, size(x.value))

    m = 0.0f0
    ex = zeros(Float32, size(x.value))
    s = 0.0f0

    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    out.compute = () -> begin
        m = maximum(x.value)
        ex .= exp.(x.value .- m)
        s = sum(ex)
        result .= ex ./ s
        result
    end

    push!(out.parents, (x, g -> begin
        y = result            
        alfa = sum(y .* g)          
        y .* (g .- alfa)          
    end))
    return out
end

function broadcasted(::typeof(softmax), x::Node)
    result = zeros(Float32, size(x.value))
    m = 0.0f0
    ex = zeros(Float32, size(x.value))
    s = 0.0f0

    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)

    out.compute = () -> begin
        m = maximum(x.value)
        ex .= exp.(x.value .- m)
        s = sum(ex)
        result .= ex ./ s
        result
    end

    push!(out.parents, (x, g -> begin
        y = result
        alfa = sum(y .* g)
        y .* (g .- alfa)
    end))
    return out
end

import Base: sum
function sum(x::Node)
    result = zeros(Float32, 1, 1)
    out = Node(result, zero_like(result), Tuple{Node, Function}[], "", nothing)
    
    out.compute = () -> (result[1, 1] = sum(x.value); result)
    
    push!(out.parents, (x, g -> fill(g[1,1], size(x.value))))
    return out
end

+(x::Node, y::Number) = x + Node(y)
+(x::Number, y::Node) = Node(x) + y
-(x::Node, y::Number) = x - Node(y)
-(x::Number, y::Node) = Node(x) - y
*(x::Node, y::Number) = x * Node(y)
*(x::Number, y::Node) = Node(x) * y
/(x::Node, y::Number) = x / Node(y)
/(x::Number, y::Node) = Node(x) / y
^(x::Node, y::Number) = x ^ Node(y)
^(x::Number, y::Node) = Node(x) ^ y

function broadcasted(::typeof(+), x::Node, y::Number)
    broadcasted(+, x, Node(y))
end
function broadcasted(::typeof(+), x::Number, y::Node)
    broadcasted(+, Node(x), y)
end
function broadcasted(::typeof(-), x::Node, y::Number)
    broadcasted(-, x, Node(y))
end
function broadcasted(::typeof(-), x::Number, y::Node)
    broadcasted(-, Node(x), y)
end
function broadcasted(::typeof(*), x::Node, y::Number)
    broadcasted(*, x, Node(y))
end
function broadcasted(::typeof(*), x::Number, y::Node)
    broadcasted(*, Node(x), y)
end
function broadcasted(::typeof(/), x::Node, y::Number)
    broadcasted(/, x, Node(y))
end
function broadcasted(::typeof(/), x::Number, y::Node)
    broadcasted(/, Node(x), y)
end

function topological_sort(out::Node)
    visited = Set{Node}()
    order   = Vector{Node}()
    sizehint!(order, 100)

    function dfs(node::Node)
        if node ∈ visited
            return
        end
        push!(visited, node)
        @inbounds @fastmath for (parent, _) in node.parents
            dfs(parent :: Node)  
        end
        push!(order, node)
    end

    dfs(out)
    return order
end

function backward!(order::Vector{Node})
    output_node = last(order)
    output_node.grad .= ones(Float32, size(output_node.value))

    @inbounds @fastmath for i in length(order):-1:1
        node = order[i]
        @inbounds @fastmath for (parent, deriv) in node.parents
            parent.grad .+= deriv(node.grad)
        end
    end
end

function forward!(graph::Vector{Node})
    @inbounds @fastmath for node in graph
        if node.compute !== nothing
            node.value .= node.compute()
        end
    end
    return last(graph).value
end

function reset_gradients!(node::AbstractNode)
    visited = Set{AbstractNode}()
    stack   = Vector{AbstractNode}()
    sizehint!(stack, 100)
    push!(stack, node)

    while !isempty(stack)
        current = pop!(stack)
        current.grad .= 0
        @inbounds @fastmath for (parent, _) in current.parents
            if parent ∉ visited
                push!(visited, parent)
                push!(stack, parent)
            end
        end
    end
end

function reset_gradients!(nodes::Vector{Node})
    @inbounds @fastmath for node in nodes
        reset_gradients!(node)
    end
end

function Base.permutedims(x::Node, p)
    out_val = permutedims(x.value, p)
    out = Node(out_val, zero_like(out_val), Tuple{AbstractNode, Function}[], "", nothing)
    inv_p = invperm(p)

    out.compute = () -> begin
        permutedims(x.value, p)
    end

    push!(out.parents, (x, g -> permutedims(g, inv_p)))
    return out
end

function Base.reshape(x::Node, dims::Tuple{Vararg{Int}})
    neg1_idx = findfirst(==( -1), dims)               
    specified_N = prod(d for d in dims if d ≠ -1)
    orig_N     = length(x.value)

    new_dims = if neg1_idx === nothing        
        orig_N == specified_N ||
            throw(ArgumentError(
                "reshape: product$(dims) ≠ $orig_N"))
        dims
    else                                  
        specified_N == 0 && orig_N != 0 &&
            throw(ArgumentError(
                "reshape: cannot infer dimension from zero"))
        orig_N % specified_N == 0 ||
            throw(ArgumentError(
                "reshape: $orig_N is not divisible by $specified_N"))
        inferred = div(orig_N, specified_N)
        ntuple(i -> i == neg1_idx ? inferred : dims[i], length(dims))
    end

    size(x.value) == new_dims && return x

    out_val = Base.reshape(x.value, new_dims)  
    out     = Node(out_val,
                   zero_like(out_val),
                   Tuple{AbstractNode,Function}[],
                   "",
                   nothing)

    out.compute = () -> (out.value = Base.reshape(x.value, new_dims))

    push!(out.parents, (x, g -> Base.reshape(g, size(x.value))))
    return out
end

@inline function reshape!(dest::AbstractArray, src::AbstractArray, dims::Tuple{Vararg{Int}})
    @assert length(dest) == length(src) == prod(dims)
    copyto!(dest, src)
    return Base.reshape(dest, dims)
end

function _outHW(H,W,kH,kW,sH,sW,pH,pW)
    ((H + 2pH - kH) ÷ sH + 1,
     (W + 2pW - kW) ÷ sW + 1)
end

function conv2d(x::Node, w::Node; stride=(1,1), pad=(0,0))
    sH,sW = stride; pH,pW = pad
    Cout,Cin,kH,kW = size(w.value); Cinx,H,W,N = size(x.value)
    @assert Cin == Cinx "channels mismatch"

    outH,outW = _outHW(H,W,kH,kW,sH,sW,pH,pW)
    Wmat  = reshape(w.value, Cout, :)                      
    Xcols = zeros(Float32, Cin*kH*kW, outH*outW*N)
    Ymat  = similar(Xcols, Cout, outH*outW*N)
    yval  = zeros(Float32, Cout,outH,outW,N)

    out = Node(yval, zero_like(yval), Node[], "", nothing)

    out.compute = () -> begin
        _im2col!(Xcols, x.value, kH,kW,sH,sW,pH,pW)
        mul!(Ymat, Wmat, Xcols)
        reshape(Ymat, size(yval))              
    end

    push!(out.parents, (x, g -> begin         
        dcols = Wmat' * reshape(g, Cout, :)
        dx = zero_like(x.value)
        _col2im!(dx, dcols, kH,kW,sH,sW,pH,pW)
    end))
    push!(out.parents, (w, g -> begin          
        _im2col!(Xcols, x.value, kH,kW,sH,sW,pH,pW)
        dW = reshape(g, Cout, :) * Xcols'
        reshape(dW, size(w.value))
    end))

    out
end

function _im2col!(cols, x, kH,kW,sH,sW,pH,pW)
    C,H,W,N = size(x)
    outH,outW = _outHW(H,W,kH,kW,sH,sW,pH,pW)
    idx = 1
    @inbounds for n in 1:N, y in 0:outH-1, xw in 0:outW-1
        y0 = y*sH - pH + 1; x0 = xw*sW - pW + 1
        for c in 1:C, ky in 0:kH-1, kx in 0:kW-1
            yy=y0+ky; xx=x0+kx
            cols[idx] = (1 ≤ yy ≤ H && 1 ≤ xx ≤ W) ? x[c,yy,xx,n] : 0
            idx += 1
        end
    end
    return cols
end

function _col2im!(dx, cols, kH,kW,sH,sW,pH,pW)
    C,H,W,N = size(dx)
    outH,outW = _outHW(H,W,kH,kW,sH,sW,pH,pW)
    idx = 1
    @inbounds for n in 1:N, y in 0:outH-1, xw in 0:outW-1
        y0 = y*sH - pH + 1; x0 = xw*sW - pW + 1
        for c in 1:C, ky in 0:kH-1, kx in 0:kW-1
            yy=y0+ky; xx=x0+kx
            if 1 ≤ yy ≤ H && 1 ≤ xx ≤ W
                dx[c,yy,xx,n] += cols[idx]
            end
            idx += 1
        end
    end
    return dx
end

function conv1d(x::Node, w::Node; stride::Int = 1, pad::Int = 0)
    s = stride
    p = pad
    Cout, Cin, k = size(w.value)
    Cinx, L, N = size(x.value)
    @assert Cin == Cinx
    Lout = (L + 2p - k) ÷ s + 1
    
    Wmat = reshape(w.value, Cout, Cin*k)
    
    Xcols = similar(w.value, Cin*k, Lout*N)
    Ymat = similar(Xcols, Cout, Lout*N)
    yval = similar(x.value, Cout, Lout, N)
    
    out = Node(yval, zero_like(yval), Node[], "", nothing)
    out.compute = () -> begin
        _im2row1d_optimized!(Xcols, x.value, k, s, p)
        
        mul!(Ymat, Wmat, Xcols)
        
        return reshape(Ymat, size(yval))
    end
    
    dx_buffer = Ref{Any}(nothing)
    dcols_buffer = Ref{Any}(nothing)
    
    push!(out.parents, (x, g -> begin
        if dcols_buffer[] === nothing
            dcols_buffer[] = similar(Xcols, Cin*k, Lout*N)
        end
        if dx_buffer[] === nothing
            dx_buffer[] = zero_like(x.value)
        else
            fill!(dx_buffer[], 0f0)
        end

        mul!(dcols_buffer[], Wmat', reshape(g, Cout, :))
        _row2im1d_optimized!(dx_buffer[], dcols_buffer[], k, s, p)
        dx_buffer[]
    end))
    
    push!(out.parents, (w, g -> begin
        _im2row1d_optimized!(Xcols, x.value, k, s, p)
        dW = reshape(g, Cout, :) * Xcols'
        reshape(dW, size(w.value))
    end))
    
    return out
end

function _im2row1d_optimized!(rows, x, k, s, p)
    Cin, L, N = size(x)
    Lout = (L + 2p - k) ÷ s + 1
    CinK = Cin * k

    fill!(rows, 0f0)
    
    @inbounds for n in 1:N
        for l in 0:Lout-1
            base_idx = ((n-1)*Lout + l) * CinK
            l0 = l*s - p + 1

            if l0 >= 1 && l0+k-1 <= L
                for c in 1:Cin
                    c_offset = (c-1)*k
                    row_idx = base_idx + c_offset + 1  

                    for kw in 0:k-1
                        rows[row_idx + kw] = x[c, l0+kw, n]
                    end
                end
            else
                for c in 1:Cin
                    c_offset = (c-1)*k
                    for kw in 0:k-1
                        ll = l0 + kw
                        if 1 ≤ ll ≤ L
                            rows[base_idx + c_offset + kw + 1] = x[c, ll, n]
                        end
                    end
                end
            end
        end
    end
    return rows
end

function _row2im1d_optimized!(dx, rows, k, s, p)
    Cin, L, N = size(dx)
    Lout = (L + 2p - k) ÷ s + 1
    CinK = Cin * k
    
    @inbounds for n in 1:N
        for l in 0:Lout-1
            base_idx = ((n-1)*Lout + l) * CinK
            l0 = l*s - p + 1
            
            if l0 >= 1 && l0+k-1 <= L
                for c in 1:Cin
                    c_offset = (c-1)*k
                    row_idx = base_idx + c_offset + 1  

                    for kw in 0:k-1
                        dx[c, l0+kw, n] += rows[row_idx + kw]
                    end
                end
            else
                for c in 1:Cin
                    c_offset = (c-1)*k
                    for kw in 0:k-1
                        ll = l0 + kw
                        if 1 ≤ ll ≤ L
                            dx[c, ll, n] += rows[base_idx + c_offset + kw + 1]
                        end
                    end
                end
            end
        end
    end
    return dx
end

function maxpool(x::Node; k=(2,2), stride=k, pad=(0,0))
    kH,kW = k; sH,sW = stride; pH,pW = pad
    C,H,W,N = size(x.value)
    outH,outW = _outHW(H,W,kH,kW,sH,sW,pH,pW)
    yval = fill(Float32(-Inf), C,outH,outW,N)
    idxs = zeros(Int, C,outH,outW,N)

    out = Node(yval, zero_like(yval), Node[], "", nothing)
    out.compute = () -> begin
        @inbounds for n in 1:N, c in 1:C, oy in 0:outH-1, ox in 0:outW-1
            y0 = oy*sH - pH + 1; x0 = ox*sW - pW + 1
            best= -Inf; bi=0
            for ky in 0:kH-1, kx in 0:kW-1
                yy=y0+ky; xx=x0+kx
                if 1 ≤ yy ≤ H && 1 ≤ xx ≤ W
                    v=x.value[c,yy,xx,n]; li=(yy-1)*W+xx
                    if v>best; best=v; bi=li; end
                end
            end
            yval[c,oy+1,ox+1,n]=best
            idxs[c,oy+1,ox+1,n]=bi
        end
        yval
    end
    push!(out.parents, (x, g->begin
        dx = zero_like(x.value)
        @inbounds for n in 1:N, c in 1:C, oy in 1:outH, ox in 1:outW
            li = idxs[c,oy,ox,n]
            yy=(li-1)÷W+1; xx=li-(yy-1)*W
            dx[c,yy,xx,n] += g[c,oy,ox,n]
        end
        dx
    end))
    return out
end

function _outL(L,k,s,p)
    (L + 2p - k) ÷ s + 1
end

function maxpool1d(x::Node; k::Int=2, stride::Int=k, pad::Int=0)
    s = stride; p = pad
    Cin, L, N = size(x.value)
    Lout = _outL(L, k, s, p)
    yval = fill(Float32(-Inf), Cin, Lout, N)
    idxs = zeros(Int, Cin, Lout, N)

    out = Node(yval, zero_like(yval), Node[], "", nothing)

    out.compute = () -> begin
        @threads for n in 1:N
            for c in 1:Cin
                x_slice = view(x.value, c, :, n)
                for ol in 1:Lout
                    start = max((ol-1)*s + 1 - p, 1)
                    stop = min(start + k - 1, L)
                    if start <= stop
                        window = view(x_slice, start:stop)
                        best, idx = findmax(window)
                        yval[c, ol, n] = best
                        global_idx = start + idx - 1
                        idxs[c, ol, n] = global_idx
                    else
                        yval[c, ol, n] = -Inf32
                        idxs[c, ol, n] = 0
                    end
                end
            end
        end
        yval
    end

    push!(out.parents, (x, g -> begin
        dx = zero_like(x.value)
        @threads for n in 1:N
            for c in 1:Cin
                for ol in 1:Lout
                    li = idxs[c, ol, n]
                    if li != 0
                        dx[c, li, n] += g[c, ol, n]
                    end
                end
            end
        end
        dx
    end))
    return out
end

function avgpool(x::Node; k=(2,2), stride=k, pad=(0,0))
    kH,kW = k; sH,sW = stride; pH,pW = pad; scale = 1/(kH*kW)
    C,H,W,N = size(x.value)
    outH,outW = _outHW(H,W,kH,kW,sH,sW,pH,pW)
    yval = zeros(Float32, C,outH,outW,N)

    out = Node(yval, zero_like(yval), Node[], "", nothing)
    out.compute = () -> begin
        @inbounds for n in 1:N, c in 1:C, oy in 0:outH-1, ox in 0:outW-1
            y0=oy*sH-pH+1; x0=ox*sW-pW+1; s=0
            for ky in 0:kH-1, kx in 0:kW-1
                yy=y0+ky; xx=x0+kx
                if 1 ≤ yy ≤ H && 1 ≤ xx ≤ W
                    s += x.value[c,yy,xx,n]
                end
            end
            yval[c,oy+1,ox+1,n] = s*scale
        end
        yval
    end
    push!(out.parents, (x, g->begin
        dx = zero_like(x.value)
        @inbounds for n in 1:N, c in 1:C, oy in 0:outH-1, ox in 0:outW-1
            y0=oy*sH-pH+1; x0=ox*sW-pW+1
            grad = g[c,oy+1,ox+1,n]*scale
            for ky in 0:kH-1, kx in 0:kW-1
                yy=y0+ky; xx=x0+kx
                if 1 ≤ yy ≤ H && 1 ≤ xx ≤ W
                    dx[c,yy,xx,n] += grad
                end
            end
        end
        dx
    end))
    return out
end

end  # module AD