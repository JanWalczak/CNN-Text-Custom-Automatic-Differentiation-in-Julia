module NETWORK

include("AD.jl")
using .AD
using Random
using Statistics
using Base.Threads

export Dense, Chain, binarycrossentropy, accuracy, train!, predict, Node, relu, sigma, forward!, topological_sort, Conv2D, MaxPool, AvgPool, Flatten, Embedding, AdamOptimizer, SGDOptimizer, update!, DataLoader, Trainer

mutable struct Dense
    W::AD.Node
    b::AD.Node
    activation::Function
end

function Dense(in_dim::Int, out_dim::Int, activation::Function)
    if activation === relu
        egs = sqrt(2.0f0 / in_dim)                   
    elseif activation === sigma
        egs = sqrt(1.0f0 / in_dim)              
    else
        egs = sqrt(2.0f0 / (in_dim + out_dim))    
    end
    W = AD.Node(randn(Float32, out_dim, in_dim) * egs)
    b = AD.Node(zeros(Float32, out_dim, 1))
    return Dense(W, b, activation)
end

function (layer::Dense)(x::AD.Node)
    z = layer.W * x
    return layer.activation(z .+ layer.b)
end

mutable struct Conv2D
    W::AD.Node
    b::AD.Node
    stride::Tuple{Int,Int}
    pad::Tuple{Int,Int}
    activation::Function
end

function Conv2D(cin::Int, cout::Int;
                kernel::Tuple{Int,Int} = (3,3),
                stride::Tuple{Int,Int} = (1,1),
                pad::Tuple{Int,Int}    = (0,0),
                activation::Function   = relu)

    kH,kW = kernel
    fan_in = cin * kH * kW
    std = sqrt(2f0 / fan_in)

    W = AD.Node(randn(Float32, cout, cin, kH, kW) * std)       
    b = AD.Node(zeros(Float32, cout, 1, 1, 1))                 
    Conv2D(W, b, stride, pad, activation)
end

function (layer::Conv2D)(x::AD.Node)
    z  = AD.conv2d(x, layer.W;
                   stride = layer.stride,
                   pad    = layer.pad)
    return layer.activation.(z .+ layer.b)   
end

struct MaxPool
    k::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    pad::Tuple{Int,Int}
end
MaxPool(k::Tuple{Int,Int}) = MaxPool(k, k, (0,0))

(structure::MaxPool)(x::AD.Node) =
    AD.maxpool(x; k = structure.k, stride = structure.stride, pad = structure.pad)

struct AvgPool
    k::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    pad::Tuple{Int,Int}
end
AvgPool(k::Tuple{Int,Int}) = AvgPool(k, k, (0,0))

(structure::AvgPool)(x::AD.Node) = AD.avgpool(x; k = structure.k, stride = structure.stride, pad = structure.pad)

struct Flatten end

function (f::Flatten)(x::AD.Node)
    orig = size(x.value)
    newshape = (prod(orig[1:end-1]), orig[end])   
    yval = reshape(x.value, newshape...) |> y -> convert(Array{Float32}, y)

    out = AD.Node(yval, zeros(Float32, size(yval)), Tuple{AD.Node,Function}[], "", nothing)
    out.compute = () -> reshape(x.value, newshape...)
    push!(out.parents, (x, g -> reshape(g, orig)))   
    return out
end

mutable struct Embedding
    weight::NETWORK.AD.Node
end

Embedding(V::Int, E::Int) =
    Embedding(NETWORK.AD.Node(randn(Float32, E, V) * 0.01f0))

function (layer::Embedding)(x::AD.Node)
    E, V = size(layer.weight.value)
    if ndims(x.value) != 2
        error("Embedding input expected to be 2D (sequence_length, batch_size), got $(ndims(x.value))D")
    end
    seq, N = size(x.value)
    yval = Array{Float32}(undef, E, seq, N)
    local indices::Matrix{Int}
    out = AD.Node(yval, zeros(Float32, size(yval)), Tuple{AD.Node,Function}[], "", nothing)
    out.compute = () -> begin
        indices = clamp.(trunc.(Int, x.value), 1, V)
        out.value .= reshape(layer.weight.value[:, vec(indices)], E, seq, N)
        return out.value
    end
    push!(out.parents, (layer.weight, g -> begin
        E_bwd, V_bwd = size(layer.weight.value)
        E_g, seq_bwd, N_bwd = size(g)
        idx_bwd = indices
        final_dW = zeros(Float32, E_bwd, V_bwd)
        for n in 1:N_bwd
            for t in 1:seq_bwd
                lookup_idx = idx_bwd[t, n]
                final_dW[:, lookup_idx] .+= g[:, t, n]
            end
        end
        return final_dW
    end))
    push!(out.parents, (x, g -> zeros(Float32, size(x.value))))
    return out
end

struct PermuteDimsLayer
    perm::Tuple{Vararg{Int}}
end
(p::PermuteDimsLayer)(x::AD.Node) = AD.permutedims(x, p.perm)

mutable struct Conv1D
    W::AD.Node 
    b::AD.Node
    stride::Int
    pad::Int
    activation::Function
end

function Conv1D(cin::Int, cout::Int; kernel::Int=3, stride::Int=1,
    pad::Int = (kernel-1) ÷ 2, activation=relu)

    fan_in  = cin * kernel
    fan_out = cout * kernel

    std     = sqrt(2.0f0 / (fan_in + fan_out))

    W = AD.Node(randn(Float32, cout, cin, kernel) * std)   
    b = AD.Node(zeros(Float32, cout, 1, 1))                

    Conv1D(W, b, stride, pad, activation)
end

function (layer::Conv1D)(x::AD.Node)
    z = AD.conv1d(x, layer.W; stride = layer.stride, pad = layer.pad)
    return layer.activation.(z .+ layer.b)
end

struct MaxPool1D
    k::Int
    stride::Int
    pad::Int
end
MaxPool1D(k::Int; stride::Int=k, pad::Int=0) = MaxPool1D(k, stride, pad)

function (pool::MaxPool1D)(x::AD.Node)
    y = AD.maxpool1d(x; k = pool.k, stride = pool.stride, pad = pool.pad)
    return y
end

mutable struct Chain
    layers::Vector{Any}
end

Chain(layers::Any...) = Chain(collect(layers))

function (m::Chain)(x::AD.Node)
    @inbounds @fastmath for layer in m.layers
        x = layer(x)
    end
    return x
end

function binarycrossentropy(ŷ::AD.Node, y::Vector{Float32})
    batch_size = size(ŷ.value, 2); ϵ = 1.0f-8
    y_mat = reshape(y, 1, :)
    y_node = AD.Node(y_mat)
    one = AD.Node(ones(Float32, 1, batch_size))
    loss = -(y_node .* AD.log(ŷ .+ ϵ) + (one .- y_node) .* AD.log(one .- ŷ .+ ϵ))
    return sum_node(loss) / batch_size
end

function binarycrossentropy(ŷ::AD.Node, y::AD.Node)
    batch_size = size(ŷ.value, 2); ϵ = 1.0f-8
    one = AD.Node(ones(Float32, size(ŷ.value)))
    loss_mat = -( y .* AD.log(ŷ .+ ϵ) .+ (one .- y) .* AD.log(one .- ŷ .+ ϵ) )
    return sum_node(loss_mat) / batch_size
end

function binarycrossentropy(ŷ::AD.Node, y::Real)
    ϵ = 1.0f-8
    return -(y * AD.log(ŷ + ϵ) + (1.0f0 - y) * AD.log(1.0f0 - ŷ + ϵ))
end

function accuracy(ŷ::AD.Node, y::Vector{Float32})
    batch_size = size(ŷ.value, 2)
    acc = sum((ŷ.value[1, :] .> 0.5f0) .== y) / batch_size
    return acc
end

function accuracy(ŷ::AD.Node, y::Real)
    pred = ŷ.value[1,1] > 0.5f0 ? 1.0f0 : 0.0f0
    return pred == y ? 1.0f0 : 0.0f0
end

function sum_node(x::AD.Node)
    s = AD.Node([sum(x.value)])
    s.compute = () -> [sum(x.value)]
    push!(s.parents, (x, g -> ones(Float32, size(x.value)) .* g))
    return s
end

abstract type Optimizer end

mutable struct AdamOptimizer <: Optimizer
    lr::Float32
    β1::Float32
    β2::Float32
    ϵ::Float32
    t::Int
    p::Vector  
    m::Vector{Array{Float32}}
    v::Vector{Array{Float32}}
end

function AdamOptimizer(model::Chain; lr=0.001f0, β1=0.9f0, β2=0.999f0, ϵ=1f-8)
    pars = Vector{Any}()
    for layer in model.layers
        hasproperty(layer,:W)      && push!(pars, layer.W)
        hasproperty(layer,:b)      && push!(pars, layer.b)
        hasproperty(layer,:weight) && push!(pars, layer.weight)
    end
    m = [zeros(Float32,size(p.value)) for p in pars]
    v = [zeros(Float32,size(p.value)) for p in pars]
    AdamOptimizer(Float32(lr),Float32(β1),Float32(β2),Float32(ϵ),0,pars,m,v)
end

function update!(opt::AdamOptimizer)
    opt.t += 1
    b1f = 1f0 - opt.β1
    b2f = 1f0 - opt.β2
    b1c = 1f0 - opt.β1^opt.t
    b2c = 1f0 - opt.β2^opt.t
    @inbounds for i in eachindex(opt.p)
        p  = opt.p[i]
        m  = opt.m[i]
        v  = opt.v[i]
        @. m = opt.β1 * m + b1f * p.grad
        @. v = opt.β2 * v + b2f * (p.grad * p.grad)
        m̂   = m / b1c
        v̂   = v / b2c
        @. p.value -= opt.lr * m̂ / (sqrt(v̂) + opt.ϵ)
    end
end

function update!(opt::AdamOptimizer, ::Chain)
    update!(opt)
end

mutable struct SGDOptimizer <: Optimizer
    lr::Float32
end

function update!(opt::SGDOptimizer, model::Chain)
    for layer in model.layers
        if hasproperty(layer,:W); layer.W.value .-= opt.lr .* layer.W.grad; end
        if hasproperty(layer,:b); layer.b.value .-= opt.lr .* layer.b.grad; end
        if hasproperty(layer,:weight) && !hasproperty(layer,:W)
            layer.weight.value .-= opt.lr .* layer.weight.grad
        end
    end
end


mutable struct DataLoader{TX<:AbstractArray, TY<:AbstractVector}
    X::TX
    y::TY
    batchsize::Int
    shuffle::Bool
    prefetch_epochs::Int
    batches::Vector{Vector{Vector{Int}}}
    epoch::Int
end

function DataLoader((X, y); batchsize::Int=64, shuffle::Bool=true, prefetch_epochs::Int=1)
    batches = Vector{Vector{Vector{Int}}}()
    for _ in 1:prefetch_epochs
        push!(batches, generate_epoch(size(X, ndims(X)), batchsize, shuffle))
    end
    return DataLoader(X, y, batchsize, shuffle, prefetch_epochs, batches, 1)
end

function Base.iterate(loader::DataLoader, state=nothing)
    if state === nothing
        if loader.epoch > length(loader.batches)
            push!(loader.batches, generate_epoch(size(loader.X, ndims(loader.X)), loader.batchsize, loader.shuffle))
        end
        return Base.iterate(loader, (loader.epoch, 1))
    end

    epoch, batch_idx = state
    current_batches = loader.batches[epoch]

    if batch_idx > length(current_batches)
        loader.epoch += 1
        return nothing
    end

    batch = current_batches[batch_idx]
    x = view(loader.X, :, batch)
    y = view(loader.y, batch)
    return (x, y), (epoch, batch_idx + 1)
end

function generate_epoch(N::Int, batchsize::Int, shuffle::Bool)
    idxs = shuffle ? randperm(N) : collect(1:N)
    return [idxs[i:min(i + batchsize - 1, N)] for i in 1:batchsize:N]
end



struct SimpleLoader{X<:AbstractArray,Y<:AbstractVector}
    X::X
    y::Y
    batchsize::Int
    shuffle::Bool
    idxs::Vector{Int}    
end

function SimpleLoader(X, y; batchsize=64, shuffle=true)
    N = size(X, ndims(X))
    idxs = shuffle ? randperm(N) : collect(1:N)
    SimpleLoader(X, y, batchsize, shuffle, idxs)
end

function Base.iterate(dl::SimpleLoader, state=1)
    if state > length(dl.idxs)
        if dl.shuffle
            Random.shuffle!(dl.idxs)
        end
        return nothing
    end
    last = min(state + dl.batchsize - 1, length(dl.idxs))
    batch_idxs = dl.idxs[state:last]
    x = view(dl.X, Colon(), batch_idxs)
    y = view(dl.y, batch_idxs)
    return (x, y), last + 1
end

struct Trainer
    x_node::AD.Node
    ŷ_dummy::AD.Node
    y_node::AD.Node
    loss_n::AD.Node
    graph::Vector{AD.Node}
    Nsamples::Int
end

function Trainer(model::Chain, X::AbstractArray{Float32,N}, batch_size::Int) where {N}
    feat_shape = ntuple(d -> size(X, d), N-1)
    dummy_X = ones(Float32, feat_shape..., batch_size)
    dummy_y = zeros(Float32, batch_size)
    x_node = AD.Node(dummy_X)
    ŷ_dummy = model(x_node)
    y_node = AD.Node(reshape(dummy_y, 1, :))
    loss_n = binarycrossentropy(ŷ_dummy, y_node)
    graph = AD.topological_sort(loss_n)
    Nsamples = size(X, N)
    Trainer(x_node, ŷ_dummy, y_node, loss_n, graph, Nsamples)
end

function loss(trainer::Trainer,
                X::AbstractArray{Float32,N},
                y::Vector{Float32},
                batch_size::Int) where {N}

    total_loss = 0.0f0
    total_correct = 0.0f0

    for start in 1:batch_size:trainer.Nsamples
        batch = start:min(start + batch_size - 1, trainer.Nsamples)
        bsz = length(batch)
        current_x_view = view(trainer.x_node.value, :, 1:bsz)
        current_y_view = view(trainer.y_node.value, 1, 1:bsz)
        @views current_x_view .= X[:, batch]
        current_y_view .= y[batch]
        
        AD.forward!(trainer.graph)
        batch_loss = trainer.loss_n.value[1]
        current_ŷ_view = view(trainer.ŷ_dummy.value, 1, 1:bsz)
        batch_correct = sum((current_ŷ_view .> 0.5f0) .== view(y, batch))
        total_loss += batch_loss * bsz
        total_correct += batch_correct
    end
    
    return total_loss/trainer.Nsamples, total_correct/trainer.Nsamples
end 

function train!(trainer::Trainer,
                model::Chain,
                X::AbstractArray{Float32,N},
                y::Vector{Float32},
                opt::Optimizer; 
                batch_size::Int) where {N}

    tot_loss = 0.0f0; tot_ok = 0.0f0
    idxs = collect(1:trainer.Nsamples)
    shuffle!(idxs)

    @inline @inbounds @fastmath for start in 1:batch_size:trainer.Nsamples
        batch      = idxs[start:min(start+batch_size-1, trainer.Nsamples)]
        bsz        = length(batch)
        front = ntuple(_ -> Colon(), N - 1)

        current_x_view = view(trainer.x_node.value, front..., 1:bsz)
        current_y_view = view(trainer.y_node.value, 1, 1:bsz)

        @views current_x_view .= X[front..., batch]
        current_y_view .= y[batch]

        forward!(trainer.graph)

        current_ŷ_view = view(trainer.ŷ_dummy.value, 1, 1:bsz)
        batch_loss = trainer.loss_n.value[1]
        tot_loss += batch_loss * bsz 
        tot_ok   += sum((current_ŷ_view .> 0.5f0) .== view(y, batch))

        trainer.loss_n.grad[1,1] = 1.0f0 
        backward!(trainer.graph)

        update!(opt, model)

        for node in trainer.graph 
             fill!(node.grad, 0.0f0)
        end
    end 

    mean_loss = tot_loss / trainer.Nsamples
    mean_acc  = tot_ok   / trainer.Nsamples

    return mean_loss, mean_acc
end

end
