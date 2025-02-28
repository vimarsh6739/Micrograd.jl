module Micrograd

export Value

mutable struct Value{T <: Real}
  data::T
  grad::T
  deps::Vector{Value{T}}
  depth::Int
  _back::Function

end

# Outer constructor that takes data and an array of dependencies
function Value{T}(data::T, deps::Vector{Value{T}}) where T <: Real
    # Compute the depth as the maximum depth of dependencies (or 0 if none)
    depth = isempty(deps) ? 0 : maximum(v -> v.depth, deps)
    # Sort the dependencies by their depth for proper backprop ordering
    sorted_deps = sort(deps, by = v -> v.depth)
    # Create the new Value instance
    return Value{T}(data, zero(T), sorted_deps, depth, () -> nothing)
end

# Outer constructor for a Value with no dependencies.
function Value{T}(data::T) where T <: Real
    return Value{T}(data, Value{T}[])
end

# Constructor for explicit type conversion.
function Value(::Type{T}, data::Real) where {T<:Real}
    return Value{T}(T(data))
end

# Default constructor using Float32 when no type is provided.
Value(data::Real) = Value(Float32, data)

# For single-line display (like in arrays)
function Base.show(io::IO, v::Value{T}) where T
    print(io, "Value($(v.data))")
end

# For standalone display (more detailed)
function Base.show(io::IO, ::MIME"text/plain", v::Value{T}) where T
    println(io, "Value{$T}")
    println(io, "  data: $(v.data)")
    print(io, "  grad: $(v.grad)")
end

function backward(v::Value)
    v.grad = one(T)
    for dep in v.deps
        dep._back()
    end
end

function Base.:+(x::Value, y::Value)
    out = Value(x.data + y.data, [x, y])

    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.grad
        y.grad += out.grad
    end
    
    out._back = _backward
    return out
end

function Base.:-(x::Value, y::Value)
    out = Value(x.data - y.data, [x, y])
    
    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.grad
        y.grad -= out.grad
    end
    
    out._back = _backward
    return out
end

function Base.:*(x::Value, y::Value)
    out = Value(x.data * y.data, [x, y])
    
    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.grad * y.data
        y.grad += out.grad * x.data
    end
    
    out._back = _backward
    return out
end

function Base.:/(x::Value, y::Value)
    out = Value(x.data / y.data, [x, y])
    
    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.grad / y.data
        y.grad -= out.grad * x.data / (y.data * y.data)
    end
    
    out._back = _backward
    return out
end


function Base.:^(x::Value, y::Value)
    out = Value(x.data ^ y.data, [x, y])
    
    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.data / y.data * out.grad
        y.grad += out.grad * out.data * log(x.data)
    end
    
    out._back = _backward
    return out
end

end # module Micrograd
