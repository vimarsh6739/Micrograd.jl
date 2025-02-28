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
function Value(data::T, deps::Vector{Value{T}}) where T <: Real
    # Compute the depth as the maximum depth of dependencies (or 0 if none)
    depth = isempty(deps) ? 0 : maximum(v -> v.depth, deps)
    # Sort the dependencies by their depth for proper backprop ordering
    sorted_deps = sort(deps, by = v -> v.depth)
    # Create the new Value instance
    return Value{T}(data, zero(T), sorted_deps, depth, () -> nothing)
end

# Outer constructor for a Value with no dependencies.
function Value(data::T) where T <: Real
    return Value(data, Value{T}[])
end

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
    v.grad = one(typeof(v.data))
    function _backward(v::Value)
        v._back()
        for dep in v.deps _backward(dep) end
    end
    _backward(v)
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
