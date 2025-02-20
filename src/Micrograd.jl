module Micrograd

export Value

mutable struct Value{T <: Real}
  data::T
  grad::T
  _back::Function 
  
  # default constructor
  Value{T}(data::T) where {T<:Real} = new{T}(data, zero(T), () -> nothing)
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

function Base.:+(x::Value, y::Value)
    out = Value(x.data + y.data)
    
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
    out = Value(x.data - y.data)
    
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
    out = Value(x.data * y.data)
    
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
    out = Value(x.data / y.data)
    
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
    out = Value(x.data^y.data)
    
    # Define the backward pass that updates gradients of inputs
    function _backward()
        # During backprop, gradient flows equally to both inputs
        # out.grad is the gradient flowing from further up the chain
        x.grad += out.data / y.data *  out.grad
        y.grad += out.grad * out.data * log(x.data)
    end
    
    out._back = _backward
    return out
end

# Outer constructors
# This one handles when type parameter is explicitly given
Value{T}(data::Real) where {T<:Real} = Value{T}(T(data))

# This one defaults to Float32 when no type parameter is given
Value(data::Real) = Value{Float32}(Float32(data))
end # module Micrograd
