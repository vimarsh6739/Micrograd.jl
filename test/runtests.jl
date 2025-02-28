# test/runtests.jl
using Test
using Micrograd

# Define the test function based on your translated code.
function f(x::Value, y::Value)
    return (x + Value(2.0)) * (x + Value(2.0)) + (y - Value(3.0)) * (y - Value(3.0))
end

@testset "Micrograd Test" begin
    loss = 1
    epsilon = 1e-10
    rate = 0.05
    x_val = 0.0
    y_val = 0.0

    while loss > epsilon
        x = Value(x_val)
        y = Value(y_val)
        z = f(x, y)
        Micrograd.backward(z)

        # show(stdout, "text/plain", x); println()
        # show(stdout, "text/plain", y); println()
        # show(stdout, "text/plain", z); println()
        println("$(x.data) $(y.data)")

        x_val -= x.grad * rate
        y_val -= y.grad * rate
        loss = abs(x.grad) + abs(y.grad)
    end
end

    # epsilon = 1e-6
    # rate = 0.0001

    # g = f(a, b)
    # backward!(g)  # Assuming your package defines backward!(g) to compute gradients

    # # Run a fixed number of iterations or until convergence.
    # for i in 1:100
    #     @test isapprox(g.data, 24.7041; atol=1e-4)   # example test on g.data
    #     @test isapprox(a.grad, 138.8338; atol=1e-4)     # example test on a.grad
    #     @test isapprox(b.grad, 645.5773; atol=1e-4)     # example test on b.grad

    #     println(@sprintf("%.4f", g.data))
    #     println(@sprintf("%.4f", a.grad))
    #     println(@sprintf("%.4f", b.grad))

    #     a -= a.grad * rate
    #     b -= b.grad * rate

    #     println(@sprintf("a: %.4f b: %.4f", a.data, b.data))

    #     g = f(a, b)
    #     backward!(g)
    # end
# end