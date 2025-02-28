# test/runtests.jl
using Test
using Micrograd

# Define the test function based on your translated code.
function f(a::Value, b::Value)
    c = a + b
    return c
end

@testset "Micrograd Test" begin
    a = Value(0.0)
    # b = Value(0.0)

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