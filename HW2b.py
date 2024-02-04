import math


# Define the secant method function
def secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    for i in range(maxiter):
        fx0 = fcn(x0)
        fx1 = fcn(x1)
        if fx1 - fx0 == 0:
            raise ValueError("Division by zero encountered in secant method")

        # Compute the next estimate using the secant method
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Check if the convergence criterion is met
        if abs(x2 - x1) < xtol:
            return x2  # Converged to a solution

        # Update the estimates for the next iteration
        x0, x1 = x1, x2

    return x1  # Return the last estimate if maxiter is reached


# Define the main function that calls the secant function with specified parameters
def main():
    # First function x - 3cos(x)
    root1 = secant(lambda x: x - 3 * math.cos(x), x0=1, x1=2, maxiter=5, xtol=1e-4)
    print(f"Root of x-3cos(x)=0: {root1}")

    # Second function cos(2x) - x^3
    root2 = secant(lambda x: math.cos(2 * x) - x ** 3, x0=1, x1=2, maxiter=15, xtol=1e-8)
    print(f"Root of cos(2x)-x^3=0: {root2}")

    # Third function cos(2x) - x^3 with different maxiter and xtol
    root3 = secant(lambda x: math.cos(2 * x) - x ** 3, x0=1, x1=2, maxiter=3, xtol=1e-8)
    print(f"Root of cos(2x)-x^3=0 (different maxiter and xtol): {root3}")


# Execute the main function
if __name__ == "__main__":
    main()
