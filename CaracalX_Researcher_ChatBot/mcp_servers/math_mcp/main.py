from mcp.server.fastmcp import FastMCP
import sys
import os

# Redirect any stray output to stderr to keep stdout clean for MCP protocol
class SuppressStdout:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.original_stdout

# Import sympy with suppressed output
with SuppressStdout():
    from sympy import (
        symbols,
        sympify,
        solve,
        diff,
        integrate,
        factor,
        expand,
        simplify,
        Matrix,
        factorint
    )
    from math import gcd, lcm
    import statistics

mcp = FastMCP("Advanced Math Server")

@mcp.tool()
def evaluate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: 2 + 3*4"""
    try:
        return str(sympify(expression))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def simplify_expression(expression: str) -> str:
    """Simplify an algebraic expression. Example: x**2 + 2*x + x**2"""
    try:
        return str(simplify(sympify(expression)))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def factor_expression(expression: str) -> str:
    """Factor an algebraic expression. Example: x**2 - 9"""
    try:
        return str(factor(sympify(expression)))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def expand_expression(expression: str) -> str:
    """Expand an algebraic expression. Example: (x+2)*(x+3)"""
    try:
        return str(expand(sympify(expression)))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve equations. Example: equation='x**2 - 5*x + 6', variable='x'"""
    try:
        x = symbols(variable)
        result = solve(sympify(equation), x)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def derivative(expression: str, variable: str = "x") -> str:
    """Compute derivative. Example: x**3 + 2*x"""
    try:
        x = symbols(variable)
        return str(diff(sympify(expression), x))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def integral(expression: str, variable: str = "x") -> str:
    """Compute indefinite integral. Example: x**2"""
    try:
        x = symbols(variable)
        return str(integrate(sympify(expression), x))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def matrix_multiply(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices."""
    try:
        A = Matrix(matrix_a)
        B = Matrix(matrix_b)
        return (A * B).tolist()
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def matrix_determinant(matrix: list[list[float]]) -> float:
    """Calculate determinant."""
    try:
        return float(Matrix(matrix).det())
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def mean(numbers: list[float]) -> float:
    """Calculate mean of numbers."""
    try:
        return statistics.mean(numbers)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def median(numbers: list[float]) -> float:
    """Calculate median of numbers."""
    try:
        return statistics.median(numbers)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def standard_deviation(numbers: list[float]) -> float:
    """Calculate standard deviation."""
    try:
        return statistics.stdev(numbers)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def prime_factors(number: int) -> dict:
    """Return prime factorization."""
    try:
        return factorint(number)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor."""
    try:
        return gcd(a, b)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate least common multiple."""
    try:
        return lcm(a, b)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)