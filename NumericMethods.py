import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt


def graph_n_function(*functions, a: float, b: float) -> None:
    # graph the function in the interval [a, b] with matplotlib
    x = np.linspace(a, b, 100)
    for f in functions:
        y = sp.lambdify(x, f, "numpy")
        plt.plot(x, y(x), label=f"f(x) = {f}")
    plt.legend()
    plt.show()
    
x = sp.symbols("x")    
func1 = x
func2 = x**2 + 6*x + 6
graph_n_function(func1, func2, a=-10, b=10)

class NumericMethods:
    def __init__(self, type_method: str, function) -> None:
        """Constructor of the class NumericMethods

        Args:
            type_method (str): Type of method to use
            function (callable): Function to evaluate
        """
        assert isinstance(type_method, str), "type_method must be a string"

        self.type_method = type_method
        self.function = function
        self.numerical_methods = {
            "bisection": self.bisection,
            "secant": self.secant,
            "false_position": self.false_position,
        }
        self.time = 0

    def check_method(self):
        if self.type_method not in self.numerical_methods:
            raise ValueError(f"Method {self.type_method} not implemented")
        else:
            return self.numerical_methods[self.type_method]

    def check_interval(self, a: float, b: float) -> bool:
        """_summary_

        Args:
            a (float): initial point
            b (float): final point

        Returns:
            bool: return True if the interval is in the domain of the function
        """
        # Check if the interval is in the domain of the function
        f_a = self.function.subs(x, a)
        f_b = self.function.subs(x, b)

        # Check limits on the interval
        limit_a_right = sp.limit(self.function, x, a, dir="+")
        limit_a_left = sp.limit(self.function, x, a, dir="-")
        limit_b_right = sp.limit(self.function, x, b, dir="+")
        limit_b_left = sp.limit(self.function, x, b, dir="-")

        # Check if the function is continuous in the interval
        is_continuous_a = f_a == limit_a_right == limit_a_left
        is_continuous_b = f_b == limit_b_right == limit_b_left

        is_continuous = is_continuous_a and is_continuous_b

        # if all the conditions are met, return True
        return is_continuous

    def graph_function(self, a: float, b: float) -> None:
        """_summary_

        Args:
            a (float): initial point
            b (float): final point
        """
        # Graph the function in the interval
        sp.plot(
            self.function,
            (x, a, b),
            show=True,
            title=f"f(x) = {self.function} in the interval [{a}, {b}]",
        )

    def bisection(self, a: float, b: float, tol: float, nmax: int) -> float:
        """_summary_

        Args:
            a (float): initial point
            b (float): final point
            tol (float): _description_
            nmax (int): _description_

        Returns:
            float: _description_
        """

        # Check if the interval is in the domain of the function
        if not self.check_interval(a, b):
            raise ValueError("Interval is not in the domain of the function")

        # Start the timer
        start_time = time.time()

        fa = self.function.subs(x, a)
        fb = self.function.subs(x, b)
        if fa * fb > 0:
            return None
        for i in range(nmax):
            p = (a + b) / 2
            fp = self.function.subs(x, p)
            if fp == 0 or (b - a) / 2 < tol:
                self.time = time.time() - start_time
                return p
            if fa * fp > 0:
                a = p
                fa = fp
            else:
                b = p
                fb = fp
        self.time = time.time() - start_time

        return None

    def secant(self, x0: float, x1: float, tol: float, nmax: int) -> float:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        # Check if the interval is in the domain of the function
        if not self.check_interval(x0, x1):
            raise ValueError("Interval is not in the domain of the function")

        # Start the timer
        start_time = time.time()

        f0 = self.function.subs(x, x0)
        f1 = self.function.subs(x, x1)

        for i in range(nmax):
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = self.function.subs(x, x2)
            if abs(f2) < tol:
                self.time = time.time() - start_time
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f2
        self.time = time.time() - start_time
        return None

    def false_position(self, a: float, b: float, tol: float, nmax: int) -> float:
        """_summary_

        Args:
            a (float): _description_
            b (float): _description_
            tol (float): _description_
            nmax (int): _description_

        Returns:
            float: _description_
        """

        # Check if the interval is in the domain of the function
        if not self.check_interval(a, b):
            raise ValueError("Interval is not in the domain of the function")

        # Start the timer
        start_time = time.time()

        fa = self.function.subs(x, a)
        fb = self.function.subs(x, b)
        if fa * fb > 0:
            return None
        for i in range(nmax):
            p = a - fa * (b - a) / (fb - fa)
            fp = self.function.subs(x, p)
            if fp == 0 or abs(fp) < tol:
                self.time = time.time() - start_time
                return p
            if fa * fp > 0:
                a = p
                fa = fp
            else:
                b = p
                fb = fp
        self.time = time.time() - start_time
        return None
    
    """ def newton_raphson(self,) """
    
    
    def __str__(self):
        return f"NumericMethods({self.type_method}, {self.function})"

    def __repr__(self):
        return f"NumericMethods({self.type_method}, {self.function})"


# Testeo

""" x = sp.symbols("x")
# Tan(x) - x
f = sp.tan(x) - x
nm = NumericMethods("bisection", f)
result = nm.check_method()(0.5, 1, 1e-6, 100)
print(nm.time)

nm = NumericMethods("secant", f)
result = nm.check_method()(0.5, 1, 1e-6, 100)
print(nm.time)

nm = NumericMethods("false_position", f)
result = nm.check_method()(0.5, 1, 1e-6, 100)
print(nm.time)

nm.graph_function(0.5, 1) """
