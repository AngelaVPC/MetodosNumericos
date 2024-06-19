import numpy as np
from numpy import inf
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

x = sp.symbols("x")

class Function:
    def __init__(self, function: sp.Expr, a=-np.inf, b=np.inf) -> None:
        """Constructor of the class Function

        Args:
            function (sp.Expr): Sympy expression representing the function
            a (numeric, optional): initial point. Defaults to -np.inf.
            b (numeric, optional): initial point. Defaults to np.inf.
        """
        self.function = function
        self.a = a
        self.b = b

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        assert isinstance(function, sp.Expr), "function must be a sympy expression"
        self._function = function

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, (int, float, np.float64)), "a must be a number"
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        assert isinstance(b, (int, float, np.float64)), "b must be a number"
        self._b = b

    def graph(self, num_points = 1000 ) -> None:
        """ Plot the function in the interval [a, b]

        Args:
            num_points (int, optional): _description_. Defaults to 1000.
        """
        assert isinstance(num_points, int), "num_points must be an integer"
        y = sp.lambdify(x, self.function, "numpy")
        x_vals = np.linspace(self.a, self.b, num_points)
        y_vals = y(x_vals)
        plt.plot(x_vals, y_vals, label=f"f(x) = {self.function}")
        plt.axvline(x=self.a, color='r', linestyle='--', label=f"x = {self.a}")
        plt.axvline(x=self.b, color='r', linestyle='--', label=f"x = {self.b}")
        plt.legend()
        plt.show()

    def __call__(self, x_val):
        """ Evaluate the function at a given point

        Args:
            x_val (numeric): Point to evaluate the function 

        Returns:
            numeric: Value of the function evaluated at x_val
        """
        #assert isinstance(x_val, (int, float, np.float64)), "x_val must be a number"
        return self.function.subs(x, x_val)

    def check_interval(self):
        """ Check if the function is real and finite in the interval [a, b]

        Returns:
            Bool: True if the function is real and finite in the interval [a, b]
        """
        # Check a dense set of points in the interval
        test_points = np.linspace(self.a, self.b, 1000)
        test_values = [self.function.subs(x, pt).evalf() for pt in test_points]

        # Check if all evaluated points are real and finite
        return all(v.is_real and v.is_finite for v in test_values)

    def check_continuity(self):
        """ Check if the function is continuous in the interval [self.a, self.b]

        Returns:
            Bool: True if the function is continuous in the interval [self.a, self.b], False otherwise
        """
        # Definir una variable para el resultado
        is_continuous = True
        
        # Intentar encontrar discontinuidades en la función
        try:
            singularities = sp.solveset(sp.diff(self.function, x), x, domain=sp.Interval(self.a, self.b))
            # Si el conjunto de singularidades es vacío, no hay discontinuidades
            is_continuous = singularities.is_EmptySet
        except Exception as e:
            # Si no se puede determinar (por ejemplo, si la derivada no se puede calcular), asumir que no es continua
            is_continuous = False

        return is_continuous

    def __repr__(self):
        return f"f(x) = {self.function}"
    
class NumericMethods(Function):
    def __init__(self, function: sp.Expr, a=-np.inf, b=np.inf, tolerance=1e-6):
        super().__init__(function, a, b)
        self.tolerance = tolerance
        self._dataTime = pd.DataFrame(columns=["Method", "Time", "Root", "Iterations", "Max_Iterations"]).copy()

    def time_data(self):
        return self._dataTime
    
    def fixed_point(self, tol, max_iter: int):
        """ Find the root of the function using the fixed point method

        Args:
            tol (float): Tolerance for the root
            max_iter (int): Maximum number of iterations

        Raises:
            ValueError: If the interval is not in the domain of the function

        Returns:
            Tuple[float, float]: Root of the function and time to find it
        """
        time_start = time.time()
        
        if not self.check_interval():
            raise ValueError("Interval is not in the domain of the function")
        
        g = x + self.function
        x0 = (self.a + self.b) / 2
        
        for i in range(max_iter):
            x1 = g.subs(x, x0)
            if abs(x1 - x0) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime._append({"Method": "Fixed Point", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                return x1, final_time
            x0 = x1
        final_time = time.time() - time_start
        self._dataTime = self._dataTime._append({"Method": "Fixed Point", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
        
        return x1, final_time

    def bisection(self, tol, max_iter: int):
        """ Find the root of the function using the bisection method

        Args:
            tol (float): Tolerance for the root
            max_iter (int): Maximum number of iterations

        Raises:
            ValueError: If the interval is not in the domain of the function
            ValueError: If the function does not have roots in the interval (no sign change)

        Returns:
            Tuple[float, float]: Root of the function and time to find it
        """
        time_start = time.time()
        
        if not self.check_interval():
            raise ValueError("Interval is not in the domain of the function")

        a, b = self.a, self.b
        fa, fb = self(a), self(b)

        if fa * fb >= 0:
            raise ValueError("Function does not have roots in the interval (no sign change)")

        for i in range(max_iter):
            p = (a + b) / 2
            fp = self(p)
            if abs(fp) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime._append({"Method": "Bisection", "Time": final_time, "Root": p, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                return p , final_time
            elif fa * fp < 0:
                b = p
                fb = fp
            else:
                a = p
                fa = fp
        final_time = time.time() - time_start
        self._dataTime = self._dataTime._append({"Method": "Bisection", "Time": final_time, "Root": p, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
    
        return (a + b) / 2,  final_time
    
    def secant(self, tol, max_iter: int):
        
        time_start = time.time()
        
        if not self.check_interval():
            raise ValueError("Interval is not in the domain of the function")
        
        if not self.check_continuity():
            raise ValueError("Function is not continuous in the interval")
        
        x0, x1 = self.a, self.b
        f0, f1 = self(x0), self(x1)
        
        if f0 * f1 >= 0:
            raise ValueError("Function does not have roots in the interval (no sign change)")
        
        for i in range(max_iter):
            p = (x0 + x1) / 2
            fp = self(p)
            if abs(fp) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime._append({"Method": "Secant", "Time": final_time, "Root": p, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                return p, final_time
            else:
                x0, x1 = x1, p
                f0, f1 = f1, fp
        final_time = time.time() - time_start
        self._dataTime = self._dataTime._append({"Method": "Secant", "Time": final_time, "Root": p, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
        
        return p, final_time
    
    def newton_raphson(self, tol, max_iter: int):
        
        time_start = time.time()
        
        if not self.check_interval():
            raise ValueError("Interval is not in the domain of the function")
        
        if not self.check_continuity():
            raise ValueError("Function is not continuous in the interval")
        
        x0 = self.a
        f0 = self(x0)
        df = sp.diff(self.function, x)
        
        for i in range(max_iter):
            x1 = x0 - f0 / df.subs(x, x0)
            f1 = self(x1)
            if abs(f1) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime._append({"Method": "Newton-Raphson", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                return x1, final_time
            else:
                x0, f0 = x1, f1
        final_time = time.time() - time_start
        self._dataTime = self._dataTime._append({"Method": "Newton-Raphson", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
        
        return x1, final_time
    
#f = sp.exp(-2*x) - 2*x + 1
f = x**3 -x - 1
nm = NumericMethods(f, 1, 2)
nm.graph()
nm.fixed_point(1e-4, 100)


print(nm.time_data())