import numpy as np
from numpy import inf
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

x, y = sp.symbols("x y")

class Function:
    def __init__(self, function: sp.Expr, a=-np.inf, b=np.inf) -> None:
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
        assert isinstance(num_points, int), "num_points must be an integer"
        y_func = sp.lambdify(x, self.function, "numpy")
        x_vals = np.linspace(self.a, self.b, num_points)
        y_vals = y_func(x_vals)
        plt.plot(x_vals, y_vals, label=f"f(x) = {self.function}")
        plt.axvline(x=self.a, color='r', linestyle='--', label=f"x = {self.a}")
        plt.axvline(x=self.b, color='r', linestyle='--', label=f"x = {self.b}")
        plt.legend()
        plt.show()

    def __call__(self, x_val):
        if isinstance(x_val, (int, float, np.float64)):
            return self.function.subs(x, x_val).evalf()
        elif isinstance(x_val, tuple):
            return self.function.subs({x: x_val[0], y: x_val[1]}).evalf()
        else:
            raise ValueError("x_val must be a number or a tuple")

    def check_interval(self):
        
        test_points = np.linspace(self.a, self.b, 1000)
        test_values = [self.function.subs(x, pt).evalf() for pt in test_points]
        return all(v.is_real and v.is_finite for v in test_values)

    def check_continuity(self):
        is_continuous = True
        try:
            singularities = sp.solveset(sp.diff(self.function, x), x, domain=sp.Interval(self.a, self.b))
            is_continuous = singularities.is_EmptySet
        except Exception as e:
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
        
        g = x - self.function
        x0 = (self.a + self.b) / 2
        
        for i in range(max_iter):
            x1 = g.subs(x, x0)
            if abs(x1 - x0) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime._append({"Method": "Fixed Point", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                return x1, final_time
            else:
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
    
class OptimizationMethods(Function):
    def __init__(self, function: sp.Expr, a=-np.inf, b=np.inf) -> None:
        super().__init__(function, a, b)
        self._dataTime = pd.DataFrame(columns=["Method", "Time", "Root", "Iterations", "Max_Iterations"]).copy()
        
    def time_data(self):
        return self._dataTime
    
    def graph_time_data(self):
        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 6))
        sns.barplot(y="Method", x="Time", data=self._dataTime, palette="viridis", orient="h", order=self._dataTime.sort_values("Time", ascending=False).Method.unique())
        plt.title("Time comparison between methods")
        plt.ylabel("Method")
        plt.xlabel("Time")
        plt.plot()
        
    def graph(self, num_points=1000) -> None:
        assert isinstance(num_points, int), "num_points must be an integer"
        f_lambdified = sp.lambdify((x, y), self.function, "numpy")
        x_vals = np.linspace(self.a, self.b, num_points)
        y_vals = np.linspace(self.a, self.b, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_lambdified(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, cmap='coolwarm')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"f(x, y) = {self.function}")

        plt.show()
        
    def gradientDescentMethod(self, tol, max_iter: int, step_size=0.01):
        time_start = time.time()
        
        if not self.check_interval():
            raise ValueError("Interval is not in the domain of the function")
        
        if not self.check_continuity():
            raise ValueError("Function is not continuous in the interval")
        
        grad = sp.Matrix([sp.diff(self.function, x), sp.diff(self.function, y)])
        x0 = np.array([self.a, self.b], dtype=float)
        points = [x0]

        for i in range(max_iter):
            gradient_evaluated = np.array(grad.subs({x: x0[0], y: x0[1]})).astype(np.float64).flatten()
            x1 = x0 - step_size * gradient_evaluated
            points.append(x1)
            if np.linalg.norm(x1 - x0) < tol:
                final_time = time.time() - time_start
                self._dataTime = self._dataTime.append({"Method": "Gradient Descent", "Time": final_time, "Root": x1, "Iterations": i, "Max_Iterations": max_iter}, ignore_index=True)
                self.graph_gradient_path(points)
                return x1, final_time
            else:
                x0 = x1
        final_time = time.time() - time_start
        self._dataTime = self._dataTime.append({"Method": "Gradient Descent", "Time": final_time, "Root": x1, "Iterations": max_iter, "Max_Iterations": max_iter}, ignore_index=True)
        self.graph_gradient_path(points)
        return x1, final_time

    def graph_gradient_path(self, points):
        f_lambdified = sp.lambdify((x, y), self.function, "numpy")
        x_vals = np.linspace(self.a, self.b, 100)
        y_vals = np.linspace(self.a, self.b, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_lambdified(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], f_lambdified(points[:, 0], points[:, 1]), 'ro-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Gradient Descent Path for f(x, y) = {self.function}")

        plt.show()

# Test Optimization Methods
f = (x**2 + 3*y**2)*sp.exp(-x**2 - y**2)
f = OptimizationMethods(f, a=-2, b=2)

expr = (sp.sin(x)*sp.sin(y))/x*y
f2 = OptimizationMethods(expr, a=-2, b=2)

f.graph()
print(f.gradientDescentMethod(1e-6, 1000, step_size=0.01))
f.graph_gradient_path(f)