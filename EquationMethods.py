import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class EquationSystem:
    def __init__(self, A: np.ndarray, b: np.ndarray, tolerance=1e-16) -> None:
        assert isinstance(A, np.ndarray), "A must be a numpy array"
        assert isinstance(b, np.ndarray), "b must be a numpy array"
        assert isinstance(tolerance, (int, float, np.float64)), "tolerance must be a number"
        self.A = A
        self.b = b
        self.tolerance = tolerance
        self._dataTime = pd.DataFrame(columns=["Method", "Time", "Answer", "Error", "Iterations"])

    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, A):
        assert isinstance(A, np.ndarray), "A must be a numpy array"
        self._A = A
        
    @A.deleter
    def A(self):
        del self._A
        
    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        assert isinstance(b, np.ndarray), "b must be a numpy array"
        self._b = b
        
    @b.deleter
    def b(self):
        del self._b
        
    @property
    def tol(self):
        return self._tolerance
    
    @tol.setter
    def tol(self, tol):
        assert isinstance(tol, (int, float, np.float64)), "tol must be a number"
        self._tolerance = tol
        
    @tol.deleter
    def tol(self):
        del self._tolerance
        
    def __str__(self):
        return f"A = {self.A}\nb = {self.b}\ntol = {self.tol}"
    
    def __repr__(self):
        return f"A = {self.A}\nb = {self.b}\ntol = {self.tol}"
    
    def __del__(self):
        del self.A
        del self.b
        del self.tolerance
        del self._dataTime
        
    def graph_time(self):
        """Plot the time of execution for each method
        """

        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 6))
        sns.barplot(y="Method", x="Time", data=self._dataTime, palette="viridis", orient="h", order=self._dataTime.sort_values("Time", ascending=False)["Method"])
        plt.ylabel("Method")
        plt.xlabel("Time (s)")
        plt.title("Time of Execution for Different Methods")
        plt.xticks(rotation=45)

        plt.show()
    
    def check_symmetric(self): # Check if the matrix is symmetric A = A^T
        return np.allclose(self.A, self.A.T, atol=self.tolerance)
        
    def check_square(self): # Check if the matrix is square (nxn)
        return self.A.shape[0] == self.A.shape[1]
    
    def check_positive_definite(self): # Check if the matrix is positive definite
        return np.all(np.linalg.eigvals(self.A) > 0)
    
    def check_singular(self): # Check if the matrix is singular
        
        if self.check_square():
            return np.linalg.det(self.A) == 0
        else:
            return np.linalg.matrix_rank(self.A) < min(self.A.shape)
    
    def gauss_elimination(self):
        assert self.check_square(), "Matrix is not square"
        start_time = time.time()
        n = len(self.b)

        # Augmented matrix
        Ab = np.hstack([self.A, self.b.reshape(-1, 1)])
        
        # Forward elimination
        for i in range(n):
            # Make the diagonal contain all 1's
            Ab[i] = Ab[i] / Ab[i, i]
            
            # Make the elements below the pivot elements equal to zero
            for j in range(i+1, n):
                Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "Gauss Elimination", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x      
    
    def lu_decomposition(self):
        assert self.check_square(), "Matrix is not square"
        assert not self.check_singular(), "Matrix is singular"
        
        start_time = time.time()
        n = len(self.b)
        
        # Augmented matrix
        Ab = np.hstack([self.A, self.b.reshape(-1, 1)])
        
        # LU decomposition
        for i in range(n):
            # Make the diagonal contain all 1's
            Ab[i] = Ab[i] / Ab[i, i]
            
            # Make the elements below the pivot elements equal to zero
            for j in range(i+1, n):
                Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "LU Decomposition", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x

    def cholesky_decomposition(self):
        assert self.check_symmetric(), "Matrix is not symmetric"
        assert self.check_square(), "Matrix is not square"
        assert self.check_positive_definite(), "Matrix is not positive definite"
        assert not self.check_singular(), "Matrix is singular"
        
        start_time = time.time()
        n = len(self.b)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(self.A)
        
        # Forward substitution
        y = np.zeros(n)
        for i in range(n):
            y[i] = (self.b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]
        
        # Back substitution
        x = np.zeros(n)
        LT = L.T
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.sum(LT[i, i+1:] * x[i+1:])) / LT[i, i]
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "Cholesky Decomposition", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x
    
    def QR_decomposition(self):
        assert self.check_square(), "Matrix is not square"
        assert not self.check_singular(), "Matrix is singular"
        
        start_time = time.time()
        n = len(self.b)
        
        # QR decomposition
        Q, R = np.linalg.qr(self.A)
        
        # Solve for y
        y = np.dot(Q.T, self.b)
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.sum(R[i, i+1:] * x[i+1:])) / R[i, i]
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "QR Decomposition", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x
    
    def svd_method(self):
        """Solve the equation system using the SVD method

        Returns:
            np.array: Solution to the equation system
        """
        
        start_time = time.time()
        # Primer paso: Descomponer la matriz A en sus componentes U, Sigma y VT
        U, Sigma, VT = np.linalg.svd(self.A)
        Sigma_inv = np.diag(1 / Sigma) # Convertir Sigma en una matriz diagonal
        
        # Resolver para y
        c = np.dot(U.T, self.b)
        y = np.dot(Sigma_inv, c)
        
        # Resolver para x
        x = np.dot(VT.T, y)
        
        final_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "SVD", "Time": final_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x
 
    # Iterative methods -------------------------------------------------------
    def jacobi_method(self, max_iter: int):
        assert self.check_square(), "Matrix is not square"
        assert not self.check_singular(), "Matrix is singular"
        
        start_time = time.time()
        n = len(self.b)
        
        # Initialize x
        x = np.zeros(n)
        
        for _ in range(max_iter):
            x_new = np.zeros(n)
            for i in range(n):
                x_new[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, i+1:], x[i+1:])) / self.A[i, i]
            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            x = x_new
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "Jacobi", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x
    
    def gauss_seidel_method(self, max_iter: int):
        assert self.check_square(), "Matrix is not square"
        assert not self.check_singular(), "Matrix is singular"
        
        start_time = time.time()
        n = len(self.b)
        
        # Initialize x
        x = np.zeros(n)
        
        for _ in range(max_iter):
            x_old = np.copy(x)
            for i in range(n):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, i+1:], x[i+1:])) / self.A[i, i]
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "Gauss Seidel", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
        return x
    
    
# Test
A = np.array([[4, 1, 1], [1, 3 ,0], [1, 0, 2]], dtype=float)
b = np.array([6, 5, 3], dtype=float)
eq = EquationSystem(A, b)
print(eq.svd_method())
print(eq.gauss_elimination())
print(eq.lu_decomposition())
print(eq.QR_decomposition())
print(eq.cholesky_decomposition())
print(eq.jacobi_method(100))
print(eq.gauss_seidel_method(100))
print(eq._dataTime)

eq.graph_time()
