import numpy as np
import pandas as pd
import time

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
        self._dataTime = pd.DataFrame(columns=["Method", "Time", "Answer", "Error"])

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
        
        # Augmented matrix
        Ab = np.hstack([self.A, self.b.reshape(-1, 1)])
        
        # Cholesky decomposition
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = np.sqrt(Ab[i, j] - s) if (i == j) else (1.0 / L[j, j] * (Ab[i, j] - s))
        
        # Forward substitution
        y = np.zeros(n)
        for i in range(n):
            y[i] = (Ab[i, -1] - np.sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.sum(L[i, j] * x[j] for j in range(i+1, n))) / L[i, i]
        
        end_time = time.time() - start_time
        
        self._dataTime = self._dataTime._append({"Method": "Cholesky Decomposition", "Time": end_time, "Answer": x, "Error": np.linalg.norm(np.dot(self.A, x) - self.b)}, ignore_index=True)
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
    
    
# Test
A = np.array([[1, 2], [2, 1]], dtype=float)
b = np.array([5, 6], dtype=float)
eq = EquationSystem(A, b)
eq.svd_method()
eq.gauss_elimination()
eq.lu_decomposition()
#eq.cholesky_decomposition()
print(eq._dataTime)
