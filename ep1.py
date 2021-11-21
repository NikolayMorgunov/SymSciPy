import numpy as np
from sympy import symbols, Eq, solve, Matrix, shape

rho, mu, lmbda = symbols('rho mu lambda')

matr = Matrix([[Matrix.zeros(3), -Matrix.eye(3) / rho, Matrix.zeros(3)], [-Matrix.eye(3) * mu, Matrix.zeros(3, 6)],
               [Matrix.zeros(3, 9)]])

matr_plus = Matrix([[0, 0, 0, -lmbda-mu, 0, 0, -lmbda, 0, -lmbda], [Matrix.zeros(8, 9)]]).T
matr += matr_plus

x = symbols('x')

matr -= Matrix.eye(9) * x
f = matr.det()
e = Eq(f, 0)
result = solve(e, x)
for i, val in enumerate(result):
    print(f'Собственное значение {i + 1}:', val)
