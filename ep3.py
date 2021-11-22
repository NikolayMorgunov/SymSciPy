import numpy as np
from sympy import symbols, Function, Eq, dsolve, solve, sqrt, lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math


def model(y, x):
    dydx = -2 * y
    return dydx


y = symbols('y', cls=Function)
x = symbols('x')
c1 = symbols('C1')

diffur = Eq(y(x).diff(), -2 * y(x))
solution = (dsolve(diffur, y(x)))
c = solve(solution, c1)[0].evalf(subs={x: 0, y(x): sqrt(2)})
res_funk = solve(solution, y(x))[0].evalf(subs={c1: c})

print('Решение SymPy y(x) =', res_funk)

xvals = np.arange(0, 10.1, 0.1)
f = lambdify(x, res_funk, 'numpy')
res_sympy = f(xvals)
fig, axs = plt.subplots(2, 1)
axs[0].plot(xvals, res_sympy)
axs[0].grid(which='major',
            color='k',
            linewidth=1)
axs[0].grid(which='minor',
            color='grey',
            linestyle=':')
axs[0].minorticks_on()


y0 = math.sqrt(2)
res_scipy = odeint(model, y0, xvals)[:, 0]

axs[0].plot(xvals, res_scipy)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y(x)')
axs[0].legend(["Решение SymPy", "Решение SciPy"])

diff = abs(res_scipy - res_sympy)

axs[1].grid(which='major',
            color='k',
            linewidth=1)
axs[1].grid(which='minor',
            color='grey',
            linestyle=':')
axs[1].minorticks_on()
axs[1].set_xlabel('x')
axs[1].set_ylabel('dy(x)')
axs[1].plot(xvals, diff)
axs[1].legend(["Разница решений SymPy и SciPy"])
fig.savefig('diffur_solution.png')
plt.show()
