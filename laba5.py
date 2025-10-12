import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sympy import symbols, sin, sympify, integrate, lambdify


def galerkin_coeffs(N, a, b, f_expr_str):
    x = symbols('x')
    L = b - a
    f = sympify(f_expr_str, locals={'x': x})
    coeffs = []

    for n in range(1, N + 1):
        phi_n_sym = sin(n * np.pi * (x - a) / L)  
        integral = integrate(f * phi_n_sym, (x, a, b))
        a_n = -(2 * L / (n**2 * np.pi**2)) * integral
        coeffs.append(float(a_n))
    return np.array(coeffs)

def y_approx(x_vals, a, b, coeffs):
    L = b - a
    pi = np.pi
    x_vals = np.asarray(x_vals)
    y_vals = np.zeros_like(x_vals, dtype=float)

    # Быстрый векторный расчёт
    N = len(coeffs)
    n = np.arange(1, N + 1).reshape(-1, 1)              # (N,1)
    X = x_vals.reshape(1, -1)                            # (1,M)
    S = np.sin(n * pi * (X - a) / L)                     # (N,M)
    y_vals = (coeffs.reshape(-1, 1) * S).sum(axis=0)     # (M,)
    return y_vals

def main():
    print("Метод Галёркина для y''(x) = f(x), y(a)=y(b)=0 на [a,b]")
    # Ввод
    a = float(input("Введите левую границу области a: ").strip())
    b = float(input("Введите правую границу области b: ").strip())
    while b <= a:
        print("Ошибка: правая граница должна быть больше левой.")
        b = float(input("Введите правую границу области b: ").strip())

    f_expr = input("Введите выражение для f(x): ").strip()

    N = int(input("Введите количество базисных функций: ").strip())
    while N < 3:
        N = int(input("Ошибка: нужно не менее 3. Повторите ввод: ").strip())

    M = int(input("Введите количество узлов сетки: ").strip())
    while M < 2:
        M = int(input("Ошибка: минимум 2 узла. Повторите ввод: ").strip())

    # Сетка
    x_vals = np.linspace(a, b, M)

    # Решение
    coeffs = galerkin_coeffs(N, a, b, f_expr)
    y_vals = y_approx(x_vals, a, b, coeffs)

    # Таблица
    table_rows = []
    for k in range(M):
        table_rows.append([k, f"{x_vals[k]:.4f}", f"{y_vals[k]:.4f}"])

    print("\nТаблица решений:")
    print(tabulate(table_rows, headers=["k", "x_k", "y_k"], tablefmt="grid"))

    # Максимальный прогиб (минимум y)
    i_min = np.argmin(y_vals)
    x_min, y_min = x_vals[i_min], y_vals[i_min]
    print(f"\nМаксимальный прогиб: y_min = y({x_min:.4f}) = {y_min:.4f}")

    # График
    xx = np.linspace(a, b, 1001)
    yy = y_approx(xx, a, b, coeffs)

    plt.figure(figsize=(8, 4))
    plt.plot(xx, yy, color="blue", label="Прогиб балки")
    plt.scatter([x_min], [y_min], color="red", zorder=3, label=f"Макс. прогиб: {y_min:.4f}")
    plt.title("Прогиб балки")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
