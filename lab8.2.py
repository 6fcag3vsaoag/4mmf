import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи (вариант 18)
a = 15.0
b = 10.0
d = 0.75  # точка излома начального условия
T = 0.025
X = 1.0

# Параметры сетки
h = 0.1          # Шаг по x
tau = 0.005      # Шаг по t
M = int(X / h) + 1  # Количество узлов по x
N = int(T / tau) + 1  # Количество узлов по t

# Функция начального условия
def y(x):
    if x <= d:
        return a
    else:
        # Линейная интерполяция от (d, a) до (1, b)
        return a + (b - a) * (x - d) / (1 - d)

# Создание сетки
x_grid = np.linspace(0, X, M)
t_grid = np.linspace(0, T, N)

# Создание массива для решения
u = np.zeros((N, M))

# Заполнение начального условия
for i in range(M):
    u[0, i] = y(x_grid[i])

# Заполнение граничных условий
for j in range(1, N):
    u[j, 0] = a
    u[j, M - 1] = b

# Явная схема
for j in range(N - 1):
    for i in range(1, M - 1):
        u[j + 1, i] = u[j, i] + tau / h**2 * (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1])

# Вывод таблицы
print("Таблица u(x, t):")
print("-" * 40)
print(f"{'i':<3} {'t':<7}", end="")
for x in x_grid:
    print(f"{x:<8.1f}", end="")
print()
print("-" * 40)
for j in range(N):
    print(f"{j:<3} {t_grid[j]:<7.3f}", end="")
    for i in range(M):
        print(f"{u[j, i]:<8.4f}", end="")
    print()

# Оценка погрешности (по разности между последними двумя слоями)
abs_error = np.abs(u[N-1, :] - u[N-2, :])

print("\nТаблица |u(t_final) - u(t_prev)| (оценка погрешности):")
print("-" * 30)
print(f"{'t':<7}", end="")
for x in x_grid:
    print(f"{x:<8.1f}", end="")
print()
print("-" * 30)
print(f"{t_grid[N-1]:<7.3f}", end="")
for i in range(M):
    print(f"{abs_error[i]:<8.4f}", end="")
print()

# Построение графиков
plt.figure(figsize=(10, 6))
for j in range(0, N, max(1, N // 6)):
    plt.plot(x_grid, u[j, :], label=f't = {t_grid[j]:.3f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Решение уравнения теплопроводности (вариант 18)')
plt.legend()
plt.grid(True)
plt.show()