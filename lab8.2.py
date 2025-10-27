import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи для ВАРИАНТА 18
a = 15.0
b = 10.0
c = 27.00
d = 0.75
T = 0.025  # Конечное время
X = 1.0    # Длина отрезка по x

# Параметры сетки
h = 0.1    # Шаг по x
tau = 0.005  # Шаг по t
M = int(X / h) + 1  # Количество узлов по x
N = int(T / tau) + 1  # Количество узлов по t

# Функция начального условия для ВАРИАНТА 18
def y(x):
    if x <= d:
        # Линейная часть от 0 до d: y = ((c - a) / d) * x + a
        return ((c - a) / d) * x + a
    else:
        # Линейная часть от d до 1: y = ((c - b) / (d - 1)) * x + (b * d - c) / (d - 1)
        return ((c - b) / (d - 1)) * x + (b * d - c) / (d - 1)

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

# Вычисление абсолютной погрешности (сравнение с предыдущим слоем)
abs_error = np.abs(u[N-1, :] - u[N-2, :])

# Вывод таблицы с погрешностью
print("\nТаблица |u(t_final) - u(t_final-tau)|:")
print("-" * 30)
print(f"{'t':<5} {'x':<7}", end="")
for i in range(M):
    print(f"{x_grid[i]:<8.1f}", end="")
print()
print("-" * 30)
print(f"{t_grid[N-1]:<7.3f}", end="")  # Print the final time step
for i in range(M):
     print(f"{abs_error[i]:<8.4f}", end="")  
print()

# Построение графиков
plt.figure(figsize=(10, 6))
for j in range(0, N, max(1, int(N/6)) ):  # Выбираем несколько временных слоев для отображения
    plt.plot(x_grid, u[j, :], label=f't = {t_grid[j]:.3f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Решение уравнения теплопроводности (Вариант 18)')
plt.legend()
plt.grid(True)
plt.show()