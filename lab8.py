import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
a = 1.4
b = 4.5
c = 0.35
T = 0.025
X = 1.0

# Параметры сетки
h = 0.1  # Шаг по x
tau = 0.005  # Шаг по t
M = int(X / h) + 1  # Количество узлов по x
N = int(T / tau) + 1  # Количество узлов по t

# Функция начального условия
def y(x):
    if x <= c:
        return a
    else:
        return (a - b) / (c - 1) * x + (b * c - a) / (c - 1)

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


# Вычисление точного решения (не всегда доступно)
#  Здесь в качестве точного решения используется решение на последнем временном слое
u_exact = u[N-1, :]

# Вычисление абсолютной погрешности 
abs_error = np.abs(u[N-1, :] - u[N-2,:])


# Вывод таблицы с погрешностью (если есть точное решение)
print("\nТаблица |u - u_exact|:")
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
for j in range(0, N, int(N/6) ):  #  выбираем несколько временных слоев для отображения
    plt.plot(x_grid, u[j, :], label=f't = {t_grid[j]:.3f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Решение уравнения теплопроводности')
plt.legend()
plt.grid(True)
plt.show()