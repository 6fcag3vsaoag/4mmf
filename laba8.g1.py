import numpy as np
import matplotlib.pyplot as plt

# --- ПАРАМЕТРЫ ЗАДАЧИ (ВАРИАНТ 18) ---
# !!! ВНИМАНИЕ: Замените эти значения на свои из варианта 18 !!!
a = 1.2      # Левое граничное условие u(0, t)
b = 0.7      # Правое граничное условие u(1, t)
c = 0.4      # Точка разрыва начального условия
d = 0.8      # Параметр из формулы для x > c
T = 0.025    # Время моделирования
X = 1.0      # Длина стержня (согласно формуле на изображении, отрезок [0, 1])

# --- ПАРАМЕТРЫ СЕТКИ ---
h = 0.1      # Шаг по x
tau = 0.005  # Шаг по t
M = int(X / h) + 1  # Количество узлов по x
N = int(T / tau) + 1  # Количество узлов по t

# --- ФУНКЦИЯ НАЧАЛЬНОГО УСЛОВИЯ (согласно формуле из задания) ---
def y_initial(x):
    """Вычисляет начальное условие u(x, 0)"""
    if x <= c:
        return a
    else:
        # Формула для прямой, проходящей через точки (d, c) и (1, b)
        return (c - b) / (d - 1) * x + (b * d - c) / (d - 1)

# --- СОЗДАНИЕ СЕТКИ ---
x_grid = np.linspace(0, X, M)
t_grid = np.linspace(0, T, N)

# --- МАССИВ ДЛЯ РЕШЕНИЯ ---
u = np.zeros((N, M))

# --- ЗАПОЛНЕНИЕ НАЧАЛЬНЫХ И ГРАНИЧНЫХ УСЛОВИЙ ---
# Начальное условие u(x, 0)
for i in range(M):
    u[0, i] = y_initial(x_grid[i])

# Граничные условия u(0, t) и u(1, t)
for j in range(1, N):
    u[j, 0] = a
    u[j, M - 1] = b

# --- РЕШЕНИЕ УРАВНЕНИЯ (ЯВНАЯ СХЕМА) ---
for j in range(N - 1):
    for i in range(1, M - 1):
        u[j + 1, i] = u[j, i] + tau / h**2 * (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1])

# --- ВЫВОД ТАБЛИЦЫ ЗНАЧЕНИЙ ---
print("Таблица u(x, t):")
print("-" * 50)
print(f"{'i':<3} {'t':<7}", end="")
for x in x_grid:
    print(f"{x:<8.1f}", end="")
print()
print("-" * 50)
for j in range(N):
    print(f"{j:<3} {t_grid[j]:<7.3f}", end="")
    for i in range(M):
        print(f"{u[j, i]:<8.4f}", end="")
    print()

# --- ВЫВОД ТАБЛИЦЫ С ПОГРЕШНОСТЬЮ ---
# В качестве "точного" решения используется решение на последнем временном слое
# для оценки изменения между шагами, как в вашем примере.
u_exact = u[N-1, :]
abs_error = np.abs(u[N-1, :] - u[N-2, :])

print("\nТаблица |u - u_exact| (изменение за последний шаг):")
print("-" * 50)
print(f"{'t':<5} {'x':<7}", end="")
for i in range(M):
    print(f"{x_grid[i]:<8.1f}", end="")
print()
print("-" * 50)
print(f"{t_grid[N-1]:<7.3f}", end="")
for i in range(M):
     print(f"{abs_error[i]:<8.4f}", end="")
print()

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
plt.figure(figsize=(10, 6))
# Выбираем несколько временных слоев для отображения
step = max(1, int(N / 6))
for j in range(0, N, step):
    plt.plot(x_grid, u[j, :], label=f't = {t_grid[j]:.3f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Решение уравнения теплопроводности (вариант 18)')
plt.legend()
plt.grid(True)
plt.show()