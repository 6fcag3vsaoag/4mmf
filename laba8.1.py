import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи для варианта 18
a = 15.0
b = 10.0
c = 27.00
d = 0.75
T = 0.025
X = 1.0

# Параметры сетки
h = 0.1  # Шаг по x
tau = 0.005  # Шаг по t
M = int(X / h) + 1  # Количество узлов по x
N = int(T / tau) + 1  # Количество узлов по t

# Функция начального условия для варианта 18
def y(x):
    # Согласно изображению, есть несколько формул:
    # Формула 1: y = (c - a)/d * x + a
    # Формула 2: y = (c - b)/(d - 1) * x + (b*d - c)/(d - 1)
    # В таблице указано: y = a
    
    # Используем формулу из таблицы 8.2 для варианта 18
    return a  # Постоянная функция

# Альтернативные варианты (раскомментировать при необходимости):
# def y(x):
#     return (c - a) / d * x + a  # Линейная функция 1
# 
# def y(x):
#     return (c - b) / (d - 1) * x + (b * d - c) / (d - 1)  # Линейная функция 2

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

# Анализ результатов
print(f"\nНачальное условие: y(x) = {a} (постоянная)")
print(f"Граничные условия: u(0,t) = {a}, u(1,t) = {b}")
print(f"Установившееся распределение: линейное от {a} до {b}")

# Построение графиков
plt.figure(figsize=(12, 8))

# График 1: Эволюция температуры
plt.subplot(2, 1, 1)
time_indices = [0, N//6, N//3, N//2, 2*N//3, N-1]
for j in time_indices:
    plt.plot(x_grid, u[j, :], marker='o', label=f't = {t_grid[j]:.3f}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Эволюция температуры стержня (Вариант 18) - y(x) = постоянная')
plt.legend()
plt.grid(True)

# График 2: Изменение в отдельных точках
plt.subplot(2, 1, 2)
x_points = [0, M//4, M//2, 3*M//4, M-1]
for i in x_points:
    plt.plot(t_grid, u[:, i], label=f'x = {x_grid[i]:.1f}')

plt.xlabel('Время t')
plt.ylabel('Температура u(x,t)')
plt.title('Температура в отдельных точках стержня')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Проверка сходимости к установившемуся состоянию
steady_state = a + (b - a) * x_grid  # Линейное распределение
final_error = np.max(np.abs(u[N-1, :] - steady_state))
print(f"\nМаксимальное отклонение от установившегося состояния: {final_error:.6f}")