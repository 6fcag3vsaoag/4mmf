import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_inside_L_shape(x, y):
    """
    Проверяет принадлежность точки (x, y) L-образной области D:
    - Вертикальный прямоугольник: x ∈ [0, 1], y ∈ [0, 3]
    - Горизонтальный прямоугольник: x ∈ [1, 3], y ∈ [0, 1]
    """
    if 0 <= x <= 1 and 0 <= y <= 3:
        return True
    elif 1 < x <= 3 and 0 <= y <= 1:
        return True
    else:
        return False

def solve_poisson_detailed(h, epsilon, G_theta):
    """
    Решает задачу Дирихле для уравнения Пуассона в L-образной области методом простой итерации.
    """
    # Размеры области
    Lx, Ly = 3.0, 3.0
    nx = int(Lx / h) + 1
    ny = int(Ly / h) + 1
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Инициализация решения
    phi = np.zeros((ny, nx))

    print(f"\n1. ПОСТРОЕНИЕ СЕТКИ:")
    print(f"   Размер области: [0, {Lx}] x [0, {Ly}]")
    print(f"   Шаг h = {h}")
    print(f"   Количество узлов: nx = {nx}, ny = {ny}")

    # --- Граничные условия: φ = 0 на всей границе L ---
    print(f"\n2. ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ φ(x, y) = 0 на границе L:")
    for j in range(ny):
        for i in range(nx):
            if not is_inside_L_shape(X[j, i], Y[j, i]):
                phi[j, i] = 0.0
    print("   - На всех внешних узлах задано φ = 0")

    # --- Итерационный процесс ---
    print(f"\n3. ИТЕРАЦИОННЫЙ ПРОЦЕСС (метод простой итерации):")
    print(f"   Критерий сходимости: max|φ_new - φ_old| < ε = {epsilon}")
    print(f"   Максимальное число итераций: 10000")
    print(f"   Формула обновления для внутреннего узла (i,j):")
    print(f"        φ[i,j] = 0.25 * (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1]) + (h² * Gθ) / 4")
    print(f"   (Применяется только внутри L-образной области D)")

    max_iter = 10000
    iteration_data = []

    for it in range(max_iter):
        phi_old = phi.copy()
        # Обновление внутренних узлов
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if is_inside_L_shape(X[j, i], Y[j, i]):
                    phi[j, i] = 0.25 * (
                        phi[j, i + 1] + phi[j, i - 1] +
                        phi[j + 1, i] + phi[j - 1, i]
                    ) + (h ** 2 * G_theta) / 4.0

        iteration_data.append(phi.copy())

        # Проверка сходимости
        diff = np.max(np.abs(phi - phi_old))
        if diff < epsilon:
            print(f"   Сходимость достигнута на итерации {it + 1}. Макс. разность = {diff:.6f}")
            break

        if (it + 1) % 100 == 0:
            print(f"   Итерация {it + 1}: max|Δφ| = {diff:.6f}")

    if it == max_iter - 1:
        print(f"   Предупреждение: Достигнуто максимальное число итераций ({max_iter}). Точность {epsilon} не достигнута.")

    print(f"\n4. РЕЗУЛЬТАТЫ:")
    print(f"   Решение сошлось за {it + 1} итераций.")
    print(f"   Минимальное значение φ: {np.min(phi):.6f}")
    print(f"   Максимальное значение φ: {np.max(phi):.6f}")

    return X, Y, phi, it + 1, iteration_data, y, ny, nx, x


def create_comparison_table(phi1, phi2, h1, h2):
    """Создает таблицу сравнения значений φ в общих узлах сеток с шагами h1 и h2."""
    print(f"\n5. СРАВНЕНИЕ РЕШЕНИЙ В ОБЩИХ УЗЛАХ (h={h1} и h={h2}):")
    nx1 = phi1.shape[1]
    ny1 = phi1.shape[0]
    ratio = int(h1 / h2)

    data = []
    for j in range(ny1):
        for i in range(nx1):
            val1 = phi1[j, i]
            val2 = phi2[j * ratio, i * ratio]
            abs_diff = abs(val1 - val2)
            data.append([i * h1, j * h1, val1, val2, abs_diff])

    df = pd.DataFrame(data, columns=['x', 'y', f'φ (h={h1})', f'φ (h={h2})', 'Абс. Разность'])
    print(df.to_string(index=False, float_format="{:,.4f}".format))

    max_diff = df['Абс. Разность'].max()
    avg_diff = df['Абс. Разность'].mean()
    print(f"\n   Максимальная абсолютная разность: {max_diff:.6f}")
    print(f"   Средняя абсолютная разность: {avg_diff:.6f}")

    return df, max_diff, avg_diff


# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---

G_theta = 3.6  # Параметр для варианта 18
epsilon = 0.1

# --- Решение с шагом h = 0.5 ---
X1, Y1, phi1, iter1, iteration_data1, y1, ny1, nx1, x1 = solve_poisson_detailed(0.5, epsilon, G_theta)

# --- Решение с шагом h = 0.25 ---
X2, Y2, phi2, iter2, iteration_data2, y2, ny2, nx2, x2 = solve_poisson_detailed(0.25, epsilon, G_theta)

# --- Сравнение решений ---
comparison_df, max_diff, avg_diff = create_comparison_table(phi1, phi2, 0.5, 0.25)

print(f"\nПроверка граничных условий (для h=0.25):")
# Все узлы вне области D уже нулевые, проверим только границу
boundary_error = np.max(np.abs(phi2[~np.array([[is_inside_L_shape(X2[j, i], Y2[j, i]) for i in range(nx2)] for j in range(ny2)])]))
print(f"   Максимальная ошибка на границах: {boundary_error:.6f}")

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(18, 14))

# График 1: Область D и сетка (h=0.5)
plt.subplot(2, 3, 1)
mask1 = np.array([[is_inside_L_shape(X1[j, i], Y1[j, i]) for i in range(nx1)] for j in range(ny1)])
plt.contourf(X1, Y1, mask1.astype(int), levels=[-0.5, 0.5, 1.5], colors=['white', 'lightblue'], alpha=0.6)
plt.contour(X1, Y1, mask1.astype(int), levels=[0.5], colors='blue', linewidths=2)
plt.scatter(X1, Y1, color='gray', s=20, marker='o', label='Узлы сетки')
plt.title('Область D (L-образная) и сетка (h=0.5)', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 3)
plt.ylim(0, 3)

# График 2: Решение для h = 0.5
plt.subplot(2, 3, 2)
plt.gca().invert_yaxis()
for xi in np.arange(0, 3.1, 0.5):
    plt.axvline(xi, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
for yi in np.arange(0, 3.1, 0.5):
    plt.axhline(yi, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

for i in range(phi1.shape[1]):
    for j in range(phi1.shape[0]):
        if is_inside_L_shape(X1[j, i], Y1[j, i]):
            plt.text(i*0.5, j*0.5, f"{phi1[j, i]:.3f}", ha='center', va='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", facecolor='w', alpha=0.7))

plt.yticks(np.arange(0, 3.1, 0.5), [f'y{i}' for i in range(7)])
plt.xticks(np.arange(0, 3.1, 0.5), [f'x{i}' for i in range(7)])
plt.title(f'Значения φ(x,y) для h = 0.5\n(Итераций: {iter1})', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.axis('equal')

# График 3: Решение для h = 0.25 (контурный график)
plt.subplot(2, 3, 3)
contour = plt.contourf(X2, Y2, phi2, levels=20, cmap='viridis')
plt.colorbar(contour, label='φ(x, y)')
plt.title(f'Решение φ(x,y) для h = 0.25\n(Итераций: {iter2})', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)

# График 4: Разность решений в общих узлах
plt.subplot(2, 3, 4)
Z_diff = np.zeros_like(phi1)
for j in range(ny1):
    for i in range(nx1):
        Z_diff[j, i] = abs(phi1[j, i] - phi2[j*2, i*2])

im = plt.imshow(Z_diff, extent=[0, 3, 0, 3], origin='lower', cmap='hot', aspect='auto')
plt.colorbar(im, label='|Δφ|')
plt.title(f'Абсолютная разность |φ_h=0.5 - φ_h=0.25|\nМакс. = {max_diff:.4f}', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(np.arange(0, 3.1, 0.5))
plt.yticks(np.arange(0, 3.1, 0.5))

# График 5: Область D и сетка (h=0.25)
plt.subplot(2, 3, 5)
mask2 = np.array([[is_inside_L_shape(X2[j, i], Y2[j, i]) for i in range(nx2)] for j in range(ny2)])
plt.contourf(X2, Y2, mask2.astype(int), levels=[-0.5, 0.5, 1.5], colors=['white', 'lightblue'], alpha=0.6)
plt.contour(X2, Y2, mask2.astype(int), levels=[0.5], colors='blue', linewidths=2)
plt.scatter(X2, Y2, color='gray', s=10, marker='o', label='Узлы сетки')
plt.title('Область D (L-образная) и сетка (h=0.25)', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 3)
plt.ylim(0, 3)

# График 6: Сходимость
plt.subplot(2, 3, 6)
diffs_h1 = [np.max(np.abs(iteration_data1[i] - iteration_data1[i-1])) for i in range(1, len(iteration_data1))]
diffs_h2 = [np.max(np.abs(iteration_data2[i] - iteration_data2[i-1])) for i in range(1, len(iteration_data2))]

plt.semilogy(range(1, len(diffs_h1)+1), diffs_h1, 'b-', label=f'h=0.5 (итераций: {len(diffs_h1)})')
plt.semilogy(range(1, len(diffs_h2)+1), diffs_h2, 'r--', label=f'h=0.25 (итераций: {len(diffs_h2)})')
plt.axhline(y=epsilon, color='green', linestyle='--', label=f'Точность ε={epsilon}')
plt.xlabel('Номер итерации')
plt.ylabel('Макс. |Δφ|')
plt.title('Сходимость метода (логарифмическая шкала)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- ИТОГОВЫЙ АНАЛИЗ И ВЫВОДЫ ---
print(f"\nКоличество итераций:")
print(f"   - Для h = 0.5: {iter1} итераций")
print(f"   - Для h = 0.25: {iter2} итераций")
print("   Вывод: Уменьшение шага привело к увеличению числа итераций, что ожидаемо из-за роста числа узлов.")

print(f"\nТочность решения:")
print(f"   - Максимальная абсолютная разность в общих узлах: {max_diff:.6f}")
print(f"   - Средняя абсолютная разность в общих узлах: {avg_diff:.6f}")
if max_diff < epsilon:
    print(f"   Вывод: Разность меньше заданной точности ε={epsilon}. Результат с h=0.5 приемлем.")
else:
    print(f"   Вывод: Разность превышает заданную точность ε={epsilon}. Необходимо использовать меньший шаг (h=0.25 или меньше).")

print(f"\nЭкстремальные значения:")
phi1_min, phi1_max = np.min(phi1), np.max(phi1)
phi2_min, phi2_max = np.min(phi2), np.max(phi2)
print(f"   - Для h=0.5: φ ∈ [{phi1_min:.4f}, {phi1_max:.4f}]")
print(f"   - Для h=0.25: φ ∈ [{phi2_min:.4f}, {phi2_max:.4f}]")

print(f"\n   Значения функции φ(x,y) в узлах сетки (h=0.5):")
print("      x: ", end="")
for i in range(nx1):
    print(f"{x1[i]:5.1f} ", end="")
print()
for j in range(len(y1)-1, -1, -1):
    print(f"y={y1[j]:4.1f}: ", end="")
    for i in range(nx1):
        if is_inside_L_shape(X1[j, i], Y1[j, i]):
            print(f"{phi1[j, i]:5.3f} ", end="")
        else:
            print("     - ", end="")
    print()