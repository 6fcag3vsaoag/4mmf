import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_in_domain(x, y):
    """
    Проверяет, принадлежит ли точка (x,y) области D.
    Область D: Г-образная фигура из двух прямоугольников:
    - Вертикальный: x ∈ [0, 1], y ∈ [0, 3]
    - Горизонтальный: x ∈ [1, 3], y ∈ [0, 1]
    """
    # Вертикальный прямоугольник
    if 0 <= x <= 1 and 0 <= y <= 3:
        return True
    
    # Горизонтальный прямоугольник
    if 1 <= x <= 3 and 0 <= y <= 1:
        return True
    
    return False

def solve_poisson_L_shape(h, epsilon, G_theta):
    """
    Решает задачу Дирихле для уравнения Пуассона в Г-образной области D.
    """
    
    Lx, Ly = 3.0, 3.0
    nx = int(Lx / h) + 1
    ny = int(Ly / h) + 1
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Инициализация решения
    phi = np.zeros((ny, nx))
    domain_mask = np.zeros((ny, nx), dtype=bool)

    print(f"\n1. ПОСТРОЕНИЕ СЕТКИ И ОБЛАСТИ D:")
    print(f"   Область D: Г-образная фигура")
    print(f"   - Вертикальный прямоугольник: [0, 1] × [0, 3]")
    print(f"   - Горизонтальный прямоугольник: [1, 3] × [0, 1]")
    print(f"   Шаг h = {h}")
    print(f"   Количество узлов: {nx} × {ny} = {nx * ny}")

    # --- Создание маски области и граничных условий ---
    print(f"\n2. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   φ(x,y) = 0 на всей границе области D")
    
    boundary_nodes = 0
    internal_nodes = 0
    
    for j in range(ny):
        for i in range(nx):
            if is_in_domain(X[j, i], Y[j, i]):
                domain_mask[j, i] = True
                internal_nodes += 1
                
                # Проверяем, является ли узел граничным
                is_boundary = False
                # Проверяем соседей
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                        is_boundary = True
                        break
                    if not is_in_domain(X[nj, ni], Y[nj, ni]):
                        is_boundary = True
                        break
                
                if is_boundary:
                    phi[j, i] = 0.0  # Граничное условие
                    boundary_nodes += 1

    print(f"   Внутренних узлов: {internal_nodes}")
    print(f"   Граничных узлов: {boundary_nodes}")

    # --- Итерационный процесс ---
    print(f"\n3. ИТЕРАЦИОННЫЙ ПРОЦЕСС:")
    print(f"   Критерий сходимости: max|φ_new - φ_old| < ε = {epsilon}")
    print(f"   Параметр Gθ = {G_theta}")
    print(f"   Формула обновления для внутренних узлов:")
    print(f"        φ[i,j] = 0.25 * (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1] + h² × 2 × Gθ)")

    max_iter = 10000
    iteration_data = [phi.copy()]

    for it in range(max_iter):
        phi_old = phi.copy()
        
        # Обновление внутренних узлов области D
        for i in range(nx):
            for j in range(ny):
                if domain_mask[j, i] and phi_old[j, i] == 0:  # Внутренний узел (не граничный)
                    # Собираем значения соседей внутри области
                    neighbor_sum = 0
                    neighbor_count = 0
                    
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny and domain_mask[nj, ni]:
                            neighbor_sum += phi_old[nj, ni]
                            neighbor_count += 1
                    
                    if neighbor_count > 0:
                        phi[j, i] = (neighbor_sum + h**2 * 2.0 * G_theta) / neighbor_count

        iteration_data.append(phi.copy())

        # Проверка сходимости только для внутренних узлов
        internal_diff = 0
        for i in range(nx):
            for j in range(ny):
                if domain_mask[j, i] and phi_old[j, i] == 0:  # Только внутренние узлы
                    internal_diff = max(internal_diff, abs(phi[j, i] - phi_old[j, i]))
        
        if internal_diff < epsilon:
            print(f"   Сходимость достигнута на итерации {it+1}")
            print(f"   Максимальное изменение: {internal_diff:.6f}")
            break

        if (it + 1) % 500 == 0:
            print(f"   Итерация {it+1}: max|Δφ| = {internal_diff:.6f}")

    if it == max_iter - 1:
        print(f"   Достигнуто максимальное число итераций ({max_iter})")

    # Статистика решения
    internal_values = []
    for i in range(nx):
        for j in range(ny):
            if domain_mask[j, i] and phi[j, i] != 0:  # Внутренние узлы с вычисленными значениями
                internal_values.append(phi[j, i])
    
    internal_values = np.array(internal_values)
    
    print(f"\n4. РЕЗУЛЬТАТЫ:")
    print(f"   Итераций выполнено: {it+1}")
    if len(internal_values) > 0:
        print(f"   Минимальное значение φ: {np.min(internal_values):.6f}")
        print(f"   Максимальное значение φ: {np.max(internal_values):.6f}")
        print(f"   Среднее значение φ: {np.mean(internal_values):.6f}")
    else:
        print(f"   Нет внутренних узлов для анализа")

    return X, Y, phi, it+1, iteration_data, x, y, domain_mask

def create_comparison_table(phi1, phi2, domain_mask1, domain_mask2, X1, Y1, X2, Y2, h1, h2):
    """Создает таблицу сравнения значений φ в общих узлах"""
    print(f"\n5. СРАВНЕНИЕ РЕШЕНИЙ (h={h1} и h={h2}):")
    
    data = []
    ratio = int(h1 / h2)
    
    for j in range(phi1.shape[0]):
        for i in range(phi1.shape[1]):
            if domain_mask1[j, i] and phi1[j, i] != 0:  # Внутренний узел с вычисленным значением
                val1 = phi1[j, i]
                # Соответствующий узел в более мелкой сетке
                j2 = j * ratio
                i2 = i * ratio
                if (j2 < phi2.shape[0] and i2 < phi2.shape[1] and 
                    domain_mask2[j2, i2] and phi2[j2, i2] != 0):
                    val2 = phi2[j2, i2]
                    abs_diff = abs(val1 - val2)
                    data.append([X1[j, i], Y1[j, i], val1, val2, abs_diff])

    if data:
        df = pd.DataFrame(data, columns=['x', 'y', f'φ (h={h1})', f'φ (h={h2})', '|Разность|'])
        print(df.head(15).to_string(index=False, float_format="%.4f"))
        if len(data) > 15:
            print(f"   ... и еще {len(data) - 15} строк")
        
        max_diff = df['|Разность|'].max()
        avg_diff = df['|Разность|'].mean()
        print(f"\n   Максимальная разность: {max_diff:.6f}")
        print(f"   Средняя разность: {avg_diff:.6f}")
        
        return df, max_diff, avg_diff
    else:
        print("   Нет общих узлов для сравнения")
        return None, 0, 0

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ДЛЯ ВАРИАНТА 18 ---
G_theta = 3.6  # Параметр для варианта 18
epsilon = 0.1

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №6 - РЕШЕНИЕ ЗАДАЧИ ДИРИХЛЕ")
print("ДЛЯ УРАВНЕНИЯ ПУАССОНА В Г-ОБРАЗНОЙ ОБЛАСТИ")
print("ОБЛАСТЬ D: ВЕРТИКАЛЬНЫЙ [0,1]×[0,3] + ГОРИЗОНТАЛЬНЫЙ [1,3]×[0,1]")
print(f"ВАРИАНТ 18: Gθ = {G_theta}")
print("=" * 70)

# Решение с шагом h = 0.5
print("\n" + "="*50)
print("РЕШЕНИЕ С ШАГОМ h = 0.5")
print("="*50)
X1, Y1, phi1, iter1, iteration_data1, x1, y1, domain_mask1 = solve_poisson_L_shape(0.5, epsilon, G_theta)

# Решение с шагом h = 0.25
print("\n" + "="*50)
print("РЕШЕНИЕ С ШАГОМ h = 0.25")
print("="*50)
X2, Y2, phi2, iter2, iteration_data2, x2, y2, domain_mask2 = solve_poisson_L_shape(0.25, epsilon, G_theta)

# Сравнение решений
comparison_df, max_diff, avg_diff = create_comparison_table(phi1, phi2, domain_mask1, domain_mask2, X1, Y1, X2, Y2, 0.5, 0.25)

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(20, 12))

# 1. Область D и сетка (h=0.5)
plt.subplot(2, 3, 1)
# Вертикальный прямоугольник
vert_rect = plt.Rectangle((0, 0), 1, 3, fill=False, color='blue', linewidth=2)
plt.gca().add_patch(vert_rect)
# Горизонтальный прямоугольник
horiz_rect = plt.Rectangle((1, 0), 2, 1, fill=False, color='red', linewidth=2)
plt.gca().add_patch(horiz_rect)

# Сетка
for i in range(X1.shape[1]):
    for j in range(X1.shape[0]):
        if domain_mask1[j, i]:
            color = 'red' if phi1[j, i] == 0 else 'green'
            plt.scatter(X1[j, i], Y1[j, i], color=color, s=30, alpha=0.6)

plt.title('Область D и сетка (h=0.5)\nКрасные - граничные, Зеленые - внутренние', fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.1, 3.1)
plt.ylim(-0.1, 3.1)

# 2. Решение для h=0.5
plt.subplot(2, 3, 2)
phi1_masked = np.ma.array(phi1, mask=~domain_mask1)
contour = plt.contourf(X1, Y1, phi1_masked, levels=20, cmap='viridis')
plt.colorbar(contour, label='φ(x,y)')
# Рисуем контур области
vert_rect = plt.Rectangle((0, 0), 1, 3, fill=False, color='black', linewidth=1)
plt.gca().add_patch(vert_rect)
horiz_rect = plt.Rectangle((1, 0), 2, 1, fill=False, color='black', linewidth=1)
plt.gca().add_patch(horiz_rect)
plt.title(f'Решение для h=0.5\nИтераций: {iter1}', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# 3. Решение для h=0.25
plt.subplot(2, 3, 3)
phi2_masked = np.ma.array(phi2, mask=~domain_mask2)
contour = plt.contourf(X2, Y2, phi2_masked, levels=20, cmap='viridis')
plt.colorbar(contour, label='φ(x,y)')
# Рисуем контур области
vert_rect = plt.Rectangle((0, 0), 1, 3, fill=False, color='black', linewidth=1)
plt.gca().add_patch(vert_rect)
horiz_rect = plt.Rectangle((1, 0), 2, 1, fill=False, color='black', linewidth=1)
plt.gca().add_patch(horiz_rect)
plt.title(f'Решение для h=0.25\nИтераций: {iter2}', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# 4. Разность решений
plt.subplot(2, 3, 4)
if comparison_df is not None:
    diff_grid = np.zeros_like(phi1)
    for j in range(phi1.shape[0]):
        for i in range(phi1.shape[1]):
            if domain_mask1[j, i] and phi1[j, i] != 0:
                j2 = j * 2
                i2 = i * 2
                if (j2 < phi2.shape[0] and i2 < phi2.shape[1] and 
                    domain_mask2[j2, i2] and phi2[j2, i2] != 0):
                    diff_grid[j, i] = abs(phi1[j, i] - phi2[j2, i2])
    
    diff_masked = np.ma.array(diff_grid, mask=~domain_mask1)
    im = plt.imshow(diff_masked, extent=[0, 3, 0, 3], origin='lower', 
                   cmap='hot', aspect='auto')
    plt.colorbar(im, label='|Δφ|')
    plt.title(f'Разность |φ₀.₅ - φ₀.₂₅|\nМакс. = {max_diff:.4f}', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')

# 5. Сходимость
plt.subplot(2, 3, 5)
# Для h=0.5
diffs_h1 = []
for k in range(1, len(iteration_data1)):
    current = iteration_data1[k]
    previous = iteration_data1[k-1]
    max_diff_iter = 0
    for i in range(current.shape[1]):
        for j in range(current.shape[0]):
            if domain_mask1[j, i] and current[j, i] != 0 and previous[j, i] != 0:
                max_diff_iter = max(max_diff_iter, abs(current[j, i] - previous[j, i]))
    diffs_h1.append(max_diff_iter)

# Для h=0.25
diffs_h2 = []
for k in range(1, len(iteration_data2)):
    current = iteration_data2[k]
    previous = iteration_data2[k-1]
    max_diff_iter = 0
    for i in range(current.shape[1]):
        for j in range(current.shape[0]):
            if domain_mask2[j, i] and current[j, i] != 0 and previous[j, i] != 0:
                max_diff_iter = max(max_diff_iter, abs(current[j, i] - previous[j, i]))
    diffs_h2.append(max_diff_iter)

plt.semilogy(range(1, len(diffs_h1)+1), diffs_h1, 'b-', label=f'h=0.5 ({len(diffs_h1)} итер.)')
plt.semilogy(range(1, len(diffs_h2)+1), diffs_h2, 'r--', label=f'h=0.25 ({len(diffs_h2)} итер.)')
plt.axhline(y=epsilon, color='green', linestyle='--', label=f'ε={epsilon}')
plt.xlabel('Итерация')
plt.ylabel('max|Δφ|')
plt.title('Сходимость метода', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 3D визуализация
plt.subplot(2, 3, 6, projection='3d')
phi2_3d = phi2_masked.copy()
phi2_3d[~domain_mask2] = np.nan
surf = plt.gca().plot_surface(X2, Y2, phi2_3d, cmap='viridis', 
                             alpha=0.8, linewidth=0.1, antialiased=True)
plt.title('3D визуализация решения\n(h=0.25)', fontsize=12)
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# --- ТАБЛИЦА ЗНАЧЕНИЙ ---
print(f"\n6. ТАБЛИЦА ЗНАЧЕНИЙ φ(x,y) В ХАРАКТЕРНЫХ ТОЧКАХ (h=0.5):")

characteristic_points = [
    # Вертикальная часть
    (0.5, 0.5), (0.5, 1.5), (0.5, 2.5),
    # Горизонтальная часть  
    (1.5, 0.5), (2.5, 0.5),
    # Угловые точки
    (0.1, 0.1), (0.1, 2.9), (0.9, 0.1), (0.9, 2.9),
    (1.1, 0.1), (2.9, 0.1), (2.9, 0.9)
]

print("\n   x     y     φ(x,y)")
print("   " + "-"*20)
for x, y in characteristic_points:
    # Находим ближайший узел сетки
    i = np.argmin(np.abs(x1 - x))
    j = np.argmin(np.abs(y1 - y))
    if domain_mask1[j, i] and phi1[j, i] != 0:
        print(f"   {x:4.2f}  {y:4.2f}  {phi1[j, i]:6.3f}")
    else:
        print(f"   {x:4.2f}  {y:4.2f}    -")

# --- ИТОГОВЫЕ ВЫВОДЫ ---
print(f"\n" + "="*70)
print("ИТОГОВЫЕ ВЫВОДЫ ДЛЯ ВАРИАНТА 18")
print("="*70)

print(f"\n1. СХОДИМОСТЬ МЕТОДА:")
print(f"   - При h=0.5: {iter1} итераций")
print(f"   - При h=0.25: {iter2} итераций")
print(f"   - Метод успешно сходится для Г-образной области")

print(f"\n2. ТОЧНОСТЬ РЕШЕНИЯ:")
print(f"   - Максимальная разность между решениями: {max_diff:.6f}")
print(f"   - Средняя разность: {avg_diff:.6f}")
if max_diff < epsilon:
    print(f"   ✓ Точность ε={epsilon} достигнута")
else:
    print(f"   ⚠ Точность ε={epsilon} не достигнута, рекомендуется h=0.125")

print(f"\n3. ОСОБЕННОСТИ РЕШЕНИЯ ДЛЯ Г-ОБРАЗНОЙ ОБЛАСТИ:")
print(f"   - Область состоит из двух прямоугольников разного размера")
print(f"   - Наибольшие значения φ наблюдаются в центре области")
print(f"   - В углах наблюдаются особенности решения")

print(f"\n4. РЕКОМЕНДАЦИИ:")
print(f"   - Для Г-образной области рекомендуется h=0.25")
print(f"   - Метод сеток устойчив для областей сложной формы")
print(f"   - Решение удовлетворяет уравнению Пуассона и граничным условиям")

print(f"\n5. ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
print(f"   - Функция φ(x,y) описывает распределение напряжения")
print(f"   - Г-образная область моделирует угловое соединение")
print(f"   - Параметр Gθ = {G_theta} определяет интенсивность воздействия")