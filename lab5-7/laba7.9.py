import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.tri import Triangulation

def solve_torsion_L_shape_by_fem(G_theta, print_matrices=True):
    """
    Решает задачу кручения стержня L-образного сечения методом конечных элементов.
    Область: [0,1]×[0,3]∪[1,3]×[0,1]
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СЕЧЕНИЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("Область: [0,1]×[0,3]∪[1,3]×[0,1]")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов сетки ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")
    
    # Создаем узлы строго внутри L-образной области
    all_nodes = []
    
    # Вертикальная часть: [0,1]×[0,3]
    nx_vert, ny_vert = 8, 16  # Увеличиваем разрешение
    for i in range(nx_vert + 1):
        x = i / nx_vert  # x от 0 до 1
        for j in range(ny_vert + 1):
            y = 3 * j / ny_vert  # y от 0 до 3
            all_nodes.append([x, y])
    
    n_vertical = len(all_nodes)
    print(f"   Узлов в вертикальной части: {n_vertical}")
    
    # Горизонтальная часть: [1,3]×[0,1]
    nx_horiz, ny_horiz = 12, 6  # Увеличиваем разрешение
    for i in range(1, nx_horiz + 1):
        x = 1 + 2 * i / nx_horiz  # x от 1+ до 3
        for j in range(ny_horiz + 1):
            y = j / ny_horiz  # y от 0 до 1
            all_nodes.append([x, y])
    
    n_horizontal = len(all_nodes) - n_vertical
    print(f"   Узлов в горизонтальной части: {n_horizontal}")
    
    # Внутренние узлы для лучшей триангуляции
    interior_nodes = [
        [0.3, 1.5], [0.7, 2.0], [0.5, 0.5],
        [1.5, 0.3], [2.0, 0.7], [2.5, 0.5],
        [0.2, 2.5], [0.8, 0.8], [1.2, 0.2], [2.2, 0.8]  # Добавляем больше узлов
    ]
    all_nodes.extend(interior_nodes)
    
    nodes = np.array(all_nodes)
    n_nodes = len(nodes)
    print(f"   Всего узлов: {n_nodes}")

    # --- Шаг 2: Правильная триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    
    def is_point_in_L_shape(x, y, tol=1e-10):
        """Проверяет, находится ли точка внутри L-образной области"""
        vertical = (0 - tol <= x <= 1 + tol) and (0 - tol <= y <= 3 + tol)
        horizontal = (1 - tol <= x <= 3 + tol) and (0 - tol <= y <= 1 + tol)
        return vertical or horizontal
    
    def is_triangle_valid(p1, p2, p3, tol=1e-6):
        """Проверяет, что треугольник в основном внутри области и не вырожден"""
        # Проверяем площадь (избегаем вырожденных треугольников)
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < tol:
            return False
        
        # Проверяем, что все вершины внутри или рядом с границей
        points_inside = 0
        for p in [p1, p2, p3]:
            if is_point_in_L_shape(p[0], p[1]):
                points_inside += 1
        
        # Принимаем треугольники, у которых хотя бы 2 вершины внутри
        return points_inside >= 2
    
    # Создаем триангуляцию Делоне
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        all_elements = tri.simplices.tolist()
        print("   Использована триангуляция Делоне")
    except:
        print("   Ошибка триангуляции Делоне, используется простая триангуляция")
        all_elements = []
    
    # Фильтруем элементы
    elements = []
    for elem in all_elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        if is_triangle_valid(p1, p2, p3):
            elements.append(elem)
    
    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    # --- Шаг 3: Вычисление площадей элементов ---
    def triangle_area(p1, p2, p3):
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    areas = []
    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        area = triangle_area(p1, p2, p3)
        areas.append(area)

    print(f"\n   Площади элементов (первые 5):")
    for i in range(min(5, len(areas))):
        print(f"   S{i+1} = {areas[i]:.4f}")

    # --- Шаг 4: Построение матриц жесткости ---
    def element_stiffness_matrix(elem_nodes, area):
        xi, yi = elem_nodes[0]
        xj, yj = elem_nodes[1]
        xk, yk = elem_nodes[2]

        bi = yj - yk
        bj = yk - yi
        bk = yi - yj

        ci = xk - xj
        cj = xi - xk
        ck = xj - xi

        B = np.array([[bi, bj, bk], [ci, cj, ck]]) / (2 * area)
        K_e = area * (B.T @ B)
        return K_e

    K_global = np.zeros((n_nodes, n_nodes))

    print(f"\n3. ПОСТРОЕНИЕ МАТРИЦ ЖЕСТКОСТИ:")
    
    for idx, elem in enumerate(elements):
        elem_nodes = nodes[elem]
        K_e = element_stiffness_matrix(elem_nodes, areas[idx])
        for i_local, i_global in enumerate(elem):
            for j_local, j_global in enumerate(elem):
                K_global[i_global, j_global] += K_e[i_local, j_local]

    # --- Шаг 5: Построение вектора нагрузки ---
    F_global = np.zeros(n_nodes)

    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")
    
    for idx, elem in enumerate(elements):
        f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
        for i_local, i_global in enumerate(elem):
            F_global[i_global] += f_e[i_local]

    # --- Шаг 6: Граничные условия ---
    boundary_nodes = []
    tol = 1e-8
    
    # Более точное определение граничных узлов
    for i, (x, y) in enumerate(nodes):
        on_boundary = False
        
        # Проверяем близость к границам с учетом области
        if abs(x) < tol and 0 <= y <= 3:  # левая граница
            on_boundary = True
        elif abs(x - 1) < tol and 1 <= y <= 3:  # внутренняя вертикаль
            on_boundary = True
        elif abs(y - 3) < tol and 0 <= x <= 1:  # верхняя граница
            on_boundary = True
        elif abs(y - 1) < tol and 1 <= x <= 3:  # верхняя горизонталь
            on_boundary = True
        elif abs(x - 3) < tol and 0 <= y <= 1:  # правая граница
            on_boundary = True
        elif abs(y) < tol and 0 <= x <= 3:  # нижняя граница
            on_boundary = True
        
        if on_boundary:
            boundary_nodes.append(i)

    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes)}")

    K_original = K_global.copy()
    F_original = F_global.copy()

    # Применяем граничные условия
    for node_idx in boundary_nodes:
        K_global[node_idx, :] = 0.0
        K_global[:, node_idx] = 0.0
        K_global[node_idx, node_idx] = 1.0
        F_global[node_idx] = 0.0

    # --- Шаг 7: Решение системы ---
    print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
    
    try:
        # Добавляем регуляризацию для устойчивости
        K_reg = K_global + 1e-12 * np.eye(n_nodes)
        Phi = np.linalg.solve(K_reg, F_global)
        print(f"   Система успешно решена")
                
    except np.linalg.LinAlgError as e:
        print(f"   Ошибка: {e}")
        try:
            Phi = np.linalg.lstsq(K_global, F_global, rcond=None)[0]
            print("   Использован метод наименьших квадратов")
        except:
            return None, None, None, None, None, None, None, None

    # --- Шаг 8: Вычисление крутящего момента ---
    integral_phi = 0.0
    for idx, elem in enumerate(elements):
        phi_vals = [Phi[elem[0]], Phi[elem[1]], Phi[elem[2]]]
        avg_phi = np.mean(phi_vals)
        integral_phi += areas[idx] * avg_phi

    T = 2 * G_theta * integral_phi

    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = 2 * Gθ * ∫∫φ dxdy = {T:.6f}")

    # Проверка решения
    min_phi = np.min(Phi)
    max_phi = np.max(Phi)
    print(f"   Минимальное φ: {min_phi:.6f}")
    print(f"   Максимальное φ: {max_phi:.6f}")
    
    if min_phi < -1e-10:
        print("   ПРЕДУПРЕЖДЕНИЕ: Отрицательные значения φ!")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---
G_theta = 3.6

print("Запуск улучшенного расчета...")
result = solve_torsion_L_shape_by_fem(G_theta, print_matrices=False)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # Визуализация
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.6, alpha=0.7)
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=15, zorder=5)
    
    L_boundary = np.array([[0,0], [0,3], [1,3], [1,1], [3,1], [3,0], [1,0], [0,0]])
    ax1.plot(L_boundary[:, 0], L_boundary[:, 1], 'k-', linewidth=2, label='Граница')
    ax1.set_title('Сетка конечных элементов', fontsize=14)
    ax1.axis('equal')
    ax1.legend()

    # График 2: Распределение φ
    try:
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
        # Используем симметричную цветовую шкалу относительно максимума
        vmax = np.max(Phi) * 1.1
        contour = ax2.tricontourf(tri, Phi, levels=50, cmap='viridis', vmin=0, vmax=vmax)
        ax2.tricontour(tri, Phi, levels=10, colors='white', linewidths=0.3, alpha=0.7)
        ax2.plot(L_boundary[:, 0], L_boundary[:, 1], 'k-', linewidth=2)
        plt.colorbar(contour, ax=ax2, label='φ(x, y)')
        ax2.set_title('Распределение функции напряжения φ', fontsize=14)
        ax2.axis('equal')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Ошибка: {e}', transform=ax2.transAxes, ha='center')

    # Графики 3-4: Матрицы жесткости
    im3 = ax3.imshow(np.log10(np.abs(K_original) + 1e-12), cmap='hot', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='log10(|K|)')
    ax3.set_title('Матрица жесткости (до граничных условий)', fontsize=12)

    im4 = ax4.imshow(np.log10(np.abs(K_bc) + 1e-12), cmap='hot', aspect='auto')
    plt.colorbar(im4, ax=ax4, label='log10(|K|)')
    ax4.set_title('Матрица жесткости (после граничных условий)', fontsize=12)

    plt.tight_layout()
    plt.show()

    print(f"\nИТОГ: T = {T:.6f}")