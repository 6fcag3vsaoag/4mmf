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
    nx_vert, ny_vert = 6, 12
    for i in range(nx_vert + 1):
        x = i / nx_vert  # x от 0 до 1
        for j in range(ny_vert + 1):
            y = 3 * j / ny_vert  # y от 0 до 3
            all_nodes.append([x, y])
    
    n_vertical = len(all_nodes)
    print(f"   Узлов в вертикальной части: {n_vertical}")
    
    # Горизонтальная часть: [1,3]×[0,1] (исключаем пересечение с вертикальной)
    nx_horiz, ny_horiz = 8, 4
    for i in range(1, nx_horiz + 1):  # начинаем с x > 1
        x = 1 + 2 * i / nx_horiz  # x от 1+ до 3
        for j in range(ny_horiz + 1):
            y = j / ny_horiz  # y от 0 до 1
            # Исключаем точки, которые уже есть в вертикальной части
            if not (abs(x - 1) < 0.001 and abs(y) < 0.001):
                all_nodes.append([x, y])
    
    n_horizontal = len(all_nodes) - n_vertical
    print(f"   Узлов в горизонтальной части: {n_horizontal}")
    
    # Внутренние узлы для лучшей триангуляции
    interior_nodes = [
        [0.3, 1.5], [0.7, 2.0], [0.5, 0.5],
        [1.5, 0.3], [2.0, 0.7], [2.5, 0.5]
    ]
    all_nodes.extend(interior_nodes)
    
    nodes = np.array(all_nodes)
    n_nodes = len(nodes)
    print(f"   Всего узлов: {n_nodes}")

    # --- Шаг 2: Правильная триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    
    def is_in_L_shape(x, y, tol=1e-10):
        """Проверяет, находится ли точка внутри L-образной области"""
        vertical = (0 <= x <= 1) and (0 <= y <= 3)
        horizontal = (1 <= x <= 3) and (0 <= y <= 1)
        return vertical or horizontal
    
    def is_triangle_in_L_shape(p1, p2, p3, tol=1e-10):
        """Проверяет, что весь треугольник находится внутри L-образной области"""
        # Проверяем все три вершины
        if not (is_in_L_shape(p1[0], p1[1], tol) and 
                is_in_L_shape(p2[0], p2[1], tol) and 
                is_in_L_shape(p3[0], p3[1], tol)):
            return False
        
        # Проверяем центр тяжести
        centroid_x = (p1[0] + p2[0] + p3[0]) / 3
        centroid_y = (p1[1] + p2[1] + p3[1]) / 3
        return is_in_L_shape(centroid_x, centroid_y, tol)
    
    # Используем триангуляцию Делоне, но фильтруем элементы
    from scipy.spatial import Delaunay
    
    # Создаем триангуляцию Делоне для всех точек
    try:
        tri = Delaunay(nodes)
        elements = tri.simplices.tolist()
    except:
        # Если scipy не доступен, создаем простую триангуляцию
        print("   Используем упрощенную триангуляцию")
        elements = []
        
        # Триангуляция для вертикальной части
        for i in range(nx_vert):
            for j in range(ny_vert):
                idx1 = i * (ny_vert + 1) + j
                idx2 = i * (ny_vert + 1) + j + 1
                idx3 = (i + 1) * (ny_vert + 1) + j
                idx4 = (i + 1) * (ny_vert + 1) + j + 1
                
                p1, p2, p3 = nodes[idx1], nodes[idx2], nodes[idx3]
                p4 = nodes[idx4]
                
                # Первый треугольник
                if is_triangle_in_L_shape(p1, p2, p3):
                    elements.append([idx1, idx2, idx3])
                
                # Второй треугольник
                if is_triangle_in_L_shape(p2, p3, p4):
                    elements.append([idx2, idx3, idx4])
        
        # Триангуляция для горизонтальной части
        start_horiz = n_vertical
        for i in range(nx_horiz):
            for j in range(ny_horiz):
                idx1 = start_horiz + i * (ny_horiz + 1) + j
                idx2 = start_horiz + i * (ny_horiz + 1) + j + 1
                idx3 = start_horiz + (i + 1) * (ny_horiz + 1) + j
                idx4 = start_horiz + (i + 1) * (ny_horiz + 1) + j + 1
                
                # Проверяем индексы
                if (idx1 < n_nodes and idx2 < n_nodes and 
                    idx3 < n_nodes and idx4 < n_nodes):
                    
                    p1, p2, p3 = nodes[idx1], nodes[idx2], nodes[idx3]
                    p4 = nodes[idx4]
                    
                    # Первый треугольник
                    if is_triangle_in_L_shape(p1, p2, p3):
                        elements.append([idx1, idx2, idx3])
                    
                    # Второй треугольник
                    if is_triangle_in_L_shape(p2, p3, p4):
                        elements.append([idx2, idx3, idx4])
    
    # Фильтруем элементы - оставляем только те, что полностью внутри L-образной области
    filtered_elements = []
    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        if is_triangle_in_L_shape(p1, p2, p3):
            filtered_elements.append(elem)
    
    elements = filtered_elements
    n_elements = len(elements)
    print(f"   Создано элементов (после фильтрации): {n_elements}")

    # --- Шаг 3: Вычисление площадей элементов ---
    def triangle_area(p1, p2, p3):
        """Вычисляет площадь треугольника по координатам вершин."""
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    areas = []
    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        area = triangle_area(p1, p2, p3)
        areas.append(area)

    print(f"\n   Площади элементов (первые 5):")
    for i in range(min(5, len(areas))):
        print(f"   S{i+1} = {areas[i]:.3f}")
    if len(areas) > 5:
        print(f"   ... и еще {len(areas)-5} элементов")

    # --- Шаг 4: Построение матриц жесткости ---
    def element_stiffness_matrix(elem_nodes, area):
        """
        Строит локальную матрицу жесткости для треугольного элемента.
        """
        xi, yi = elem_nodes[0]
        xj, yj = elem_nodes[1]
        xk, yk = elem_nodes[2]

        # Вычисляем коэффициенты
        bi = yj - yk
        bj = yk - yi
        bk = yi - yj

        ci = xk - xj
        cj = xi - xk
        ck = xj - xi

        # Матрица градиентов B
        B = np.array([[bi, bj, bk], [ci, cj, ck]]) / (2 * area)

        # Матрица жесткости K_e = area * B^T * B
        K_e = area * (B.T @ B)
        return K_e

    # Глобальная матрица жесткости
    K_global = np.zeros((n_nodes, n_nodes))

    print(f"\n3. ПОСТРОЕНИЕ МАТРИЦ ЖЕСТКОСТИ:")
    
    # Сборка матрицы жесткости
    for idx, elem in enumerate(elements):
        elem_nodes = nodes[elem]
        K_e = element_stiffness_matrix(elem_nodes, areas[idx])
        
        # Сборка в глобальную матрицу
        for i_local, i_global in enumerate(elem):
            for j_local, j_global in enumerate(elem):
                K_global[i_global, j_global] += K_e[i_local, j_local]

    # --- Шаг 5: Построение вектора нагрузки ---
    F_global = np.zeros(n_nodes)

    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")
    
    # Сборка вектора нагрузки
    for idx, elem in enumerate(elements):
        f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
        for i_local, i_global in enumerate(elem):
            F_global[i_global] += f_e[i_local]

    # --- Шаг 6: Граничные условия ---
    boundary_nodes = []
    
    # Определяем граничные узлы (внешний контур L-образной области)
    for i, (x, y) in enumerate(nodes):
        on_boundary = False
        
        tol = 1e-10
        # Внешняя граница L-образной области
        if (abs(x) < tol and 0 <= y <= 3):  # левая граница
            on_boundary = True
        elif (abs(y - 3) < tol and 0 <= x <= 1):  # верхняя граница вертикальной части
            on_boundary = True
        elif (abs(x - 1) < tol and 1 <= y <= 3):  # правая граница вертикальной части
            on_boundary = True
        elif (abs(y - 1) < tol and 1 <= x <= 3):  # верхняя граница горизонтальной части
            on_boundary = True
        elif (abs(x - 3) < tol and 0 <= y <= 1):  # правая граница
            on_boundary = True
        elif (abs(y) < tol and 0 <= x <= 3):  # нижняя граница
            on_boundary = True
        
        if on_boundary:
            boundary_nodes.append(i)

    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes)}")

    # Сохраняем оригинальные матрицы
    K_original = K_global.copy()
    F_original = F_global.copy()

    # Применяем граничные условия (φ = 0 на границе)
    for node_idx in boundary_nodes:
        # Обнуляем строку и столбец, кроме диагонали
        for j in range(n_nodes):
            if j != node_idx:
                K_global[node_idx, j] = 0.0
                K_global[j, node_idx] = 0.0
        # Устанавливаем диагональный элемент в 1
        K_global[node_idx, node_idx] = 1.0
        # Обнуляем нагрузку на границе
        F_global[node_idx] = 0.0

    # --- Шаг 7: Решение системы ---
    print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
    
    try:
        # Решаем систему
        Phi = np.linalg.solve(K_global, F_global)
        print(f"   Система успешно решена")
                
    except np.linalg.LinAlgError as e:
        print(f"   Ошибка при решении системы: {e}")
        # Используем псевдообратную матрицу как запасной вариант
        try:
            Phi = np.linalg.lstsq(K_global, F_global, rcond=None)[0]
            print("   Система решена методом наименьших квадратов")
        except:
            print("   Не удалось решить систему")
            return None, None, None, None, None, None, None, None

    # --- Шаг 8: Вычисление крутящего момента ---
    integral_phi = 0.0
    for idx, elem in enumerate(elements):
        phi_values = [Phi[elem[0]], Phi[elem[1]], Phi[elem[2]]]
        avg_phi = np.mean(phi_values)
        integral_phi += areas[idx] * avg_phi

    T = 2 * G_theta * integral_phi

    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = 2 * Gθ * ∫∫φ dxdy = {T:.6f}")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---
G_theta = 3.6

# Решение задачи
result = solve_torsion_L_shape_by_fem(G_theta, print_matrices=False)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка конечных элементов
    # Рисуем только элементы внутри L-образной области
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.8)

    # Рисуем узлы
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)

    # Рисуем границу L-образной области
    L_boundary_x = [0, 0, 1, 1, 3, 3, 1, 1, 0]
    L_boundary_y = [0, 3, 3, 1, 1, 0, 0, 1, 1]
    ax1.plot(L_boundary_x, L_boundary_y, 'k-', linewidth=3, label='Граница L-образной области')

    ax1.set_title('Сетка конечных элементов L-образного сечения', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.set_xlim(-0.2, 3.2)
    ax1.set_ylim(-0.2, 3.2)

    # График 2: Распределение функции напряжения
    try:
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
        contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
        ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
        
        # Рисуем границу
        ax2.plot(L_boundary_x, L_boundary_y, 'k-', linewidth=2, label='Граница')
        
        ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10, zorder=5, alpha=0.6)
        
        plt.colorbar(contour, ax=ax2, label='φ(x, y)')
        ax2.set_title('Распределение функции напряжения φ', fontsize=14)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend()
        ax2.set_xlim(-0.2, 3.2)
        ax2.set_ylim(-0.2, 3.2)
    except Exception as e:
        print(f"Ошибка при построении контурного графика: {e}")
        ax2.set_title('Распределение функции напряжения φ', fontsize=14)
        ax2.text(0.5, 0.5, 'Ошибка визуализации', transform=ax2.transAxes, ha='center')

    # График 3: Визуализация матрицы жесткости
    im3 = ax3.imshow(np.abs(K_original), cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Абсолютное значение')
    ax3.set_title('Матрица жесткости K (до граничных условий)', fontsize=14)
    ax3.set_xlabel('Номер столбца')
    ax3.set_ylabel('Номер строки')

    # График 4: Визуализация матрицы жесткости после граничных условий
    im4 = ax4.imshow(np.abs(K_bc), cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar(im4, ax=ax4, label='Абсолютное значение')
    ax4.set_title('Матрица жесткости K (после граничных условий)', fontsize=14)
    ax4.set_xlabel('Номер столбца')
    ax4.set_ylabel('Номер строки')

    plt.tight_layout()
    plt.show()

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"Область: [0,1]×[0,3]∪[1,3]×[0,1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {len(nodes)}")
    print(f"Количество элементов: {len(elements)}")

# --- ФУНКЦИЯ ДЛЯ КРАТКОГО ВЫВОДА ---
def solve_torsion_L_shape_quick(G_theta):
    """Краткое решение без вывода матриц"""
    return solve_torsion_L_shape_by_fem(G_theta, print_matrices=False)

# Демонстрация
print("\nЗапуск расчета для L-образного сечения...")
result_quick = solve_torsion_L_shape_quick(G_theta)
if result_quick[2] is not None:
    print(f"Расчет завершен успешно! T = {result_quick[3]:.6f}")