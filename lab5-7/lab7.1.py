import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def is_in_L_shape(x, y):
    """
    Проверяет принадлежность точки (x, y) L-образной области D:
    - Вертикальный прямоугольник: x ∈ [0, 1], y ∈ [0, 3]
    - Горизонтальный прямоугольник: x ∈ [1, 3], y ∈ [0, 1]
    """
    vertical_rect = (0 <= x <= 1) and (0 <= y <= 3)
    horizontal_rect = (1 <= x <= 3) and (0 <= y <= 1)
    return vertical_rect or horizontal_rect

def solve_torsion_L_shape_by_fem(G_theta=3.6, print_matrices=True):
    """
    Решает задачу кручения стержня L-образного сечения методом конечных элементов.
    Область D: L-образная фигура
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СТЕРЖНЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("L-образная область:")
    print("  - Вертикальный прямоугольник: x ∈ [0, 1], y ∈ [0, 3]")
    print("  - Горизонтальный прямоугольник: x ∈ [1, 3], y ∈ [0, 1]")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов сетки ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")
    
    # Создаем равномерную сетку в прямоугольной области [0, 3] x [0, 3]
    # но используем только точки внутри L-образной области
    
    # Параметры сетки
    nx, ny = 8, 8  # Количество узлов по x и y
    x_coords = np.linspace(0, 3, nx)
    y_coords = np.linspace(0, 3, ny)
    
    # Отбираем только узлы внутри L-образной области
    nodes = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if is_in_L_shape(x, y):
                nodes.append([x, y])
    
    nodes = np.array(nodes)
    n_nodes = len(nodes)
    print(f"   Всего узлов: {n_nodes}")

    # Вывод координат узлов
    if print_matrices:
        print(f"\n   КООРДИНАТЫ УЗЛОВ:")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i, (x, y) in enumerate(nodes):
            print(f"   {i+1:6d} | {x:9.3f} | {y:9.3f}")

    # --- Шаг 2: Триангуляция области (Delaunay для L-образной области) ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    
    # Простая триангуляция - создаем треугольники вручную для L-образной области
    elements = []
    
    # Функция для получения индекса узла по координатам
    def get_node_index(x, y, tolerance=0.01):
        for i, node in enumerate(nodes):
            if abs(node[0] - x) < tolerance and abs(node[1] - y) < tolerance:
                return i
        return -1
    
    # Создаем триангуляцию - разбиваем на прямоугольники и затем на треугольники
    for i in range(len(x_coords)-1):
        for j in range(len(y_coords)-1):
            x1, x2 = x_coords[i], x_coords[i+1]
            y1, y2 = y_coords[j], y_coords[j+1]
            
            # Проверяем, находится ли прямоугольник внутри L-образной области
            corners = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            ]
            
            # Если хотя бы один угол внутри L-образной области, создаем элементы
            valid_corners = []
            for corner in corners:
                if is_in_L_shape(*corner):
                    idx = get_node_index(*corner)
                    if idx != -1:
                        valid_corners.append(idx)
            
            # Создаем треугольники из валидных углов
            if len(valid_corners) >= 3:
                # Первый треугольник
                if len(valid_corners) >= 3:
                    elements.append(valid_corners[:3])
                # Второй треугольник (если есть 4 угла)
                if len(valid_corners) == 4:
                    elements.append([valid_corners[0], valid_corners[2], valid_corners[3]])
    
    # Упрощенный подход: создаем регулярную триангуляцию
    elements = []
    for i in range(nx-1):
        for j in range(ny-1):
            # Индексы четырех углов прямоугольника
            idx1 = i * ny + j
            idx2 = i * ny + (j+1)
            idx3 = (i+1) * ny + j
            idx4 = (i+1) * ny + (j+1)
            
            # Проверяем, все ли узлы внутри L-образной области
            corners = [idx1, idx2, idx3, idx4]
            valid = True
            for idx in corners:
                if idx >= n_nodes or not is_in_L_shape(nodes[idx][0], nodes[idx][1]):
                    valid = False
                    break
            
            if valid:
                # Два треугольника для прямоугольника
                elements.append([idx1, idx2, idx3])
                elements.append([idx2, idx3, idx4])
    
    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    # Вывод информации об элементах
    if print_matrices:
        print(f"\n   ИНФОРМАЦИЯ ОБ ЭЛЕМЕНТАХ:")
        print("   № элемента | Узлы (индексы) |   Площадь   ")
        print("   -----------|----------------|-------------")
        for i, elem in enumerate(elements):
            p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            print(f"   {i+1:11d} | {elem[0]:2d}, {elem[1]:2d}, {elem[2]:2d}     | {area:10.3f}")

    # --- Шаг 3: Вычисление площадей элементов ---
    def triangle_area(p1, p2, p3):
        """Вычисляет площадь треугольника по координатам вершин."""
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    areas = []
    for elem in elements:
        p1 = nodes[elem[0]]
        p2 = nodes[elem[1]]
        p3 = nodes[elem[2]]
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
    
    # Вывод локальных матриц жесткости
    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ МАТРИЦЫ ЖЕСТКОСТИ:")
        for idx, elem in enumerate(elements):
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            
            print(f"\n   Элемент {idx+1} (узлы {elem[0]+1}, {elem[1]+1}, {elem[2]+1}):")
            for i in range(3):
                row_str = " ".join([f"{K_e[i, j]:10.6f}" for j in range(3)])
                print(f"      [{row_str}]")
            
            # Сборка в глобальную матрицу
            for i_local, i_global in enumerate(elem):
                for j_local, j_global in enumerate(elem):
                    K_global[i_global, j_global] += K_e[i_local, j_local]
    else:
        for idx, elem in enumerate(elements):
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            
            # Сборка в глобальную матрицу
            for i_local, i_global in enumerate(elem):
                for j_local, j_global in enumerate(elem):
                    K_global[i_global, j_global] += K_e[i_local, j_local]

    # Вывод глобальной матрицы жесткости
    if print_matrices:
        print(f"\n   ГЛОБАЛЬНАЯ МАТРИЦА ЖЕСТКОСТИ K (размер {n_nodes}×{n_nodes}):")
        print("   (показаны первые 8 строк и столбцов)")
        for i in range(min(8, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(8, n_nodes))])
            print(f"   [{row_str}]")
        if n_nodes > 8:
            print(f"   ... и еще {n_nodes-8} строк и столбцов")

    # --- Шаг 5: Построение вектора нагрузки ---
    F_global = np.zeros(n_nodes)

    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")
    
    # Вывод локальных векторов нагрузки
    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ ВЕКТОРЫ НАГРУЗКИ:")
        for idx, elem in enumerate(elements):
            # Вектор нагрузки для элемента
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            
            print(f"   Элемент {idx+1}: [{f_e[0]:.6f}, {f_e[1]:.6f}, {f_e[2]:.6f}]")
            
            # Сборка в глобальный вектор
            for i_local, i_global in enumerate(elem):
                F_global[i_global] += f_e[i_local]
    else:
        for idx, elem in enumerate(elements):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            for i_local, i_global in enumerate(elem):
                F_global[i_global] += f_e[i_local]

    # Вывод глобального вектора нагрузки
    if print_matrices:
        print(f"\n   ГЛОБАЛЬНЫЙ ВЕКТОР НАГРУЗКИ F:")
        for i in range(min(10, len(F_global))):
            print(f"   F[{i+1}] = {F_global[i]:.6f}")
        if len(F_global) > 10:
            print(f"   ... и еще {len(F_global)-10} элементов")

    # --- Шаг 6: Граничные условия ---
    # Все узлы на границе L-образной области имеют φ = 0
    boundary_nodes = []
    for i, (x, y) in enumerate(nodes):
        # Проверяем, находится ли узел на границе
        on_boundary = False
        
        # Границы вертикального прямоугольника
        if (abs(x - 0) < 0.01 or abs(x - 1) < 0.01) and (0 <= y <= 3):
            on_boundary = True
        if (abs(y - 0) < 0.01 or abs(y - 3) < 0.01) and (0 <= x <= 1):
            on_boundary = True
            
        # Границы горизонтального прямоугольника  
        if (abs(x - 1) < 0.01 or abs(x - 3) < 0.01) and (0 <= y <= 1):
            on_boundary = True
        if (abs(y - 0) < 0.01 or abs(y - 1) < 0.01) and (1 <= x <= 3):
            on_boundary = True
            
        if on_boundary:
            boundary_nodes.append(i)

    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes)}")

    # Сохраняем оригинальные матрицы для вывода
    K_original = K_global.copy()
    F_original = F_global.copy()

    # Модификация системы
    for node_idx in boundary_nodes:
        K_global[node_idx, :] = 0.0
        K_global[:, node_idx] = 0.0
        K_global[node_idx, node_idx] = 1.0
        F_global[node_idx] = 0.0

    # Вывод модифицированных матриц
    if print_matrices:
        print(f"\n   МАТРИЦА ПОСЛЕ ГРАНИЧНЫХ УСЛОВИЙ:")
        print("   (первые 8 строк и столбцов)")
        for i in range(min(8, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(8, n_nodes))])
            print(f"   [{row_str}]")
        
        print(f"\n   ВЕКТОР НАГРУЗКИ ПОСЛЕ ГРАНИЧНЫХ УСЛОВИЙ:")
        for i in range(min(10, len(F_global))):
            print(f"   F[{i+1}] = {F_global[i]:.6f}")

    # --- Шаг 7: Решение системы ---
    try:
        Phi = np.linalg.solve(K_global, F_global)
        print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
        print(f"   Система успешно решена")
        
        # Вывод решения
        if print_matrices:
            print(f"\n   РЕШЕНИЕ СИСТЕМЫ (вектор φ):")
            for i in range(min(10, len(Phi))):
                print(f"   φ[{i+1}] = {Phi[i]:.6f}")
            if len(Phi) > 10:
                print(f"   ... и еще {len(Phi)-10} элементов")
                
    except np.linalg.LinAlgError:
        print("   Ошибка: Матрица вырождена")
        return None, None, None, None

    # --- Шаг 8: Вычисление крутящего момента ---
    integral_phi = 0.0
    for idx, elem in enumerate(elements):
        avg_phi = np.mean([Phi[elem[0]], Phi[elem[1]], Phi[elem[2]]])
        integral_phi += areas[idx] * avg_phi

    T = 2 * G_theta * integral_phi

    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = 2 * Gθ * ∫∫φ dxdy = {T:.3f}")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---
G_theta = 3.6  # Параметр кручения для варианта 18

# Решение задачи
result = solve_torsion_L_shape_by_fem(G_theta, print_matrices=True)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка конечных элементов
    # Рисуем L-образную область
    vertical_rect = patches.Rectangle((0, 0), 1, 3, linewidth=2, edgecolor='black', 
                                    facecolor='lightblue', alpha=0.3, label='Вертикальная часть')
    horizontal_rect = patches.Rectangle((1, 0), 2, 1, linewidth=2, edgecolor='black', 
                                      facecolor='lightgreen', alpha=0.3, label='Горизонтальная часть')
    
    ax1.add_patch(vertical_rect)
    ax1.add_patch(horizontal_rect)

    # Рисуем элементы
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)

    # Рисуем узлы
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=30, zorder=5)

    ax1.set_title('Сетка конечных элементов L-образной области', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 3.5)

    # График 2: Распределение функции напряжения
    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
    ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    # Рисуем границы L-образной области
    ax2.plot([0, 0], [0, 3], 'k-', linewidth=2, label='Граница')
    ax2.plot([0, 1], [3, 3], 'k-', linewidth=2)
    ax2.plot([1, 1], [1, 3], 'k-', linewidth=2)
    ax2.plot([1, 3], [1, 1], 'k-', linewidth=2)
    ax2.plot([3, 3], [0, 1], 'k-', linewidth=2)
    ax2.plot([1, 3], [0, 0], 'k-', linewidth=2)
    ax2.plot([0, 1], [0, 0], 'k-', linewidth=2)
    
    ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)
    
    plt.colorbar(contour, ax=ax2, label='φ(x, y)')
    ax2.set_title('Распределение функции напряжения φ', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 3.5)

    # График 3: Визуализация матрицы жесткости (до граничных условий)
    im3 = ax3.imshow(np.abs(K_original), cmap='hot', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='Абсолютное значение')
    ax3.set_title('Матрица жесткости K (до граничных условий)', fontsize=14)
    ax3.set_xlabel('Номер столбца')
    ax3.set_ylabel('Номер строки')

    # График 4: Визуализация матрицы жесткости (после граничных условий)
    im4 = ax4.imshow(np.abs(K_bc), cmap='hot', aspect='auto')
    plt.colorbar(im4, ax=ax4, label='Абсолютное значение')
    ax4.set_title('Матрица жесткости K (после граничных условий)', fontsize=14)
    ax4.set_xlabel('Номер столбца')
    ax4.set_ylabel('Номер строки')

    plt.tight_layout()
    plt.show()

    # --- ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ МАТРИЦ ---
    print("\n" + "="*60)
    print("АНАЛИЗ МАТРИЦ")
    print("="*60)
    
    print(f"Свойства матрицы жесткости (до граничных условий):")
    print(f"   Размер: {K_original.shape}")
    print(f"   Норма: {np.linalg.norm(K_original):.6f}")
    print(f"   Определитель: {np.linalg.det(K_original):.6e}")
    print(f"   Число обусловленности: {np.linalg.cond(K_original):.6e}")
    
    print(f"\nСвойства матрицы жесткости (после граничных условий):")
    print(f"   Размер: {K_bc.shape}")
    print(f"   Норма: {np.linalg.norm(K_bc):.6f}")
    print(f"   Определитель: {np.linalg.det(K_bc):.6e}")
    print(f"   Число обусловленности: {np.linalg.cond(K_bc):.6e}")

    # --- ВЫВОД РЕЗУЛЬТАТОВ В ТАБЛИЦУ ---
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print("L-образная область:")
    print("  - Вертикальный прямоугольник: x ∈ [0, 1], y ∈ [0, 3]")
    print("  - Горизонтальный прямоугольник: x ∈ [1, 3], y ∈ [0, 1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {len(nodes)}")
    print(f"Количество элементов: {len(elements)}")
    
    print(f"\nЗНАЧЕНИЯ ФУНКЦИИ НАПРЯЖЕНИЯ В УЗЛАХ:")
    print("№ узла |     x     |     y     |    φ(x,y)    ")
    print("-------|-----------|-----------|--------------")
    for i in range(len(nodes)):
        print(f"{i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f} | {Phi[i]:12.6f}")

# --- ФУНКЦИЯ ДЛЯ КРАТКОГО ВЫВОДА (без матриц) ---
def solve_torsion_L_shape_quick(G_theta=3.6):
    """Краткое решение без вывода матриц"""
    return solve_torsion_L_shape_by_fem(G_theta, print_matrices=False)

# Пример быстрого расчета
# nodes, elements, Phi, T, _, _, _, _ = solve_torsion_L_shape_quick(3.0)