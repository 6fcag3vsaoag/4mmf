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
    
    # Вертикальная часть: [0,1]×[0,3] — включая x=1
    nx_vert, ny_vert = 8, 24
    for i in range(nx_vert + 1):
        x = i / nx_vert  # x от 0 до 1 (включительно)
        for j in range(ny_vert + 1):
            y = 3 * j / ny_vert  # y от 0 до 3
            all_nodes.append([x, y])
    
    n_vertical = len(all_nodes)
    print(f"   Узлов в вертикальной части: {n_vertical}")
    
    # Горизонтальная часть: [1,3]×[0,1] — начинаем с x=1 (включительно), но пропускаем y>1
    nx_horiz, ny_horiz = 16, 8
    for i in range(nx_horiz + 1):  # теперь включаем i=0 → x=1
        x = 1 + 2 * i / nx_horiz  # x от 1 до 3
        for j in range(ny_horiz + 1):
            y = j / ny_horiz  # y от 0 до 1
            # Добавляем ВСЕ точки, даже на x=1 — дубликаты удалим позже
            all_nodes.append([x, y])
    
    n_horizontal = len(all_nodes) - n_vertical
    print(f"   Узлов в горизонтальной части: {n_horizontal}")
    
    # Дополнительные узлы в критическом углу для улучшения связности
    corner_nodes = [
        [0.95, 0.95], [1.05, 0.95],
        [0.95, 1.05], [1.05, 1.05],
        [0.98, 0.98], [1.02, 0.98],
        [0.98, 1.02], [1.02, 1.02],
    ]
    all_nodes.extend(corner_nodes)
    
    # Преобразуем в массив и удаляем дубликаты
    nodes = np.array(all_nodes)
    rounded = np.round(nodes, decimals=8)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    nodes = nodes[np.sort(unique_indices)]
    n_nodes = len(nodes)
    print(f"   Всего узлов после удаления дубликатов: {n_nodes}")

    # Вывод координат узлов (ограниченное количество)
    if print_matrices:
        max_nodes_to_print = 20
        print(f"\n   КООРДИНАТЫ УЗЛОВ:")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i, (x, y) in enumerate(nodes[:max_nodes_to_print]):
            print(f"   {i+1:6d} | {x:9.3f} | {y:9.3f}")
        if n_nodes > max_nodes_to_print:
            print(f"   ... и еще {n_nodes - max_nodes_to_print} узлов")

    # --- Шаг 2: Триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    
    def is_in_L_shape(x, y, tol=1e-10):
        """Проверяет, находится ли точка внутри L-образной области"""
        vertical = (0 - tol <= x <= 1 + tol) and (0 - tol <= y <= 3 + tol)
        horizontal = (1 - tol <= x <= 3 + tol) and (0 - tol <= y <= 1 + tol)
        return vertical or horizontal
    
    # Генерация элементов без Delaunay — по прямоугольным подобластям
    elements = []
    
    # Карта координат -> индекс узла
    node_map = {}
    for idx, (x, y) in enumerate(nodes):
        key = (round(x, 8), round(y, 8))
        node_map[key] = idx

    # Триангуляция вертикальной части [0,1]×[0,3]
    x_vert = np.linspace(0, 1, nx_vert + 1)
    y_vert = np.linspace(0, 3, ny_vert + 1)
    for i in range(nx_vert):
        for j in range(ny_vert):
            # Четыре угла прямоугольника
            coords = [
                (x_vert[i],     y_vert[j]),
                (x_vert[i+1],   y_vert[j]),
                (x_vert[i+1],   y_vert[j+1]),
                (x_vert[i],     y_vert[j+1])
            ]
            # Проверяем, что все точки внутри области
            if all(is_in_L_shape(x, y) for x, y in coords):
                try:
                    idx0 = node_map[(round(coords[0][0], 8), round(coords[0][1], 8))]
                    idx1 = node_map[(round(coords[1][0], 8), round(coords[1][1], 8))]
                    idx2 = node_map[(round(coords[2][0], 8), round(coords[2][1], 8))]
                    idx3 = node_map[(round(coords[3][0], 8), round(coords[3][1], 8))]
                    # Два треугольника
                    elements.append([idx0, idx1, idx2])
                    elements.append([idx0, idx2, idx3])
                except KeyError:
                    continue  # Пропускаем, если узел не найден

    # Триангуляция горизонтальной части [1,3]×[0,1]
    x_horiz = np.linspace(1, 3, nx_horiz + 1)
    y_horiz = np.linspace(0, 1, ny_horiz + 1)
    for i in range(nx_horiz):
        for j in range(ny_horiz):
            coords = [
                (x_horiz[i],     y_horiz[j]),
                (x_horiz[i+1],   y_horiz[j]),
                (x_horiz[i+1],   y_horiz[j+1]),
                (x_horiz[i],     y_horiz[j+1])
            ]
            if all(is_in_L_shape(x, y) for x, y in coords):
                try:
                    idx0 = node_map[(round(coords[0][0], 8), round(coords[0][1], 8))]
                    idx1 = node_map[(round(coords[1][0], 8), round(coords[1][1], 8))]
                    idx2 = node_map[(round(coords[2][0], 8), round(coords[2][1], 8))]
                    idx3 = node_map[(round(coords[3][0], 8), round(coords[3][1], 8))]
                    elements.append([idx0, idx1, idx2])
                    elements.append([idx0, idx2, idx3])
                except KeyError:
                    continue

    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    # Вывод информации об элементах (ограниченное количество)
    if print_matrices:
        max_elements_to_print = 15
        print(f"\n   ИНФОРМАЦИЯ ОБ ЭЛЕМЕНТАХ:")
        print("   № элемента | Узлы (индексы) |   Площадь   ")
        print("   -----------|----------------|-------------")
        for i, elem in enumerate(elements[:max_elements_to_print]):
            p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            print(f"   {i+1:11d} | {elem[0]:2d}, {elem[1]:2d}, {elem[2]:2d}     | {area:10.3f}")
        if n_elements > max_elements_to_print:
            print(f"   ... и еще {n_elements - max_elements_to_print} элементов")

    # --- Шаг 3: Вычисление площадей элементов ---
    def triangle_area(p1, p2, p3):
        """Вычисляет площадь треугольника по координатам вершин."""
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    areas = []
    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        area = triangle_area(p1, p2, p3)
        areas.append(area)

    max_areas_to_print = 10
    print(f"\n   Площади элементов (первые {max_areas_to_print}):")
    for i in range(min(max_areas_to_print, len(areas))):
        print(f"   S{i+1} = {areas[i]:.3f}")
    if len(areas) > max_areas_to_print:
        print(f"   ... и еще {len(areas) - max_areas_to_print} элементов")

    # --- Шаг 4: Построение матриц жесткости ---
    def element_stiffness_matrix(elem_nodes, area):
        """
        Строит локальную матрицу жесткости для треугольного элемента.
        """
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

    if print_matrices:
        max_local_matrices_to_print = 5
        print(f"\n   ЛОКАЛЬНЫЕ МАТРИЦЫ ЖЕСТКОСТИ (первые {max_local_matrices_to_print}):")
        for idx, elem in enumerate(elements[:max_local_matrices_to_print]):
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            print(f"\n   Элемент {idx+1} (узлы {elem[0]+1}, {elem[1]+1}, {elem[2]+1}):")
            for i in range(3):
                row_str = " ".join([f"{K_e[i, j]:10.6f}" for j in range(3)])
                print(f"      [{row_str}]")
        if n_elements > max_local_matrices_to_print:
            print(f"   ... и еще {n_elements - max_local_matrices_to_print} локальных матриц")

    if print_matrices:
        max_matrix_rows_to_print = 8
        print(f"\n   ГЛОБАЛЬНАЯ МАТРИЦА ЖЕСТКОСТИ K (размер {n_nodes}×{n_nodes}):")
        print(f"   (показаны первые {max_matrix_rows_to_print} строк и столбцов)")
        for i in range(min(max_matrix_rows_to_print, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(max_matrix_rows_to_print, n_nodes))])
            print(f"   [{row_str}]")
        if n_nodes > max_matrix_rows_to_print:
            print(f"   ... и еще {n_nodes - max_matrix_rows_to_print} строк и столбцов")

    # --- Шаг 5: Построение вектора нагрузки ---
    F_global = np.zeros(n_nodes)
    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")
    
    for idx, elem in enumerate(elements):
        f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
        for i_local, i_global in enumerate(elem):
            F_global[i_global] += f_e[i_local]

    if print_matrices:
        max_local_vectors_to_print = 5
        print(f"\n   ЛОКАЛЬНЫЕ ВЕКТОРЫ НАГРУЗКИ (первые {max_local_vectors_to_print}):")
        for idx, elem in enumerate(elements[:max_local_vectors_to_print]):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            print(f"   Элемент {idx+1}: [{f_e[0]:.6f}, {f_e[1]:.6f}, {f_e[2]:.6f}]")
        if n_elements > max_local_vectors_to_print:
            print(f"   ... и еще {n_elements - max_local_vectors_to_print} локальных векторов")

    if print_matrices:
        max_vector_to_print = 10
        print(f"\n   ГЛОБАЛЬНЫЙ ВЕКТОР НАГРУЗКИ F:")
        for i in range(min(max_vector_to_print, len(F_global))):
            print(f"   F[{i+1}] = {F_global[i]:.6f}")
        if len(F_global) > max_vector_to_print:
            print(f"   ... и еще {len(F_global) - max_vector_to_print} элементов")

    # --- Шаг 6: Граничные условия ---
    boundary_nodes = []
    tol = 1e-10
    for i, (x, y) in enumerate(nodes):
        on_boundary = False
        if (abs(x) < tol and 0 <= y <= 3):  # левая
            on_boundary = True
        elif (abs(y - 3) < tol and 0 <= x <= 1):  # верх верт
            on_boundary = True
        elif (abs(x - 1) < tol and 1 <= y <= 3):  # правая верт
            on_boundary = True
        elif (abs(y - 1) < tol and 1 <= x <= 3):  # верх гориз
            on_boundary = True
        elif (abs(x - 3) < tol and 0 <= y <= 1):  # правая
            on_boundary = True
        elif (abs(y) < tol and 0 <= x <= 3):  # нижняя
            on_boundary = True
        if on_boundary:
            boundary_nodes.append(i)

    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes)}")

    K_original = K_global.copy()
    F_original = F_global.copy()

    for node_idx in boundary_nodes:
        K_global[node_idx, :] = 0.0
        K_global[:, node_idx] = 0.0
        K_global[node_idx, node_idx] = 1.0
        F_global[node_idx] = 0.0

    if print_matrices:
        max_matrix_rows_to_print = 8
        print(f"\n   МАТРИЦА ПОСЛЕ ГРАНИЧНЫХ УСЛОВИЙ:")
        for i in range(min(max_matrix_rows_to_print, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(max_matrix_rows_to_print, n_nodes))])
            print(f"   [{row_str}]")
        
        max_vector_to_print = 10
        print(f"\n   ВЕКТОР НАГРУЗКИ ПОСЛЕ ГРАНИЧНЫХ УСЛОВИЙ:")
        for i in range(min(max_vector_to_print, len(F_global))):
            print(f"   F[{i+1}] = {F_global[i]:.6f}")
        if len(F_global) > max_vector_to_print:
            print(f"   ... и еще {len(F_global) - max_vector_to_print} элементов")

    # --- Шаг 7: Решение системы ---
    print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
    
    try:
        Phi = np.linalg.solve(K_global, F_global)
        print(f"   Система успешно решена")
        if print_matrices:
            max_solution_to_print = 15
            print(f"\n   РЕШЕНИЕ СИСТЕМЫ (вектор φ):")
            for i in range(min(max_solution_to_print, len(Phi))):
                print(f"   φ[{i+1}] = {Phi[i]:.6f}")
            if len(Phi) > max_solution_to_print:
                print(f"   ... и еще {len(Phi) - max_solution_to_print} элементов")
    except np.linalg.LinAlgError as e:
        print(f"   Ошибка при решении системы: {e}")
        return None, None, None, None, None, None, None, None

    # --- Шаг 8: Вычисление крутящего момента ---
    integral_phi = 0.0
    for idx, elem in enumerate(elements):
        avg_phi = np.mean([Phi[elem[0]], Phi[elem[1]], Phi[elem[2]]])
        integral_phi += areas[idx] * avg_phi
    T = 2 * G_theta * integral_phi

    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = 2 * Gθ * ∫∫φ dxdy = {T:.6f}")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---
G_theta = 3.6
result = solve_torsion_L_shape_by_fem(G_theta, print_matrices=True)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # --- АНАЛИЗ МАТРИЦ ---
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

    # --- ИТОГОВЫЕ РЕЗУЛЬТАТЫ ---
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"Область: [0,1]×[0,3]∪[1,3]×[0,1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {len(nodes)}")
    print(f"Количество элементов: {len(elements)}")
    
    max_nodes_to_print = 15
    print(f"\nЗНАЧЕНИЯ ФУНКЦИИ НАПРЯЖЕНИЯ В УЗЛАХ:")
    print("№ узла |     x     |     y     |    φ(x,y)    ")
    print("-------|-----------|-----------|--------------")
    for i in range(min(max_nodes_to_print, len(nodes))):
        print(f"{i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f} | {Phi[i]:12.6f}")
    if len(nodes) > max_nodes_to_print:
        print(f"... и еще {len(nodes) - max_nodes_to_print} узлов")

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Сетка
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.8)
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)
    L_boundary_x = [0, 0, 1, 1, 3, 3, 1, 1, 0]
    L_boundary_y = [0, 3, 3, 1, 1, 0, 0, 1, 1]
    ax1.plot(L_boundary_x, L_boundary_y, 'k-', linewidth=3)
    ax1.set_title('Сетка конечных элементов L-образного сечения', fontsize=14)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3); ax1.axis('equal')
    ax1.set_xlim(-0.2, 3.2); ax1.set_ylim(-0.2, 3.2)

    # Распределение φ
    try:
        tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
        contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
        ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
        ax2.plot(L_boundary_x, L_boundary_y, 'k-', linewidth=2)
        ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10, alpha=0.6)
        plt.colorbar(contour, ax=ax2, label='φ(x, y)')
        ax2.set_title('Распределение функции напряжения φ', fontsize=14)
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.grid(True, alpha=0.3); ax2.axis('equal')
        ax2.set_xlim(-0.2, 3.2); ax2.set_ylim(-0.2, 3.2)
    except Exception as e:
        ax2.text(0.5, 0.5, 'Ошибка визуализации', transform=ax2.transAxes, ha='center')

    # Матрицы
    im3 = ax3.imshow(np.abs(K_original), cmap='hot', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='|K|')
    ax3.set_title('Матрица жесткости K (до ГУ)', fontsize=14)

    im4 = ax4.imshow(np.abs(K_bc), cmap='hot', aspect='auto')
    plt.colorbar(im4, ax=ax4, label='|K|')
    ax4.set_title('Матрица жесткости K (после ГУ)', fontsize=14)

    plt.tight_layout()
    plt.show()

def solve_torsion_L_shape_quick(G_theta):
    return solve_torsion_L_shape_by_fem(G_theta, print_matrices=False)

print("\nЗапуск расчета для L-образного сечения...")
result_quick = solve_torsion_L_shape_quick(G_theta)
if result_quick[2] is not None:
    print(f"Расчет завершен успешно! T = {result_quick[3]:.6f}")