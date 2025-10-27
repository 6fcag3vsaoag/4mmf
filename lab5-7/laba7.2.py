import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def solve_torsion_L_shape_by_fem(G_theta=3.6, print_matrices=True):
    """
    Решает задачу кручения стержня с L-образным сечением методом конечных элементов.
    Область D: [0,1]×[0,3] ∪ [1,3]×[0,1]
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СЕЧЕНИЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("Область: [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов сетки ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")

    # Граница L-области (по часовой стрелке)
    boundary_points = [
        [0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]
    ]
    boundary = np.array(boundary_points)

    # Функция принадлежности точки L-области
    def is_inside_L(x, y):
        return (0 <= x <= 1 and 0 <= y <= 3) or (1 <= x <= 3 and 0 <= y <= 1)

    # Узлы на границе (равномерно по периметру)
    num_boundary_nodes = 24
    edge_lengths = []
    for i in range(len(boundary) - 1):
        p1, p2 = boundary[i], boundary[i+1]
        edge_lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(edge_lengths)
    spacing = total_length / num_boundary_nodes

    boundary_nodes = []
    current_length = 0.0
    for i in range(len(boundary) - 1):
        p1, p2 = boundary[i], boundary[i+1]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0:
            continue
        unit_vec = edge_vec / edge_len
        while current_length < sum(edge_lengths[:i+1]) - 1e-9:
            s = current_length - sum(edge_lengths[:i])
            point = p1 + s * unit_vec
            boundary_nodes.append(point.tolist())
            current_length += spacing
    # Добавляем последнюю точку, если нужно
    if len(boundary_nodes) < num_boundary_nodes:
        boundary_nodes.append(boundary[-1].tolist())
    boundary_nodes = boundary_nodes[:num_boundary_nodes]

    n_boundary = len(boundary_nodes)
    print(f"   Узлов на границе: {n_boundary}")

    # Внутренние узлы (вручную или по сетке)
    interior_candidates = []
    for x in np.linspace(0.2, 2.8, 8):
        for y in np.linspace(0.2, 2.8, 8):
            if is_inside_L(x, y):
                interior_candidates.append([x, y])

    # Удаляем дубликаты и близкие точки
    interior_nodes = []
    min_dist = 0.3
    for p in interior_candidates:
        too_close = False
        for q in interior_nodes:
            if np.linalg.norm(np.array(p) - np.array(q)) < min_dist:
                too_close = True
                break
        if not too_close:
            interior_nodes.append(p)

    n_interior = len(interior_nodes)
    print(f"   Внутренних узлов: {n_interior}")

    # Объединяем узлы
    all_nodes = boundary_nodes + interior_nodes
    nodes = np.array(all_nodes)
    n_nodes = len(nodes)
    print(f"   Всего узлов: {n_nodes}")

    if print_matrices:
        print(f"\n   КООРДИНАТЫ УЗЛОВ:")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i, (x, y) in enumerate(nodes):
            print(f"   {i+1:6d} | {x:9.3f} | {y:9.3f}")

    # --- Шаг 2: Триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")

    from scipy.spatial import Delaunay

    # Триангуляция всех узлов
    tri = Delaunay(nodes)
    elements = []

    for simplex in tri.simplices:
        # Центр треугольника
        centroid = np.mean(nodes[simplex], axis=0)
        if is_inside_L(centroid[0], centroid[1]):
            elements.append(simplex.tolist())

    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    if print_matrices:
        print(f"\n   ИНФОРМАЦИЯ ОБ ЭЛЕМЕНТАХ:")
        print("   № элемента | Узлы (индексы) |   Площадь   ")
        print("   -----------|----------------|-------------")
        for i, elem in enumerate(elements):
            p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            print(f"   {i+1:11d} | {elem[0]:2d}, {elem[1]:2d}, {elem[2]:2d}     | {area:10.3f}")

    # --- Шаг 3: Площади элементов ---
    def triangle_area(p1, p2, p3):
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

    # --- Шаг 4: Матрицы жёсткости ---
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

    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ МАТРИЦЫ ЖЕСТКОСТИ:")
        for idx, elem in enumerate(elements):
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            print(f"\n   Элемент {idx+1} (узлы {elem[0]+1}, {elem[1]+1}, {elem[2]+1}):")
            for i in range(3):
                row_str = " ".join([f"{K_e[i, j]:10.6f}" for j in range(3)])
                print(f"      [{row_str}]")
            for i_local, i_global in enumerate(elem):
                for j_local, j_global in enumerate(elem):
                    K_global[i_global, j_global] += K_e[i_local, j_local]
    else:
        for idx, elem in enumerate(elements):
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            for i_local, i_global in enumerate(elem):
                for j_local, j_global in enumerate(elem):
                    K_global[i_global, j_global] += K_e[i_local, j_local]

    if print_matrices:
        print(f"\n   ГЛОБАЛЬНАЯ МАТРИЦА ЖЕСТКОСТИ K (размер {n_nodes}×{n_nodes}):")
        print("   (показаны первые 8 строк и столбцов)")
        for i in range(min(8, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(8, n_nodes))])
            print(f"   [{row_str}]")
        if n_nodes > 8:
            print(f"   ... и еще {n_nodes-8} строк и столбцов")

    # --- Шаг 5: Вектор нагрузки ---
    F_global = np.zeros(n_nodes)
    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")

    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ ВЕКТОРЫ НАГРУЗКИ:")
        for idx, elem in enumerate(elements):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            print(f"   Элемент {idx+1}: [{f_e[0]:.6f}, {f_e[1]:.6f}, {f_e[2]:.6f}]")
            for i_local, i_global in enumerate(elem):
                F_global[i_global] += f_e[i_local]
    else:
        for idx, elem in enumerate(elements):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            for i_local, i_global in enumerate(elem):
                F_global[i_global] += f_e[i_local]

    if print_matrices:
        print(f"\n   ГЛОБАЛЬНЫЙ ВЕКТОР НАГРУЗКИ F:")
        for i in range(min(10, len(F_global))):
            print(f"   F[{i+1}] = {F_global[i]:.6f}")
        if len(F_global) > 10:
            print(f"   ... и еще {len(F_global)-10} элементов")

    # --- Шаг 6: Граничные условия ---
    boundary_nodes_indices = list(range(n_boundary))
    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes_indices)}")

    K_original = K_global.copy()
    F_original = F_global.copy()

    for node_idx in boundary_nodes_indices:
        K_global[node_idx, :] = 0.0
        K_global[:, node_idx] = 0.0
        K_global[node_idx, node_idx] = 1.0
        F_global[node_idx] = 0.0

    if print_matrices:
        print(f"\n   МАТРИЦА ПОСЛЕ ГРАНИЧНЫХ УСЛОВИЙ:")
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
        if print_matrices:
            print(f"\n   РЕШЕНИЕ СИСТЕМЫ (вектор φ):")
            for i in range(min(10, len(Phi))):
                print(f"   φ[{i+1}] = {Phi[i]:.6f}")
            if len(Phi) > 10:
                print(f"   ... и еще {len(Phi)-10} элементов")
    except np.linalg.LinAlgError:
        print("   Ошибка: Матрица вырождена")
        return None, None, None, None

    # --- Шаг 8: Крутящий момент ---
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

result = solve_torsion_L_shape_by_fem(G_theta=G_theta, print_matrices=True)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка
    boundary_path = Path([
        [0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]
    ])
    patch = PathPatch(boundary_path, facecolor='lightblue', edgecolor='black', lw=2, alpha=0.4)
    ax1.add_patch(patch)

    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)

    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=30, zorder=5)
    ax1.set_title('Сетка конечных элементов (L-область)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-0.2, 3.2)
    ax1.set_ylim(-0.2, 3.2)

    # График 2: Распределение φ
    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
    ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax2.add_patch(PathPatch(boundary_path, facecolor='none', edgecolor='black', lw=2))
    ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)
    plt.colorbar(contour, ax=ax2, label='φ(x, y)')
    ax2.set_title('Распределение функции напряжения φ', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim(-0.2, 3.2)
    ax2.set_ylim(-0.2, 3.2)

    # График 3 и 4: Матрицы жёсткости
    im3 = ax3.imshow(np.abs(K_original), cmap='hot', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='|K|')
    ax3.set_title('Матрица жесткости K (до ГУ)', fontsize=14)

    im4 = ax4.imshow(np.abs(K_bc), cmap='hot', aspect='auto')
    plt.colorbar(im4, ax=ax4, label='|K|')
    ax4.set_title('Матрица жесткости K (после ГУ)', fontsize=14)

    plt.tight_layout()
    plt.show()

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

    # --- ИТОГОВАЯ ТАБЛИЦА ---
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"Область: L-образная [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {len(nodes)}")
    print(f"Количество элементов: {len(elements)}")
    print(f"\nЗНАЧЕНИЯ ФУНКЦИИ НАПРЯЖЕНИЯ В УЗЛАХ:")
    print("№ узла |     x     |     y     |    φ(x,y)    ")
    print("-------|-----------|-----------|--------------")
    for i in range(len(nodes)):
        print(f"{i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f} | {Phi[i]:12.6f}")