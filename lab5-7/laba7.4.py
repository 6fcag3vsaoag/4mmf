import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def solve_torsion_L_shape_by_fem(G_theta=3.6, nx_vert=10, ny_vert=30, nx_horiz=20, ny_horiz=10, print_matrices=True):
    """
    Решает задачу кручения стержня с L-образным сечением методом конечных элементов.
    Область D: [0,1]×[0,3] ∪ [1,3]×[0,1]
    Сетка строится вручную по прямоугольным подобластям → гарантирует полное покрытие.
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СЕЧЕНИЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("Область: [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print(f"Сетка: вертикальная часть {nx_vert}×{ny_vert}, горизонтальная {nx_horiz}×{ny_horiz}")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")

    # Вертикальная часть: [0,1] x [0,3]
    x_vert = np.linspace(0, 1, nx_vert)
    y_vert = np.linspace(0, 3, ny_vert)
    X_vert, Y_vert = np.meshgrid(x_vert, y_vert)
    nodes_vert = np.column_stack([X_vert.ravel(), Y_vert.ravel()])

    # Горизонтальная часть: [1,3] x [0,1]
    x_horiz = np.linspace(1, 3, nx_horiz)
    y_horiz = np.linspace(0, 1, ny_horiz)
    X_horiz, Y_horiz = np.meshgrid(x_horiz, y_horiz)
    nodes_horiz = np.column_stack([X_horiz.ravel(), Y_horiz.ravel()])

    # Объединяем узлы и удаляем дубликаты (в пересечении [1]×[0,1])
    all_nodes = np.vstack([nodes_vert, nodes_horiz])
    # Удаление дубликатов с точностью
    rounded = np.round(all_nodes, decimals=8)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    nodes = all_nodes[np.sort(unique_indices)]
    n_nodes = len(nodes)

    print(f"   Всего узлов после объединения: {n_nodes}")

    if print_matrices:
        print(f"\n   КООРДИНАТЫ УЗЛОВ (первые 10):")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i in range(min(10, n_nodes)):
            print(f"   {i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f}")
        if n_nodes > 10:
            print(f"   ... и ещё {n_nodes - 10} узлов")

    # --- Шаг 2: Генерация элементов (треугольников) ---
    print(f"\n2. ГЕНЕРАЦИЯ ЭЛЕМЕНТОВ:")

    def create_elements_from_grid(x_vals, y_vals, node_map_func):
        """Создаёт треугольные элементы из прямоугольной сетки."""
        nx, ny = len(x_vals), len(y_vals)
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Индексы четырёх узлов прямоугольника
                n1 = node_map_func(i, j)
                n2 = node_map_func(i + 1, j)
                n3 = node_map_func(i + 1, j + 1)
                n4 = node_map_func(i, j + 1)
                # Два треугольника
                elements.append([n1, n2, n3])
                elements.append([n1, n3, n4])
        return elements

    # Создаём карты индексов для каждой подобласти
    def get_index_vert(i, j):
        return np.where((np.isclose(nodes[:,0], x_vert[i])) & (np.isclose(nodes[:,1], y_vert[j])))[0][0]

    def get_index_horiz(i, j):
        return np.where((np.isclose(nodes[:,0], x_horiz[i])) & (np.isclose(nodes[:,1], y_horiz[j])))[0][0]

    # Генерация элементов
    elements = []

    # Элементы вертикальной части
    for j in range(ny_vert - 1):
        for i in range(nx_vert - 1):
            n1 = get_index_vert(i, j)
            n2 = get_index_vert(i + 1, j)
            n3 = get_index_vert(i + 1, j + 1)
            n4 = get_index_vert(i, j + 1)
            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])

    # Элементы горизонтальной части
    for j in range(ny_horiz - 1):
        for i in range(nx_horiz - 1):
            n1 = get_index_horiz(i, j)
            n2 = get_index_horiz(i + 1, j)
            n3 = get_index_horiz(i + 1, j + 1)
            n4 = get_index_horiz(i, j + 1)
            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])

    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    if print_matrices:
        print(f"\n   ИНФОРМАЦИЯ ОБ ЭЛЕМЕНТАХ (первые 5):")
        print("   № элемента | Узлы (индексы) |   Площадь   ")
        print("   -----------|----------------|-------------")
        for i in range(min(5, n_elements)):
            elem = elements[i]
            p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            print(f"   {i+1:11d} | {elem[0]:2d}, {elem[1]:2d}, {elem[2]:2d}     | {area:10.3f}")
        if n_elements > 5:
            print(f"   ... и ещё {n_elements - 5} элементов")

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
        print(f"   ... и ещё {len(areas)-5} элементов")

    # --- Шаг 4: Матрица жёсткости ---
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
        print(f"\n   ЛОКАЛЬНЫЕ МАТРИЦЫ ЖЕСТКОСТИ (первые 2 элемента):")
        for idx in range(min(2, n_elements)):
            elem = elements[idx]
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
            print(f"\n   Элемент {idx+1} (узлы {elem[0]+1}, {elem[1]+1}, {elem[2]+1}):")
            for i in range(3):
                row_str = " ".join([f"{K_e[i, j]:10.6f}" for j in range(3)])
                print(f"      [{row_str}]")
            for i_local, i_global in enumerate(elem):
                for j_local, j_global in enumerate(elem):
                    K_global[i_global, j_global] += K_e[i_local, j_local]
        # Остальные элементы без вывода
        for idx in range(2, n_elements):
            elem = elements[idx]
            elem_nodes = nodes[elem]
            K_e = element_stiffness_matrix(elem_nodes, areas[idx])
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
            print(f"   ... и ещё {n_nodes-8} строк и столбцов")

    # --- Шаг 5: Вектор нагрузки ---
    F_global = np.zeros(n_nodes)
    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")

    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ ВЕКТОРЫ НАГРУЗКИ (первые 2 элемента):")
        for idx in range(min(2, n_elements)):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            print(f"   Элемент {idx+1}: [{f_e[0]:.6f}, {f_e[1]:.6f}, {f_e[2]:.6f}]")
            elem = elements[idx]
            for i_local, i_global in enumerate(elem):
                F_global[i_global] += f_e[i_local]
        for idx in range(2, n_elements):
            f_e = (G_theta * areas[idx] / 3) * np.array([1.0, 1.0, 1.0])
            elem = elements[idx]
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
            print(f"   ... и ещё {len(F_global)-10} элементов")

    # --- Шаг 6: Граничные условия ---
    # Все узлы на внешней границе L-области имеют φ = 0
    def is_boundary_node(x, y):
        # Левая граница
        if np.isclose(x, 0.0) and 0 <= y <= 3:
            return True
        # Правая граница горизонтальной части
        if np.isclose(x, 3.0) and 0 <= y <= 1:
            return True
        # Верхняя граница вертикальной части
        if np.isclose(y, 3.0) and 0 <= x <= 1:
            return True
        # Нижняя граница всей фигуры
        if np.isclose(y, 0.0) and 0 <= x <= 3:
            return True
        # Внутренний угол: правая граница вертикальной части (x=1, y∈[1,3])
        if np.isclose(x, 1.0) and 1 <= y <= 3:
            return True
        # Внутренний угол: верхняя граница горизонтальной части (y=1, x∈[1,3])
        if np.isclose(y, 1.0) and 1 <= x <= 3:
            return True
        return False

    boundary_nodes_indices = []
    for i, (x, y) in enumerate(nodes):
        if is_boundary_node(x, y):
            boundary_nodes_indices.append(i)

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
                print(f"   ... и ещё {len(Phi)-10} элементов")
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
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.7)

    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=15, zorder=5)
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
    ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.3, alpha=0.5)
    ax2.add_patch(PathPatch(boundary_path, facecolor='none', edgecolor='black', lw=2))
    ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10, zorder=5)
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
    ax3.set_xlabel('Столбец')
    ax3.set_ylabel('Строка')

    im4 = ax4.imshow(np.abs(K_bc), cmap='hot', aspect='auto')
    plt.colorbar(im4, ax=ax4, label='|K|')
    ax4.set_title('Матрица жесткости K (после ГУ)', fontsize=14)
    ax4.set_xlabel('Столбец')
    ax4.set_ylabel('Строка')

    plt.tight_layout()
    plt.show()

    # --- АНАЛИЗ МАТРИЦ ---
    print("\n" + "="*60)
    print("АНАЛИЗ МАТРИЦ")
    print("="*60)
    try:
        cond_orig = np.linalg.cond(K_original)
    except:
        cond_orig = float('inf')
    try:
        cond_bc = np.linalg.cond(K_bc)
    except:
        cond_bc = float('inf')
    print(f"Свойства матрицы жесткости (до граничных условий):")
    print(f"   Размер: {K_original.shape}")
    print(f"   Норма: {np.linalg.norm(K_original):.6f}")
    print(f"   Определитель: {np.linalg.det(K_original):.6e}")
    print(f"   Число обусловленности: {cond_orig:.3e}")
    print(f"\nСвойства матрицы жесткости (после граничных условий):")
    print(f"   Размер: {K_bc.shape}")
    print(f"   Норма: {np.linalg.norm(K_bc):.6f}")
    print(f"   Определитель: {np.linalg.det(K_bc):.6e}")
    print(f"   Число обусловленности: {cond_bc:.3e}")

    # --- ИТОГОВАЯ ТАБЛИЦА ---
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"Область: L-образная [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {len(nodes)}")
    print(f"Количество элементов: {len(elements)}")
    print(f"\nЗНАЧЕНИЯ ФУНКЦИИ НАПРЯЖЕНИЯ В УЗЛАХ (первые 10):")
    print("№ узла |     x     |     y     |    φ(x,y)    ")
    print("-------|-----------|-----------|--------------")
    for i in range(min(10, len(nodes))):
        print(f"{i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f} | {Phi[i]:12.6f}")
    if len(nodes) > 10:
        print(f"... и ещё {len(nodes) - 10} узлов")