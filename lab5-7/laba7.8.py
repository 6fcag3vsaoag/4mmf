import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def solve_torsion_L_shape_by_fem(G_theta=3.6, print_matrices=True):
    """
    Решает задачу кручения стержня L-образного сечения методом конечных элементов.
    Область: [0,1]×[0,3] ∪ [1,3]×[0,1]
    Сетка строится с общими узлами на пересечении — гарантирует связность.
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СЕЧЕНИЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("Область: [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов с общими точками на x=1, y∈[0,1] ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")

    # Общий шаг по y для перекрытия [0,1]
    ny_common = 12  # должно делить и вертикальную, и горизонтальную части
    y_common = np.linspace(0, 1, ny_common + 1)

    # Вертикальная часть: [0,1] × [0,3]
    nx_vert = 10
    ny_vert_upper = 24  # для [1,3] по y
    y_vert_upper = np.linspace(1, 3, ny_vert_upper + 1)
    y_vert = np.concatenate([y_common, y_vert_upper[1:]])  # избегаем дубляжа y=1
    x_vert = np.linspace(0, 1, nx_vert + 1)
    
    nodes_vert = []
    for x in x_vert:
        for y in y_vert:
            nodes_vert.append([x, y])
    nodes_vert = np.array(nodes_vert)

    # Горизонтальная часть: [1,3] × [0,1] — используем тот же y_common
    nx_horiz = 20
    x_horiz = np.linspace(1, 3, nx_horiz + 1)
    nodes_horiz = []
    for x in x_horiz:
        for y in y_common:
            nodes_horiz.append([x, y])
    nodes_horiz = np.array(nodes_horiz)

    # Объединяем
    all_nodes = np.vstack([nodes_vert, nodes_horiz])
    
    # Удаляем дубликаты (на x=1, y∈[0,1])
    rounded = np.round(all_nodes, decimals=10)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    nodes = all_nodes[np.sort(unique_indices)]
    n_nodes = len(nodes)

    print(f"   Всего узлов: {n_nodes}")

    if print_matrices:
        print(f"\n   КООРДИНАТЫ УЗЛОВ (первые 15):")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i in range(min(15, n_nodes)):
            print(f"   {i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f}")
        if n_nodes > 15:
            print(f"   ... и ещё {n_nodes - 15} узлов")

    # --- Шаг 2: Создание элементов ---
    print(f"\n2. ГЕНЕРАЦИЯ ЭЛЕМЕНТОВ:")

    # Создаём карту координат -> индекс
    node_map = {}
    for i, (x, y) in enumerate(nodes):
        key = (round(x, 10), round(y, 10))
        node_map[key] = i

    elements = []

    # Элементы вертикальной части
    for i in range(nx_vert):
        for j in range(len(y_vert) - 1):
            x0, x1 = x_vert[i], x_vert[i+1]
            y0, y1 = y_vert[j], y_vert[j+1]
            try:
                n0 = node_map[(round(x0,10), round(y0,10))]
                n1 = node_map[(round(x1,10), round(y0,10))]
                n2 = node_map[(round(x1,10), round(y1,10))]
                n3 = node_map[(round(x0,10), round(y1,10))]
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
            except KeyError:
                continue

    # Элементы горизонтальной части
    for i in range(nx_horiz):
        for j in range(len(y_common) - 1):
            x0, x1 = x_horiz[i], x_horiz[i+1]
            y0, y1 = y_common[j], y_common[j+1]
            try:
                n0 = node_map[(round(x0,10), round(y0,10))]
                n1 = node_map[(round(x1,10), round(y0,10))]
                n2 = node_map[(round(x1,10), round(y1,10))]
                n3 = node_map[(round(x0,10), round(y1,10))]
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
            except KeyError:
                continue

    n_elements = len(elements)
    print(f"   Создано элементов: {n_elements}")

    # --- Шаг 3: Площади ---
    def triangle_area(p1, p2, p3):
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    areas = [triangle_area(nodes[e[0]], nodes[e[1]], nodes[e[2]]) for e in elements]

    # --- Шаг 4: Матрица жёсткости ---
    def element_stiffness_matrix(elem_nodes, area):
        xi, yi = elem_nodes[0]; xj, yj = elem_nodes[1]; xk, yk = elem_nodes[2]
        bi, bj, bk = yj - yk, yk - yi, yi - yj
        ci, cj, ck = xk - xj, xi - xk, xj - xi
        B = np.array([[bi, bj, bk], [ci, cj, ck]]) / (2 * area)
        return area * (B.T @ B)

    K_global = np.zeros((n_nodes, n_nodes))
    for elem, area in zip(elements, areas):
        K_e = element_stiffness_matrix(nodes[elem], area)
        for i_local, i_global in enumerate(elem):
            for j_local, j_global in enumerate(elem):
                K_global[i_global, j_global] += K_e[i_local, j_local]

    # --- Шаг 5: Вектор нагрузки ---
    F_global = np.zeros(n_nodes)
    for elem, area in zip(elements, areas):
        f_e = (G_theta * area / 3) * np.array([1.0, 1.0, 1.0])
        for i_local, i_global in enumerate(elem):
            F_global[i_global] += f_e[i_local]

    # --- Шаг 6: Граничные условия ---
    boundary_nodes = []
    tol = 1e-10
    for i, (x, y) in enumerate(nodes):
        if (abs(x) < tol and 0 <= y <= 3) or \
           (abs(y - 3) < tol and 0 <= x <= 1) or \
           (abs(x - 1) < tol and 1 <= y <= 3) or \
           (abs(y - 1) < tol and 1 <= x <= 3) or \
           (abs(x - 3) < tol and 0 <= y <= 1) or \
           (abs(y) < tol and 0 <= x <= 3):
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

    # --- Шаг 7: Решение ---
    try:
        Phi = np.linalg.solve(K_global, F_global)
        print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
        print(f"   Система успешно решена")
    except np.linalg.LinAlgError as e:
        print(f"   Ошибка: {e}")
        return None, None, None, None, None, None, None, None

    # --- Шаг 8: Крутящий момент ---
    integral_phi = sum(np.mean([Phi[e[0]], Phi[e[1]], Phi[e[2]]]) * area 
                      for e, area in zip(elements, areas))
    T = 2 * G_theta * integral_phi
    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = {T:.6f}")

    # --- Визуализация и вывод ---
    if print_matrices:
        print(f"\n   ГЛОБАЛЬНАЯ МАТРИЦА ЖЕСТКОСТИ K (размер {n_nodes}×{n_nodes}):")
        for i in range(min(6, n_nodes)):
            row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(6, n_nodes))])
            print(f"   [{row_str}]")

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Сетка
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.7)
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=15)
    # Граница L
    L_x = [0, 0, 1, 1, 3, 3, 1, 1, 0]
    L_y = [0, 3, 3, 1, 1, 0, 0, 1, 1]
    ax1.plot(L_x, L_y, 'k-', linewidth=2)
    ax1.set_title('Сетка конечных элементов')
    ax1.axis('equal')
    ax1.set_xlim(-0.2, 3.2)
    ax1.set_ylim(-0.2, 3.2)

    # Распределение φ
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
    ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.3, alpha=0.5)
    ax2.plot(L_x, L_y, 'k-', linewidth=2)
    plt.colorbar(contour, ax=ax2, label='φ(x, y)')
    ax2.set_title('Распределение функции напряжения φ')
    ax2.axis('equal')
    ax2.set_xlim(-0.2, 3.2)
    ax2.set_ylim(-0.2, 3.2)

    plt.tight_layout()
    plt.show()

    # Итоговая таблица
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Крутящий момент: T = {T:.6f}")
    print(f"Количество узлов: {n_nodes}")
    print(f"Количество элементов: {n_elements}")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global

# --- ЗАПУСК ---
G_theta = 3.6
result = solve_torsion_L_shape_by_fem(G_theta, print_matrices=True)