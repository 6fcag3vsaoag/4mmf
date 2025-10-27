import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def solve_torsion_ellipse_by_fem(G_theta, a=4.0, b=3.0, x_max=1.0, num_nodes_ellipse=16, num_nodes_straight=8, print_matrices=True):
    """
    Решает задачу кручения стержня эллиптического сечения методом конечных элементов.
    Область D: эллипс (x/a)^2 + (y/b)^2 <= 1, x ∈ [-a, x_max], где x_max = 1
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ ЭЛЛИПСА МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print(f"Эллипс: a = {a}, b = {b}")
    print(f"Закрашенная область: x ∈ [{-a}, {x_max}]")
    print("=" * 80)

    # --- Шаг 1: Генерация узлов сетки ---
    print(f"\n1. ГЕНЕРАЦИЯ УЗЛОВ СЕТКИ:")
    
    # Узлы на границе эллипса (левая часть от -a до x_max)
    theta_ellipse = np.linspace(np.pi/2, 3*np.pi/2, num_nodes_ellipse)
    ellipse_nodes = []
    for angle in theta_ellipse:
        x = a * np.cos(angle)
        y = b * np.sin(angle)
        if x <= x_max:  # Только левая часть эллипса до x_max
            ellipse_nodes.append([x, y])
    
    n_ellipse = len(ellipse_nodes)
    print(f"   Узлов на эллипсе: {n_ellipse}")

    # Узлы на прямой x = x_max
    y_straight = np.linspace(-b, b, num_nodes_straight)
    straight_nodes = []
    for y in y_straight:
        if abs(y) < b:  # Исключаем точки на границе
            straight_nodes.append([x_max, y])
    
    n_straight = len(straight_nodes)
    print(f"   Узлов на прямой x={x_max}: {n_straight}")

    # Внутренние узлы для триангуляции
    interior_nodes = [
        [-(a + x_max)/3, 0.0],      # Центральный узел в левой части
        [x_max - (a + x_max)/3, 0.0], # Центральный узел в правой части
        [-a*0.7, b*0.5],            # Дополнительные узлы
        [-a*0.7, -b*0.5],
        [x_max*0.3, b*0.7],
        [x_max*0.3, -b*0.7]
    ]
    n_interior = len(interior_nodes)
    print(f"   Внутренних узлов: {n_interior}")

    # Объединяем все узлы
    all_nodes = ellipse_nodes + straight_nodes + interior_nodes
    nodes = np.array(all_nodes)
    n_nodes = len(nodes)
    print(f"   Всего узлов: {n_nodes}")

    # Вывод координат узлов
    if print_matrices:
        print(f"\n   КООРДИНАТЫ УЗЛОВ:")
        print("   № узла |     x     |     y    ")
        print("   -------|-----------|-----------")
        for i, (x, y) in enumerate(nodes):
            print(f"   {i+1:6d} | {x:9.3f} | {y:9.3f}")

    # --- Шаг 2: Триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    
    elements = []
    
    # Индексы узлов по группам
    ellipse_indices = list(range(n_ellipse))
    straight_indices = list(range(n_ellipse, n_ellipse + n_straight))
    interior_indices = list(range(n_ellipse + n_straight, n_ellipse + n_straight + n_interior))
    
    # Функция для проверки, находится ли точка внутри области
    def is_inside_ellipse_segment(x, y):
        """Проверяет, находится ли точка внутри эллипса и левее x_max"""
        ellipse_condition = (x/a)**2 + (y/b)**2 <= 1.01  # Небольшой запас
        x_condition = x <= x_max + 0.01
        return ellipse_condition and x_condition
    
    # Создаем триангуляцию (упрощенный алгоритм)
    # Соединяем соседние узлы эллипса с ближайшими внутренними узлами
    for i in range(n_ellipse):
        next_i = (i + 1) % n_ellipse
        # Основной треугольник с центральным узлом
        if is_inside_ellipse_segment(*nodes[interior_indices[0]]):
            elements.append([ellipse_indices[i], ellipse_indices[next_i], interior_indices[0]])
    
    # Соединяем узлы прямой
    for i in range(n_straight - 1):
        if is_inside_ellipse_segment(*nodes[interior_indices[1]]):
            elements.append([straight_indices[i], straight_indices[i + 1], interior_indices[1]])
    
    # Соединяем эллипс и прямую
    if n_ellipse > 0 and n_straight > 0:
        # Верхняя связь
        elements.append([ellipse_indices[0], straight_indices[0], interior_indices[0]])
        elements.append([straight_indices[0], interior_indices[0], interior_indices[1]])
        
        # Нижняя связь  
        elements.append([ellipse_indices[-1], straight_indices[-1], interior_indices[0]])
        elements.append([straight_indices[-1], interior_indices[0], interior_indices[1]])
    
    # Дополнительные треугольники для лучшего покрытия
    for i in range(2, n_interior):
        # Соединяем внутренние узлы с ближайшими граничными
        for j in range(min(3, n_ellipse)):
            if is_inside_ellipse_segment(*nodes[interior_indices[i]]):
                elements.append([ellipse_indices[j], interior_indices[0], interior_indices[i]])
    
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
    # Все узлы на границе (эллипс и прямая) имеют φ = 0
    boundary_nodes = ellipse_indices + straight_indices

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
G_theta = 3.6  # Параметр кручения
a, b = 4.0, 3.0  # Полуоси эллипса
x_max = 1.0     # Правая граница закрашенной области

# Решение задачи
result = solve_torsion_ellipse_by_fem(G_theta, a, b, x_max, print_matrices=True)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc = result

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка конечных элементов
    theta_full = np.linspace(0, 2*np.pi, 100)
    x_ellipse_full = a * np.cos(theta_full)
    y_ellipse_full = b * np.sin(theta_full)
    ax1.plot(x_ellipse_full, y_ellipse_full, 'k--', alpha=0.5, label='Полный эллипс')
    
    # Закрашенная область
    theta_segment = np.linspace(np.pi/2, 3*np.pi/2, 100)
    x_segment = a * np.cos(theta_segment)
    y_segment = b * np.sin(theta_segment)
    mask = x_segment <= x_max
    x_filled = np.concatenate([x_segment[mask], [x_max, x_segment[mask][0]]])
    y_filled = np.concatenate([y_segment[mask], [y_segment[mask][-1], y_segment[mask][0]]])
    ax1.fill(x_filled, y_filled, 'lightblue', alpha=0.3, label='Закрашенная область')

    # Рисуем элементы
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)

    # Рисуем узлы
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=30, zorder=5)

    ax1.set_title('Сетка конечных элементов эллипса', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.set_xlim(-a-0.5, a+0.5)
    ax1.set_ylim(-b-0.5, b+0.5)

    # График 2: Распределение функции напряжения
    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    contour = ax2.tricontourf(tri, Phi, levels=20, cmap='viridis')
    ax2.tricontour(tri, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    ax2.plot(x_ellipse_full, y_ellipse_full, 'k-', linewidth=2, label='Граница эллипса')
    ax2.axvline(x=x_max, color='r', linestyle='--', linewidth=2, label=f'x = {x_max}')
    
    ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)
    
    plt.colorbar(contour, ax=ax2, label='φ(x, y)')
    ax2.set_title('Распределение функции напряжения φ', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    ax2.set_xlim(-a-0.5, a+0.5)
    ax2.set_ylim(-b-0.5, b+0.5)

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
    print(f"Эллипс: a = {a}, b = {b}")
    print(f"Закрашенная область: x ∈ [{-a}, {x_max}]")
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
def solve_torsion_ellipse_quick(G_theta, a=4.0, b=3.0, x_max=1.0):
    """Краткое решение без вывода матриц"""
    return solve_torsion_ellipse_by_fem(G_theta, a, b, x_max, print_matrices=False)

# Пример быстрого расчета
# nodes, elements, Phi, T, _, _, _, _ = solve_torsion_ellipse_quick(2.0)