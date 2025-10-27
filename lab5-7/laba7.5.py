import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from typing import Tuple, List, Dict, Any

class LShapeMeshGenerator:
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.boundaries = None
        self.vertical_part_elements = None
        self.horizontal_part_elements = None
        
    def generate_mesh(self, h: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация согласованной сетки для L-образной области
        L-образная область: [0,1]×[0,3] ∪ [1,3]×[0,1]
        """
        print("Генерация узлов сетки...")
        
        # Создаем узлы для всей L-образной области
        nodes_list = []
        node_coord_to_index = {}
        
        # Вертикальная часть: [0, 1] x [0, 3]
        print("  - Вертикальная часть: [0, 1] x [0, 3]")
        x_vert = np.arange(0, 1 + h/2, h)
        y_vert = np.arange(0, 3 + h/2, h)
        
        for j, y in enumerate(y_vert):
            for i, x in enumerate(x_vert):
                coord_key = (round(x, 6), round(y, 6))
                if coord_key not in node_coord_to_index:
                    nodes_list.append([x, y])
                    node_coord_to_index[coord_key] = len(nodes_list) - 1
        
        # Горизонтальная часть: [1, 3] x [0, 1]  
        print("  - Горизонтальная часть: [1, 3] x [0, 1]")
        x_horiz = np.arange(1, 3 + h/2, h)
        y_horiz = np.arange(0, 1 + h/2, h)
        
        for j, y in enumerate(y_horiz):
            for i, x in enumerate(x_horiz):
                # Пропускаем узел (1,0) т.к. он уже добавлен
                if abs(x - 1) < 1e-10 and abs(y) < 1e-10:
                    continue
                coord_key = (round(x, 6), round(y, 6))
                if coord_key not in node_coord_to_index:
                    nodes_list.append([x, y])
                    node_coord_to_index[coord_key] = len(nodes_list) - 1
        
        self.nodes = np.array(nodes_list)
        
        # Создаем элементы
        print("Триангуляция области...")
        elements, vert_elements, horiz_elements = self._create_elements(
            node_coord_to_index, x_vert, y_vert, x_horiz, y_horiz, h)
        
        self.elements = np.array(elements)
        self.vertical_part_elements = vert_elements
        self.horizontal_part_elements = horiz_elements
        
        self._identify_boundaries()
        
        print("✓ Сетка успешно создана")
        return self.nodes, self.elements
    
    def _create_elements(self, node_dict, x_vert, y_vert, x_horiz, y_horiz, h):
        """Создание треугольных элементов для обеих частей"""
        elements = []
        vertical_elements = []
        horizontal_elements = []
        
        # Триангуляция вертикальной части
        print("  - Триангуляция вертикальной части")
        for j in range(len(y_vert) - 1):
            for i in range(len(x_vert) - 1):
                # Координаты узлов прямоугольника
                coords = [
                    (x_vert[i], y_vert[j]),
                    (x_vert[i+1], y_vert[j]), 
                    (x_vert[i], y_vert[j+1]),
                    (x_vert[i+1], y_vert[j+1])
                ]
                
                # Индексы узлов
                indices = [node_dict[(round(x,6), round(y,6))] for x, y in coords]
                
                # Два треугольника
                elem1 = [indices[0], indices[1], indices[3]]
                elem2 = [indices[0], indices[3], indices[2]]
                
                elements.extend([elem1, elem2])
                vertical_elements.extend([elem1, elem2])
        
        # Триангуляция горизонтальной части
        print("  - Триангуляция горизонтальной части") 
        for j in range(len(y_horiz) - 1):
            for i in range(len(x_horiz) - 1):
                # Координаты узлов прямоугольника
                coords = [
                    (x_horiz[i], y_horiz[j]),
                    (x_horiz[i+1], y_horiz[j]),
                    (x_horiz[i], y_horiz[j+1]), 
                    (x_horiz[i+1], y_horiz[j+1])
                ]
                
                # Индексы узлов
                indices = [node_dict[(round(x,6), round(y,6))] for x, y in coords]
                
                # Два треугольника
                elem1 = [indices[0], indices[1], indices[3]]
                elem2 = [indices[0], indices[3], indices[2]]
                
                elements.extend([elem1, elem2])
                horizontal_elements.extend([elem1, elem2])
        
        return elements, vertical_elements, horizontal_elements
    
    def _identify_boundaries(self):
        """Идентификация граничных узлов"""
        boundaries = {
            'left': [],    # x = 0
            'right': [],   # x = 3  
            'bottom': [],  # y = 0
            'top': [],     # y = 3
            'inner_corner': []  # внутренний угол
        }
        
        for i, (x, y) in enumerate(self.nodes):
            if abs(x) < 1e-10:
                boundaries['left'].append(i)
            elif abs(x - 3) < 1e-10:
                boundaries['right'].append(i) 
            elif abs(y) < 1e-10:
                boundaries['bottom'].append(i)
            elif abs(y - 3) < 1e-10:
                boundaries['top'].append(i)
            elif (abs(x - 1) < 1e-10 and y > 1) or (abs(y - 1) < 1e-10 and x > 1):
                boundaries['inner_corner'].append(i)
        
        self.boundaries = boundaries
    
    def calculate_element_area(self, element):
        """Вычисление площади одного элемента"""
        p1, p2, p3 = self.nodes[element]
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def get_mesh_statistics(self):
        """Получение статистики сетки"""
        areas = [self.calculate_element_area(elem) for elem in self.elements]
        vert_areas = [self.calculate_element_area(elem) for elem in self.vertical_part_elements]
        horiz_areas = [self.calculate_element_area(elem) for elem in self.horizontal_part_elements]
        
        return {
            'total_nodes': len(self.nodes),
            'total_elements': len(self.elements),
            'vertical_elements': len(self.vertical_part_elements),
            'horizontal_elements': len(self.horizontal_part_elements),
            'min_area': min(areas),
            'max_area': max(areas), 
            'mean_area': sum(areas) / len(areas),
            'total_area': sum(areas),
            'vertical_area': sum(vert_areas),
            'horizontal_area': sum(horiz_areas),
            'boundary_nodes': sum(len(nodes) for nodes in self.boundaries.values())
        }

def solve_torsion_lshape_by_fem(G_theta, print_matrices=True):
    """
    Решает задачу кручения стержня L-образного сечения методом конечных элементов.
    Область D: [0,1]×[0,3] ∪ [1,3]×[0,1]
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ КРУЧЕНИЯ L-ОБРАЗНОГО СЕЧЕНИЯ МЕТОДОМ КОНЕЧНЫХ ЭЛЕМЕНТОВ")
    print(f"Параметр Gθ = {G_theta}")
    print("Область: [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print("=" * 80)

    # --- Шаг 1: Генерация сетки ---
    mesh_gen = LShapeMeshGenerator()
    nodes, elements = mesh_gen.generate_mesh(h=0.5)
    
    n_nodes = len(nodes)
    n_elements = len(elements)
    
    print(f"\n1. ИНФОРМАЦИЯ О СЕТКЕ:")
    stats = mesh_gen.get_mesh_statistics()
    print(f"   Всего узлов: {stats['total_nodes']}")
    print(f"   Всего элементов: {stats['total_elements']}")
    print(f"   Элементов в вертикальной части: {stats['vertical_elements']}") 
    print(f"   Элементов в горизонтальной части: {stats['horizontal_elements']}")

    # Вывод координат узлов (первые 10)
    print(f"\n   КООРДИНАТЫ УЗЛОВ (первые 10):")
    print("   № узла |     x     |     y    ")
    print("   -------|-----------|-----------")
    for i, (x, y) in enumerate(nodes[:10]):
        print(f"   {i+1:6d} | {x:9.3f} | {y:9.3f}")
    if n_nodes > 10:
        print(f"   ... и еще {n_nodes-10} узлов")

    # --- Шаг 2: Триангуляция области ---
    print(f"\n2. ТРИАНГУЛЯЦИЯ ОБЛАСТИ:")
    print(f"   Создано элементов: {n_elements}")

    # Вычисляем площади ВСЕХ элементов
    areas = [mesh_gen.calculate_element_area(elem) for elem in elements]

    # Вывод информации об элементах (первые 10)
    print(f"\n   ИНФОРМАЦИЯ ОБ ЭЛЕМЕНТАХ (первые 10):")
    print("   № элемента | Узлы (индексы) |   Площадь   ")
    print("   -----------|----------------|-------------")
    
    for i, elem in enumerate(elements[:10]):
        print(f"   {i+1:11d} | {elem[0]:2d}, {elem[1]:2d}, {elem[2]:2d}     | {areas[i]:10.6f}")
    if n_elements > 10:
        print(f"   ... и еще {n_elements-10} элементов")

    # Вывод площадей элементов (первые 10)
    print(f"\n   ПЛОЩАДИ ЭЛЕМЕНТОВ (первые 10):")
    for i in range(min(10, len(areas))):
        print(f"   S{i+1} = {areas[i]:.6f}")
    if len(areas) > 10:
        print(f"   ... и еще {len(areas)-10} элементов")

    # --- Шаг 3: Построение матриц жесткости ---
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
    
    # Вывод локальных матриц жесткости (первые 3)
    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ МАТРИЦЫ ЖЕСТКОСТИ (первые 3 элемента):")
        for idx, elem in enumerate(elements[:3]):
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
    print(f"\n   ГЛОБАЛЬНАЯ МАТРИЦА ЖЕСТКОСТИ K (размер {n_nodes}×{n_nodes}):")
    print("   (показаны первые 6 строк и столбцов)")
    for i in range(min(6, n_nodes)):
        row_str = " ".join([f"{K_global[i, j]:8.3f}" for j in range(min(6, n_nodes))])
        print(f"   [{row_str}]")
    if n_nodes > 6:
        print(f"   ... и еще {n_nodes-6} строк и столбцов")

    # --- Шаг 4: Построение вектора нагрузки ---
    F_global = np.zeros(n_nodes)

    print(f"\n4. ПОСТРОЕНИЕ ВЕКТОРА НАГРУЗКИ:")
    
    # Вывод локальных векторов нагрузки (первые 3)
    if print_matrices:
        print(f"\n   ЛОКАЛЬНЫЕ ВЕКТОРЫ НАГРУЗКИ (первые 3 элемента):")
        for idx, elem in enumerate(elements[:3]):
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

    # Вывод глобального вектора нагрузки (первые 8)
    print(f"\n   ГЛОБАЛЬНЫЙ ВЕКТОР НАГРУЗКИ F (первые 8 элементов):")
    for i in range(min(8, len(F_global))):
        print(f"   F[{i+1}] = {F_global[i]:.6f}")
    if len(F_global) > 8:
        print(f"   ... и еще {len(F_global)-8} элементов")

    # --- Шаг 5: Граничные условия ---
    # Только внешние границы имеют φ = 0, внутренний угол - свободный
    boundary_nodes = (mesh_gen.boundaries['left'] + mesh_gen.boundaries['right'] + 
                     mesh_gen.boundaries['bottom'] + mesh_gen.boundaries['top'])
    # Убираем внутренний угол из граничных условий

    print(f"\n5. ГРАНИЧНЫЕ УСЛОВИЯ:")
    print(f"   Граничных узлов: {len(boundary_nodes)}")
    print(f"   Узлов на внутреннем углу: {len(mesh_gen.boundaries['inner_corner'])}")

    # Сохраняем оригинальные матрицы для вывода
    K_original = K_global.copy()
    F_original = F_global.copy()

    # ПРАВИЛЬНОЕ применение граничных условий
    # Создаем маску для внутренних узлов (включая внутренний угол)
    internal_nodes = [i for i in range(n_nodes) if i not in boundary_nodes]
    
    if not internal_nodes:
        print("   Ошибка: Нет внутренних узлов для решения")
        return None, None, None, None
    
    print(f"   Внутренних узлов для решения: {len(internal_nodes)}")
    
    # Выделяем подматрицу для внутренних узлов
    K_internal = K_global[np.ix_(internal_nodes, internal_nodes)]
    F_internal = F_global[internal_nodes]

    # --- Шаг 6: Решение системы ---
    try:
        # Решаем систему только для внутренних узлов
        Phi_internal = np.linalg.solve(K_internal, F_internal)
        
        # Создаем полный вектор решения
        Phi = np.zeros(n_nodes)
        Phi[internal_nodes] = Phi_internal
        # Граничные узлы уже равны 0 (по умолчанию в zeros)
        
        print(f"\n6. РЕШЕНИЕ СИСТЕМЫ:")
        print(f"   Система успешно решена")
        
        # Вывод решения (первые 10)
        print(f"\n   РЕШЕНИЕ СИСТЕМЫ (первые 10 значений φ):")
        for i in range(min(10, len(Phi))):
            print(f"   φ[{i+1}] = {Phi[i]:.6f}")
        if len(Phi) > 10:
            print(f"   ... и еще {len(Phi)-10} элементов")
                
    except np.linalg.LinAlgError as e:
        print(f"   Ошибка при решении системы: {e}")
        # Пробуем использовать псевдообратную матрицу
        try:
            print("   Попытка использовать псевдообратную матрицу...")
            Phi_internal = np.linalg.pinv(K_internal) @ F_internal
            Phi = np.zeros(n_nodes)
            Phi[internal_nodes] = Phi_internal
            print("   Система решена с использованием псевдообратной матрицы")
        except:
            print("   Ошибка: Не удалось решить систему")
            return None, None, None, None

    # --- Шаг 7: Вычисление крутящего момента ---
    integral_phi = 0.0
    for idx, elem in enumerate(elements):
        avg_phi = np.mean([Phi[elem[0]], Phi[elem[1]], Phi[elem[2]]])
        integral_phi += areas[idx] * avg_phi

    T = 2 * G_theta * integral_phi

    print(f"\n7. КРУТЯЩИЙ МОМЕНТ:")
    print(f"   T = 2 * Gθ * ∫∫φ dxdy = {T:.6f}")

    return nodes, elements, Phi, T, K_original, F_original, K_global, F_global, mesh_gen

# --- ОСНОВНОЕ ВЫПОЛНЕНИЕ ---
G_theta = 3.6  # Параметр кручения

# Решение задачи
result = solve_torsion_lshape_by_fem(G_theta, print_matrices=True)

if result[2] is not None:
    nodes, elements, Phi, T, K_original, F_original, K_bc, F_bc, mesh_gen = result

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Сетка конечных элементов
    # Рисуем L-образную область как единое целое
    lshape_outline = np.array([
        [0, 0], [0, 3], [1, 3], [1, 1], [3, 1], [3, 0], [1, 0], [0, 0]
    ])
    
    ax1.plot(lshape_outline[:, 0], lshape_outline[:, 1], 'k-', linewidth=2, label='L-образная область')
    
    # Рисуем элементы
    for elem in elements:
        x_coords = [nodes[elem[i]][0] for i in range(3)] + [nodes[elem[0]][0]]
        y_coords = [nodes[elem[i]][1] for i in range(3)] + [nodes[elem[0]][1]]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.7)

    # Рисуем узлы
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='red', s=20, zorder=5)

    ax1.set_title('Сетка конечных элементов L-образной области', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 3.5)

    # График 2: Распределение функции напряжения
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    contour = ax2.tricontourf(triangulation, Phi, levels=20, cmap='viridis')
    ax2.tricontour(triangulation, Phi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    # Рисуем границу области как единое целое
    ax2.plot(lshape_outline[:, 0], lshape_outline[:, 1], 'k-', linewidth=2)
    
    ax2.scatter(nodes[:, 0], nodes[:, 1], color='red', s=5, zorder=5, alpha=0.6)
    
    plt.colorbar(contour, ax=ax2, label='φ(x, y)')
    ax2.set_title('Распределение функции напряжения φ', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
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
    print(f"Область: [0,1]×[0,3] ∪ [1,3]×[0,1]")
    print(f"Параметр кручения: Gθ = {G_theta}")
    print(f"Крутящий момент: T = {T:.6f}")
    
    stats = mesh_gen.get_mesh_statistics()
    print(f"Количество узлов: {stats['total_nodes']}")
    print(f"Количество элементов: {stats['total_elements']}")
    print(f"Элементов в вертикальной части: {stats['vertical_elements']}")
    print(f"Элементов в горизонтальной части: {stats['horizontal_elements']}")
    print(f"Общая площадь: {stats['total_area']:.6f}")
    
    print(f"\nЗНАЧЕНИЯ ФУНКЦИИ НАПРЯЖЕНИЯ В УЗЛАХ (первые 15):")
    print("№ узла |     x     |     y     |    φ(x,y)    ")
    print("-------|-----------|-----------|--------------")
    for i in range(min(15, len(nodes))):
        print(f"{i+1:6d} | {nodes[i,0]:9.3f} | {nodes[i,1]:9.3f} | {Phi[i]:12.6f}")
    if len(nodes) > 15:
        print(f"... и еще {len(nodes)-15} узлов")

# --- ФУНКЦИЯ ДЛЯ КРАТКОГО ВЫВОДА (без матриц) ---
def solve_torsion_lshape_quick(G_theta):
    """Краткое решение без вывода матриц"""
    return solve_torsion_lshape_by_fem(G_theta, print_matrices=False)

# Пример быстрого расчета
# nodes, elements, Phi, T, _, _, _, _, _ = solve_torsion_lshape_quick(2.0)