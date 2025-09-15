import math

# Безопасный контекст для eval
SAFE_DICT = {
    'x': None,
    'y': None,
    'math': math,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'exp': math.exp,
    'log': math.log,
    'sqrt': math.sqrt,
    'pi': math.pi,
    'e': math.e,
}

def safe_eval(expr, x_val, y_val):
    """Безопасная оценка выражения с подстановкой x и y"""
    SAFE_DICT['x'] = x_val
    SAFE_DICT['y'] = y_val
    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_DICT)
        return result
    except Exception as e:
        raise ValueError(f"Ошибка в выражении: {e}")

def runge_kutta_4(f_expr, x0, y0, h, n_steps, decimals):
    """
    Метод Рунге–Кутты 4-го порядка для получения начальных значений
    """
    x = x0
    y = y0
    steps = []

    for i in range(n_steps):
        try:
            # g1 = h * f(x_k, y_k)
            g1 = h * safe_eval(f_expr, x, y)

            # g2 = h * f(x_k + h/2, y_k + g1/2)
            g2 = h * safe_eval(f_expr, x + h/2, y + g1/2)

            # g3 = h * f(x_k + h/2, y_k + g2/2)
            g3 = h * safe_eval(f_expr, x + h/2, y + g2/2)

            # g4 = h * f(x_k + h, y_k + g3)
            g4 = h * safe_eval(f_expr, x + h, y + g3)

            # y_{k+1} = y_k + (g1 + 2*g2 + 2*g3 + g4)/6
            y_new = y + (g1 + 2*g2 + 2*g3 + g4) / 6

            x += h
            y = y_new

            f_val = safe_eval(f_expr, x, y)
            steps.append((x, y, f_val))

        except ValueError as e:
            raise ValueError(f"Ошибка вычисления на x={x}, y={y}: {e}")

    return steps

def adams_method(f_expr, initial_points, b, h, decimals):
    """
    Метод Адамса 3-го порядка (использует 4 начальные точки)
    """
    x0, y0, f0 = initial_points[0]
    x1, y1, f1 = initial_points[1]
    x2, y2, f2 = initial_points[2]
    x3, y3, f3 = initial_points[3]
    
    results = initial_points.copy()
    x_current = x3
    y_current = y3
    
    n_steps = int(round((b - x3) / h))
    
    for i in range(n_steps):
        try:
            # Формула Адамса 3-го порядка (явная)
            # y_{k+1} = y_k + h/24 * (55f_k - 59f_{k-1} + 37f_{k-2} - 9f_{k-3})
            y_next = y_current + h/24 * (55*f3 - 59*f2 + 37*f1 - 9*f0)
            
            x_next = x_current + h
            f_next = safe_eval(f_expr, x_next, y_next)
            
            # Обновляем значения для следующего шага
            x0, y0, f0 = x1, y1, f1
            x1, y1, f1 = x2, y2, f2
            x2, y2, f2 = x3, y3, f3
            x3, y3, f3 = x_next, y_next, f_next
            
            results.append((x_next, y_next, f_next))
            x_current = x_next
            y_current = y_next
            
        except ValueError as e:
            raise ValueError(f"Ошибка вычисления на x={x_current}, y={y_current}: {e}")
    
    return results

def count_output_decimals(eps_str):
    """
    Определяет количество знаков после запятой для вывода: n + 1
    где n — количество цифр после точки в eps_str
    """
    if '.' not in eps_str:
        return 2
    fractional = eps_str.split('.')[1].rstrip('0')  # убираем trailing нули
    return len(fractional) + 1

def format_value(value, width, decimals):
    """Форматирует значение для вывода (обрабатывает None)"""
    if value is None:
        return f"{'':<{width}}"
    return f"{value:<{width}.{decimals}f}"

def main():
    print("=== Решение ОДУ методом Адамса 3-го порядка ===\n")

    # Выбор уравнения
    print("Выберите способ ввода уравнения:")
    print("1. Использовать предустановленное уравнение: (x - y) / (2*x + 1)")
    print("2. Ввести своё уравнение")
    choice = input("Введите 1 или 2: ").strip()

    if choice == '1':
        f_expr = "(x - y) / (2*x + 1)"
        print(f"Используется уравнение: y' = {f_expr}\n")
    elif choice == '2':
        f_expr = input("Введите уравнение в виде y' = f(x,y): ").strip()
        if not f_expr:
            print("Уравнение не введено. Используем предустановленное.")
            f_expr = "(x - y) / (2*x + 1)"
        print(f"Используется уравнение: y' = {f_expr}\n")
    else:
        print("Неверный выбор. Используем предустановленное уравнение.")
        f_expr = "(x - y) / (2*x + 1)"

    # Запрос параметров
    try:
        x0 = float(input("Введите начальное значение x0: "))
        y0 = float(input("Введите начальное значение y0: "))
        b = float(input("Введите конец отрезка b: "))
        h = float(input("Введите шаг h: "))
        eps_str = input("Введите точность (например, 0.001): ").strip()
        eps = float(eps_str)
        if h <= 0 or eps <= 0:
            raise ValueError("Шаг и точность должны быть положительными.")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return

    if x0 > b:
        print("Ошибка: x0 должно быть <= b")
        return

    # Определяем количество знаков после запятой для вывода
    decimals = count_output_decimals(eps_str)
    print(f"\nТочность введена: {eps} → Вывод всех значений с {decimals} знаками после запятой\n")

    print("=== Результаты (метод Адамса 3-го порядка) ===\n")

    try:
        # Получаем начальные точки методом Рунге-Кутты
        print("Получение начальных точек методом Рунге-Кутты:")
        rk_steps = runge_kutta_4(f_expr, x0, y0, h, 3, decimals)
        
        # Формируем начальные точки для метода Адамса
        initial_points = [(x0, y0, safe_eval(f_expr, x0, y0))]
        for x, y, f_val in rk_steps:
            initial_points.append((x, y, f_val))
        
        # Применяем метод Адамса
        adams_results = adams_method(f_expr, initial_points, b, h, decimals)
        
    except ValueError as e:
        print(f"Ошибка вычисления: {e}")
        return

    # Вычисляем конечные разности
    f_values = [point[2] for point in adams_results]
    delta_f = []
    delta2_f = []
    delta3_f = []
    
    for i in range(len(f_values) - 1):
        delta_f.append(f_values[i+1] - f_values[i])
    
    for i in range(len(delta_f) - 1):
        delta2_f.append(delta_f[i+1] - delta_f[i])
    
    for i in range(len(delta2_f) - 1):
        delta3_f.append(delta2_f[i+1] - delta2_f[i])

    # Дополняем списки разностей значениями None для выравнивания
    while len(delta_f) < len(adams_results):
        delta_f.append(None)
    while len(delta2_f) < len(adams_results):
        delta2_f.append(None)
    while len(delta3_f) < len(adams_results):
        delta3_f.append(None)

    # Форматированный вывод в виде таблицы
    col_width = 10 + decimals
    print(f"{'k':<3} {'x_k':<{col_width}} {'y_k':<{col_width}} {'f_k':<{col_width}} {'Δf_k':<{col_width}} {'Δ²f_k':<{col_width}} {'Δ³f_k':<{col_width}}")
    print("-" * (3 + 6 * col_width + 5))

    # Выводим результаты
    for k, (x, y, f) in enumerate(adams_results):
        x_rounded = round(x, decimals)
        y_rounded = round(y, decimals)
        f_rounded = round(f, decimals)
        
        # Получаем конечные разности для текущей строки
        delta_f_val = delta_f[k] if k < len(delta_f) else None
        delta2_f_val = delta2_f[k] if k < len(delta2_f) else None
        delta3_f_val = delta3_f[k] if k < len(delta3_f) else None
        
        # Форматируем значения
        delta_f_str = format_value(delta_f_val, col_width, decimals)
        delta2_f_str = format_value(delta2_f_val, col_width, decimals)
        delta3_f_str = format_value(delta3_f_val, col_width, decimals)
        
        print(
            f"{k:<3} "
            f"{x_rounded:<{col_width}.{decimals}f} "
            f"{y_rounded:<{col_width}.{decimals}f} "
            f"{f_rounded:<{col_width}.{decimals}f} "
            f"{delta_f_str} "
            f"{delta2_f_str} "
            f"{delta3_f_str}"
        )

    print(f"\nРешение завершено. Точность: {eps} → Вывод с {decimals} знаками после запятой.")
    print("Примечание: Используется метод Адамса 3-го порядка для ОДУ 1-го порядка.")

if __name__ == "__main__":
    main()