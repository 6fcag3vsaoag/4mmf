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
    # Добавьте другие нужные функции по необходимости
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

def exact_solution(x):
    """Точное решение уравнения y = x^2 + 1"""
    return x**2 + 1

def euler_method(f_expr, x0, y0, b, h):
    """Метод Эйлера"""
    x = x0
    y = y0
    steps = []
    current_step = 0

    while x <= b + 1e-10:  # Учитываем погрешность floating point
        exact_y = exact_solution(x)
        steps.append((current_step, x, y, exact_y, abs(y - exact_y)))
        if x >= b:
            break
        try:
            dydx = safe_eval(f_expr, x, y)
        except ValueError as e:
            raise ValueError(f"Ошибка вычисления производной на x={x}, y={y}: {e}")
        y += h * dydx
        x += h
        current_step += 1

    return steps

def euler_cauchy_method(f_expr, x0, y0, b, h):
    """Метод Эйлера-Коши (улучшенный Эйлер)"""
    x = x0
    y = y0
    steps = []
    current_step = 0

    while x <= b + 1e-10:
        exact_y = exact_solution(x)
        steps.append((current_step, x, y, exact_y, abs(y - exact_y)))
        if x >= b:
            break
        try:
            k1 = safe_eval(f_expr, x, y)
            k2 = safe_eval(f_expr, x + h, y + h * k1)
            y += h * (k1 + k2) / 2
            x += h
            current_step += 1
        except ValueError as e:
            raise ValueError(f"Ошибка вычисления производной на x={x}, y={y}: {e}")

    return steps

def runge_kutta_method(f_expr, x0, y0, b, h):
    """Метод Рунге-Кутты 4-го порядка (для проверки точности)"""
    x = x0
    y = y0
    steps = []
    current_step = 0

    while x <= b + 1e-10:
        exact_y = exact_solution(x)
        steps.append((current_step, x, y, exact_y, abs(y - exact_y)))
        if x >= b:
            break
        try:
            k1 = h * safe_eval(f_expr, x, y)
            k2 = h * safe_eval(f_expr, x + h/2, y + k1/2)
            k3 = h * safe_eval(f_expr, x + h/2, y + k2/2)
            k4 = h * safe_eval(f_expr, x + h, y + k3)
            y += (k1 + 2*k2 + 2*k3 + k4) / 6
            x += h
            current_step += 1
        except ValueError as e:
            raise ValueError(f"Ошибка вычисления производной на x={x}, y={y}: {e}")

    return steps

def find_step_for_epsilon(f_expr, x0, y0, b, epsilon, method):
    """Найти шаг h для достижения точности epsilon"""
    h = (b - x0) / 10  # Начальный шаг
    max_error = float('inf')
    
    # Ограничим количество итераций для избежания бесконечного цикла
    iterations = 0
    max_iterations = 20
    
    while max_error > epsilon and iterations < max_iterations:
        if method == 'euler':
            steps = euler_method(f_expr, x0, y0, b, h)
        elif method == 'cauchy':
            steps = euler_cauchy_method(f_expr, x0, y0, b, h)
        else:
            raise ValueError("Неизвестный метод")
            
        # Найти максимальную погрешность
        max_error = max(step[4] for step in steps)
        
        if max_error > epsilon:
            h /= 2  # Уменьшить шаг в 2 раза
        iterations += 1
        
    return h, max_error, steps

def main():
    print("=== Решение ОДУ методами Эйлера и Эйлера-Коши с контролем точности ===\n")

    # Предустановленное уравнение
    f_expr = "(x + 1)**2 - y"
    print(f"Используется уравнение: y' = {f_expr}")
    print("Точное решение: y = x^2 + 1\n")

    # Запрос параметров
    try:
        x0 = float(input("Введите начальное значение x0: "))
        y0 = float(input("Введите начальное значение y0: "))
        b = float(input("Введите конец отрезка b: "))
        if x0 >= b:
            raise ValueError("x0 должно быть < b")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return

    # Проверка начальных условий
    if y0 != exact_solution(x0):
        print(f"Предупреждение: Начальное значение y0={y0} не соответствует точному решению y={exact_solution(x0)} при x0={x0}")

    # Заданная точность
    epsilon = 10**-2
    print(f"\nЗаданная точность: ε = {epsilon}")

    print("\n=== Поиск шага для достижения точности ===\n")

    # Найти шаг для метода Эйлера
    print("Поиск шага для метода Эйлера...")
    try:
        h_euler, error_euler, euler_steps = find_step_for_epsilon(f_expr, x0, y0, b, epsilon, 'euler')
        print(f"Найденный шаг: h = {h_euler:.6f}")
        print(f"Достигнутая точность: ε = {error_euler:.6f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера: {e}")
        return

    # Найти шаг для метода Эйлера-Коши
    print("\nПоиск шага для метода Эйлера-Коши...")
    try:
        h_cauchy, error_cauchy, cauchy_steps = find_step_for_epsilon(f_expr, x0, y0, b, epsilon, 'cauchy')
        print(f"Найденный шаг: h = {h_cauchy:.6f}")
        print(f"Достигнутая точность: ε = {error_cauchy:.6f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера-Коши: {e}")
        return

    print("\n=== Результаты ===\n")

    # Метод Эйлера
    print("МЕТОД ЭЙЛЕРА:")
    print(f"Шаг: h = {h_euler:.6f}")
    print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15} {'y (точное)':<15} {'Погрешность':<15}")
    print("-" * 65)
    for step, x, y, exact_y, error in euler_steps:
        print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y:<15.8f} {error:<15.8f}")

    print("\nМЕТОД ЭЙЛЕРА-КОШИ:")
    print(f"Шаг: h = {h_cauchy:.6f}")
    print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15} {'y (точное)':<15} {'Погрешность':<15}")
    print("-" * 65)
    for step, x, y, exact_y, error in cauchy_steps:
        print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y:<15.8f} {error:<15.8f}")

    # Сравнение с методом Рунге-Кутты (для дополнительной проверки)
    print("\nДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА:")
    print("Сравнение с методом Рунге-Кутты 4-го порядка (шаг h = 0.001):")
    try:
        rk_steps = runge_kutta_method(f_expr, x0, y0, b, 0.001)
        print(f"{'x':<10} {'y (Эйлер)':<15} {'y (Эйлер-Коши)':<15} {'y (Рунге-Кутта)':<15}")
        print("-" * 55)
        
        # Для сравнения возьмем точки, которые есть в обоих методах
        rk_dict = {step[1]: step[2] for step in rk_steps}  # x: y
        
        for step in euler_steps:
            x = step[1]
            y_euler = step[2]
            # Найти соответствующее значение в методе Эйлера-Коши
            y_cauchy = None
            for c_step in cauchy_steps:
                if abs(c_step[1] - x) < 1e-10:
                    y_cauchy = c_step[2]
                    break
            
            # Найти соответствующее значение в методе Рунге-Кутты
            y_rk = None
            if x in rk_dict:
                y_rk = rk_dict[x]
            elif x + 0.001 in rk_dict:
                y_rk = rk_dict[x + 0.001]
            elif x - 0.001 in rk_dict:
                y_rk = rk_dict[x - 0.001]
                
            if y_cauchy is not None and y_rk is not None:
                print(f"{x:<10.6f} {y_euler:<15.8f} {y_cauchy:<15.8f} {y_rk:<15.8f}")
    except ValueError as e:
        print(f"Ошибка в методе Рунге-Кутты: {e}")

    print("\nРешение завершено.")

if __name__ == "__main__":
    main()