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
        # Для пользовательского уравнения точное решение неизвестно
        exact_y = exact_solution(x) if f_expr == "(x + 1)**2 - y" else "N/A"
        error = abs(y - exact_y) if exact_y != "N/A" else "N/A"
        steps.append((current_step, x, y, exact_y, error))
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
        # Для пользовательского уравнения точное решение неизвестно
        exact_y = exact_solution(x) if f_expr == "(x + 1)**2 - y" else "N/A"
        error = abs(y - exact_y) if exact_y != "N/A" else "N/A"
        steps.append((current_step, x, y, exact_y, error))
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
    """Метод Рунге-Кутты 4-го порядка"""
    x = x0
    y = y0
    steps = []
    current_step = 0

    while x <= b + 1e-10:
        # Для пользовательского уравнения точное решение неизвестно
        exact_y = exact_solution(x) if f_expr == "(x + 1)**2 - y" else "N/A"
        error = abs(y - exact_y) if exact_y != "N/A" else "N/A"
        steps.append((current_step, x, y, exact_y, error))
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

def compare_methods(f_expr, x0, y0, b, h):
    """Сравнение методов между собой (когда точное решение неизвестно)"""
    euler_steps = euler_method(f_expr, x0, y0, b, h)
    cauchy_steps = euler_cauchy_method(f_expr, x0, y0, b, h)
    
    print(f"Сравнение методов (шаг h = {h}):")
    print(f"{'x':<10} {'y (Эйлер)':<15} {'y (Эйлер-Коши)':<15} {'Разница':<15}")
    print("-" * 55)
    
    # Для сравнения возьмем точки, которые есть в обоих методах
    for i in range(min(len(euler_steps), len(cauchy_steps))):
        x = euler_steps[i][1]
        y_euler = euler_steps[i][2]
        y_cauchy = cauchy_steps[i][2]
        diff = abs(y_euler - y_cauchy)
        print(f"{x:<10.6f} {y_euler:<15.8f} {y_cauchy:<15.8f} {diff:<15.8f}")

def find_step_for_epsilon(f_expr, x0, y0, b, epsilon, method):
    """Найти шаг h для достижения точности epsilon"""
    h = (b - x0) / 10  # Начальный шаг
    max_error = float('inf')
    
    # Ограничим количество итераций для избежания бесконечного цикла
    iterations = 0
    max_iterations = 15
    
    while max_error > epsilon and iterations < max_iterations and h > 1e-10:
        if method == 'euler':
            steps = euler_method(f_expr, x0, y0, b, h)
        elif method == 'cauchy':
            steps = euler_cauchy_method(f_expr, x0, y0, b, h)
        else:
            raise ValueError("Неизвестный метод")
            
        # Найти максимальную погрешность (только для предустановленного уравнения)
        if f_expr == "(x + 1)**2 - y":
            max_error = max(step[4] for step in steps if step[4] != "N/A")
        else:
            # Для пользовательских уравнений используем сравнение методов
            # Пока что установим большую погрешность, чтобы не зацикливаться
            max_error = epsilon / 2  # Это значение позволит выйти из цикла
            break
            
        if max_error > epsilon:
            h /= 2  # Уменьшить шаг в 2 раза
        iterations += 1
        
    return h, max_error, steps

def main():
    print("=== Решение ОДУ методами Эйлера и Эйлера-Коши с контролем точности ===\n")

    # Выбор уравнения
    print("Выберите способ ввода уравнения:")
    print("1. Использовать предустановленное уравнение: (x + 1)**2 - y")
    print("2. Ввести своё уравнение")
    choice = input("Введите 1 или 2: ").strip()

    if choice == '1':
        f_expr = "(x + 1)**2 - y"
        print(f"Используется уравнение: y' = {f_expr}")
        print("Точное решение: y = x^2 + 1")
    elif choice == '2':
        f_expr = input("Введите уравнение в виде y' = f(x,y): ").strip()
        if not f_expr:
            print("Уравнение не введено. Используем предустановленное.")
            f_expr = "(x + 1)**2 - y"
            print(f"Используется уравнение: y' = {f_expr}")
            print("Точное решение: y = x^2 + 1")
        else:
            print(f"Используется уравнение: y' = {f_expr}")
            print("Точное решение недоступно для пользовательского уравнения")
    else:
        print("Неверный выбор. Используем предустановленное уравнение.")
        f_expr = "(x + 1)**2 - y"
        print(f"Используется уравнение: y' = {f_expr}")
        print("Точное решение: y = x^2 + 1")

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

    # Проверка начальных условий (только для предустановленного уравнения)
    if f_expr == "(x + 1)**2 - y" and y0 != exact_solution(x0):
        print(f"Предупреждение: Начальное значение y0={y0} не соответствует точному решению y={exact_solution(x0)} при x0={x0}")

    # Заданная точность
    epsilon = 10**-2
    print(f"\nЗаданная точность: ε = {epsilon}")

    print("\n=== Поиск шага для достижения точности ===\n")

    # Обработка разных случаев
    if f_expr == "(x + 1)**2 - y":
        # Для предустановленного уравнения используем контроль точности
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
            error_str = f"{error:.8f}" if error != "N/A" else "N/A"
            exact_y_str = f"{exact_y:.8f}" if exact_y != "N/A" else "N/A"
            print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y_str:<15} {error_str:<15}")

        print("\nМЕТОД ЭЙЛЕРА-КОШИ:")
        print(f"Шаг: h = {h_cauchy:.6f}")
        print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15} {'y (точное)':<15} {'Погрешность':<15}")
        print("-" * 65)
        for step, x, y, exact_y, error in cauchy_steps:
            error_str = f"{error:.8f}" if error != "N/A" else "N/A"
            exact_y_str = f"{exact_y:.8f}" if exact_y != "N/A" else "N/A"
            print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y_str:<15} {error_str:<15}")

        # Сравнение с методом Рунге-Кутты (для дополнительной проверки)
        print("\nДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА:")
        print("Сравнение с методом Рунге-Кутты 4-го порядка (шаг h = 0.001):")
        try:
            rk_steps = runge_kutta_method(f_expr, x0, y0, b, 0.001)
            print(f"{'x':<10} {'y (Эйлер)':<15} {'y (Эйлер-Коши)':<15} {'y (Рунге-Кутта)':<15}")
            print("-" * 55)
            
            # Для сравнения возьмем точки, которые есть в методе Эйлера
            rk_dict = {step[1]: step[2] for step in rk_steps}  # x: y
            
            for step in euler_steps[:len(rk_steps)]:
                x = step[1]
                y_euler = step[2]
                # Найти соответствующее значение в методе Эйлера-Коши
                y_cauchy = None
                for c_step in cauchy_steps:
                    if abs(c_step[1] - x) < 1e-10:
                        y_cauchy = c_step[2]
                        break
                
                # Найти соответствующее значение в методе Рунге-Кутты
                y_rk = rk_dict.get(x, "N/A")
                
                if y_cauchy is not None and y_rk != "N/A":
                    y_rk_str = f"{y_rk:.8f}" if y_rk != "N/A" else "N/A"
                    print(f"{x:<10.6f} {y_euler:<15.8f} {y_cauchy:<15.8f} {y_rk_str:<15}")
        except ValueError as e:
            print(f"Ошибка в методе Рунге-Кутты: {e}")
    else:
        # Для пользовательского уравнения используем фиксированный шаг
        print("\nДля пользовательского уравнения используется фиксированный шаг h = 0.1")
        h = 0.1
        
        # Метод Эйлера
        print("\nМЕТОД ЭЙЛЕРА:")
        print(f"Шаг: h = {h}")
        try:
            euler_steps = euler_method(f_expr, x0, y0, b, h)
            print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15}")
            print("-" * 30)
            for step, x, y, _, _ in euler_steps:
                print(f"{step:<5} {x:<10.6f} {y:<15.8f}")
        except ValueError as e:
            print(f"Ошибка в методе Эйлера: {e}")
            return

        # Метод Эйлера-Коши
        print("\nМЕТОД ЭЙЛЕРА-КОШИ:")
        print(f"Шаг: h = {h}")
        try:
            cauchy_steps = euler_cauchy_method(f_expr, x0, y0, b, h)
            print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15}")
            print("-" * 30)
            for step, x, y, _, _ in cauchy_steps:
                print(f"{step:<5} {x:<10.6f} {y:<15.8f}")
        except ValueError as e:
            print(f"Ошибка в методе Эйлера-Коши: {e}")
            return
            
        # Сравнение методов
        print("\nСРАВНЕНИЕ МЕТОДОВ:")
        compare_methods(f_expr, x0, y0, b, h)

    print("\nРешение завершено.")

if __name__ == "__main__":
    main()