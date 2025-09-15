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

def format_output(steps, method_name):
    """Форматированный вывод результатов"""
    print(f"\n{method_name}")
    
    # Вывод номеров шагов
    print("k  ", end="")
    for step, _, _, _, _ in steps:
        print(f"{step:>6}", end="")
    print()
    
    # Вывод x значений
    print("x  ", end="")
    for _, x, _, _, _ in steps:
        print(f"{x:>6.1f}", end="")
    print()
    
    # Вывод y значений
    print("y  ", end="")
    for _, _, y, _, _ in steps:
        print(f"{y:>6.3f}", end="")
    print()

def main():
    print("=== Решение ОДУ методами Эйлера и Эйлера-Коши ===\n")

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
        # Исправляем возможные ошибки ввода
        f_expr = f_expr.replace("2x", "2*x")
        if not f_expr:
            print("Уравнение не введено. Используем предустановленное.")
            f_expr = "(x + 1)**2 - y"
            print(f"Используется уравнение: y' = {f_expr}")
            print("Точное решение: y = x^2 + 1")
        else:
            print(f"Используется уравнение: y' = {f_expr}")
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

    # Шаг
    h = 0.1
    print(f"\nИспользуется шаг: h = {h}")

    # Метод Эйлера
    try:
        euler_steps = euler_method(f_expr, x0, y0, b, h)
        format_output(euler_steps, "МЕТОД ЭЙЛЕРА")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера: {e}")
        return

    # Метод Эйлера-Коши
    try:
        cauchy_steps = euler_cauchy_method(f_expr, x0, y0, b, h)
        format_output(cauchy_steps, "МЕТОД ЭЙЛЕРА-КОШИ")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера-Коши: {e}")
        return

    print("\nРешение завершено.")

if __name__ == "__main__":
    main()