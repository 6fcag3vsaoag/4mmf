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

def main():
    print("=== Решение ОДУ методами Эйлера и Эйлера-Коши ===\n")

    # Предустановленное уравнение
    f_expr = "(x + 1)**2 - y"
    print(f"Используется уравнение: y' = {f_expr}")
    print("Точное решение: y = x^2 + 1\n")

    # Запрос параметров
    try:
        x0 = float(input("Введите начальное значение x0: "))
        y0 = float(input("Введите начальное значение y0: "))
        b = float(input("Введите конец отрезка b: "))
        h = float(input("Введите шаг h: "))
        if h <= 0:
            raise ValueError("Шаг должен быть положительным.")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return

    # Проверка, что x0 <= b
    if x0 > b:
        print("Ошибка: x0 должно быть <= b")
        return

    # Проверка начальных условий
    if y0 != exact_solution(x0):
        print(f"Предупреждение: Начальное значение y0={y0} не соответствует точному решению y={exact_solution(x0)} при x0={x0}")

    print("\n=== Результаты ===\n")

    # Метод Эйлера
    print("МЕТОД ЭЙЛЕРА:")
    print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15} {'y (точное)':<15} {'Погрешность':<15}")
    print("-" * 65)
    try:
        euler_steps = euler_method(f_expr, x0, y0, b, h)
        for step, x, y, exact_y, error in euler_steps:
            print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y:<15.8f} {error:<15.8f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера: {e}")
        return

    print("\nМЕТОД ЭЙЛЕРА-КОШИ:")
    print(f"{'Шаг':<5} {'x':<10} {'y (приближ)':<15} {'y (точное)':<15} {'Погрешность':<15}")
    print("-" * 65)
    try:
        cauchy_steps = euler_cauchy_method(f_expr, x0, y0, b, h)
        for step, x, y, exact_y, error in cauchy_steps:
            print(f"{step:<5} {x:<10.6f} {y:<15.8f} {exact_y:<15.8f} {error:<15.8f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера-Коши: {e}")
        return

    print("\nРешение завершено.")

if __name__ == "__main__":
    main()