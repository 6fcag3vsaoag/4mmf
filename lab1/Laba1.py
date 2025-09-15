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

def euler_method(f_expr, x0, y0, b, h):
    """Метод Эйлера"""
    x = x0
    y = y0
    steps = []
    current_step = 0

    while x <= b + 1e-10:  # Учитываем погрешность floating point
        steps.append((current_step, x, y))
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
        steps.append((current_step, x, y))
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
        if h <= 0:
            raise ValueError("Шаг должен быть положительным.")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return

    # Проверка, что x0 <= b
    if x0 > b:
        print("Ошибка: x0 должно быть <= b")
        return

    print("\n=== Результаты ===\n")

    # Метод Эйлера
    print("МЕТОД ЭЙЛЕРА:")
    print(f"{'Шаг':<5} {'x':<10} {'y':<15}")
    print("-" * 30)
    try:
        euler_steps = euler_method(f_expr, x0, y0, b, h)
        for step, x, y in euler_steps:
            print(f"{step:<5} {x:<10.6f} {y:<15.8f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера: {e}")
        return

    print("\nМЕТОД ЭЙЛЕРА-КОШИ:")
    print(f"{'Шаг':<5} {'x':<10} {'y':<15}")
    print("-" * 30)
    try:
        cauchy_steps = euler_cauchy_method(f_expr, x0, y0, b, h)
        for step, x, y in cauchy_steps:
            print(f"{step:<5} {x:<10.6f} {y:<15.8f}")
    except ValueError as e:
        print(f"Ошибка в методе Эйлера-Коши: {e}")
        return

    print("\nРешение завершено.")

if __name__ == "__main__":
    main()