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

def runge_kutta_4(f_expr, x0, y0, b, h, decimals):
    """
    Метод Рунге–Кутты 4-го порядка — чистая реализация без None-заглушек.
    Вычисляем и сохраняем только корректные шаги.
    """
    x = x0
    y = y0
    steps = []  # Будем добавлять только завершённые шаги

    # Сохраняем начальную точку
    steps.append((0, x, None, None, None, None, y))

    current_step = 1
    n = int(round((b - x0) / h))  # Исправлено: добавляем округление

    for i in range(n):  # Делаем ровно n шагов
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

            # Сохраняем результат этого шага
            steps.append((current_step, x, g1, g2, g3, g4, y))
            current_step += 1

        except ValueError as e:
            raise ValueError(f"Ошибка вычисления на x={x}, y={y}: {e}")

    return steps

def count_output_decimals(eps_str):
    """
    Определяет количество знаков после запятой для вывода: n + 1
    где n — количество цифр после точки в eps_str
    """
    if '.' not in eps_str:
        return 2
    fractional = eps_str.split('.')[1].rstrip('0')  # убираем trailing нули
    return len(fractional) + 1

def main():
    print("=== Решение ОДУ методом Рунге–Кутты 4-го порядка ===\n")

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

    # Определяем количество знаков после запятой для вывода: n + 1
    decimals = count_output_decimals(eps_str)
    print(f"\nТочность введена: {eps} → Вывод всех значений с {decimals} знаками после запятой\n")

    print("=== Результаты (метод Рунге–Кутты 4-го порядка) ===\n")

    try:
        rk4_steps = runge_kutta_4(f_expr, x0, y0, b, h, decimals)
    except ValueError as e:
        print(f"Ошибка в методе Рунге–Кутты: {e}")
        return
    


    # Форматированный вывод
    print(f"{'Шаг':<5} {'x':<10} {'g1':<{10 + decimals}} {'g2':<{10 + decimals}} {'g3':<{10 + decimals}} {'g4':<{10 + decimals}} {'y':<{10 + decimals}}")
    print("-" * (70 + 4 * decimals))

        # Форматированный вывод
    print(f"{'Шаг':<5} {'x':<10} {'g1':<{10 + decimals}} {'g2':<{10 + decimals}} {'g3':<{10 + decimals}} {'g4':<{10 + decimals}} {'y':<{10 + decimals}}")
    print("-" * (70 + 4 * decimals))

    for step, x, g1, g2, g3, g4, y in rk4_steps:
        x_rounded = round(x, decimals)
        y_rounded = round(y, decimals)

        if step == 0:
            # Для начального условия не выводим g1-g4
            print(f"{step:<5} {x_rounded:<10.{decimals}f} {'':<{10 + decimals}} {'':<{10 + decimals}} {'':<{10 + decimals}} {'':<{10 + decimals}} {y_rounded:<{10 + decimals}.{decimals}f}")
        else:
            # Для остальных шагов — округляем и выводим все g1-g4
            g1_rounded = round(g1, decimals)
            g2_rounded = round(g2, decimals)
            g3_rounded = round(g3, decimals)
            g4_rounded = round(g4, decimals)
            print(
                f"{step:<5} "
                f"{x_rounded:<10.{decimals}f} "
                f"{g1_rounded:<{10 + decimals}.{decimals}f} "
                f"{g2_rounded:<{10 + decimals}.{decimals}f} "
                f"{g3_rounded:<{10 + decimals}.{decimals}f} "
                f"{g4_rounded:<{10 + decimals}.{decimals}f} "
                f"{y_rounded:<{10 + decimals}.{decimals}f}"
            )

    print(f"\nРешение завершено. Точность: {eps} → Вывод с {decimals} знаками после запятой.")
    print("Примечание: Используется метод Рунге–Кутты для ОДУ 1-го порядка.")

if __name__ == "__main__":
    main()