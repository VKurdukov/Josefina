import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gamma
from math import factorial


# Загружаем данные из Excel (каждый лист соответствует команде)
excel_file = pd.ExcelFile(r"C:\Users\Владимир\Desktop\Josefina\LaLiga1.xlsx")

# Параметры
num = np.arange(0, 10)

# Создаем списки для scored и skipped
scored = []
skipped = []
matches_per_team = []

# Загружаем данные с каждого листа Excel файла
for sheet_name in excel_file.sheet_names:
    data = pd.read_excel(excel_file, sheet_name=sheet_name, header=0).values  # Убираем заголовок

    # Проверяем, что данные являются числовыми
    try:
        data = data.astype(float)
    except ValueError:
        raise ValueError(f"В листе '{sheet_name}' обнаружены нечисловые значения. Убедитесь, что все значения числовые.")

    # 1 и 2 столбцы: забитые голы, количество матчей, в которых они были забиты
    scoredraw = data[:10, 1]  # Забитые голы
    # 3 и 4 столбцы: пропущенные голы, количество матчей, в которых они были пропущены
    skippedraw = data[:10, 3]  # Пропущенные голы

    # Сохраняем данные scored и skipped для команды
    scored.append(list(zip(num, scoredraw)))
    skipped.append(list(zip(num, skippedraw)))
    matches = np.sum(scoredraw)  # или np.sum(skippedraw), если пропущенные голы также учитываются
    matches_per_team.append(matches)

# Функция Пуассона для подгонки
def poisson_func(x, lambd,matches):
    return matches * poisson.pmf(x, lambd)

# Функция для вычисления коэффициентов
def Coef(a, b):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных.")

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    lambda1, lambda2, lambda3, lambda4 = lambda1[0], lambda2[0], lambda3[0], lambda4[0]
    

    # Вероятности для забитых и пропущенных голов
    Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    Pscore2 = lambda x: poisson.pmf(x, (lambda2 + lambda3) / 2)

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    # Матрица вероятностей счета
    Mmatrix = np.array([[Pscore1(i) * Pscore2(j) for j in range(7)] for i in range(7)])

    return Win1, draw, Win2, TotalGoals, lambda1, lambda2, lambda3, lambda4




def Coef_fans(a, b, fans):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных.")

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    lambda1, lambda2, lambda3, lambda4 = lambda1[0], lambda2[0], lambda3[0], lambda4[0]
    lambda1 = lambda1 + fans/100000
    

    # Вероятности для забитых и пропущенных голов
    Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    Pscore2 = lambda x: poisson.pmf(x, (lambda2 + lambda3) / 2)

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    # Матрица вероятностей счета
    Mmatrix = np.array([[Pscore1(i) * Pscore2(j) for j in range(7)] for i in range(7)])

    return Win1, draw, Win2, TotalGoals


def Coef_boomer(a, b, w1, dr, w2):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных.")

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    lambda1, lambda2, lambda3, lambda4 = lambda1[0], lambda2[0], lambda3[0], lambda4[0]
    
    

    # Вероятности для забитых и пропущенных голов
    Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    Pscore2 = lambda x: poisson.pmf(x, (lambda2 + lambda3) / 2)

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))
    if w1 < 1.65 and w1 * Win1 < 1:
        Win1 = 1.1/w1
        Win2 = (1-Win1)*Win2/(Win2+draw)
        draw = (1-Win1-Win2)
    elif w2 < 1.65 and w2 * Win2 < 1:
        Win2 = 1.1/w2
        Win1 = (1-Win2)*Win1/(Win1+draw)
        draw = (1-Win1-Win2)
    

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    # Матрица вероятностей счета
    Mmatrix = np.array([[Pscore1(i) * Pscore2(j) for j in range(7)] for i in range(7)])

    return Win1, draw, Win2, TotalGoals
 
def Coef_quad(a, b):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных.")

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    lambda1, lambda2, lambda3, lambda4 = lambda1[0], lambda2[0], lambda3[0], lambda4[0]
    

    # Вероятности для забитых и пропущенных голов
    Pscore1 = lambda x: poisson.pmf(x, ((lambda1**2 + lambda4**2) / 2)**(1/2))
    Pscore2 = lambda x: poisson.pmf(x, ((lambda2**2 + lambda3**2) / 2)**(1/2))

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    # Матрица вероятностей счета
    Mmatrix = np.array([[Pscore1(i) * Pscore2(j) for j in range(7)] for i in range(7)])

    return Win1, draw, Win2, TotalGoals

def Prime(a, b, grade_1, grade_2):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(
            f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных."
        )

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    # Переводим их в числа (float)
    lambda1, lambda2, lambda3, lambda4 = (
        lambda1[0],
        lambda2[0],
        lambda3[0],
        lambda4[0],
    )

    # --- Определяем Pscore1 на основе grade_1 (атака A vs. защита B) ---
    if grade_1 == 0:
        Pscore1 = lambda x: poisson.pmf(x, min(lambda1, lambda4))
    elif grade_1 == 1:
        # Гармоническое среднее: 2/(1/a + 1/b)
        Pscore1 = lambda x: poisson.pmf(
            x, 2 * lambda1 * lambda4 / (lambda1 + lambda4)
        )
    elif grade_1 == 2:
        # Геометрическое среднее: sqrt(a*b)
        Pscore1 = lambda x: poisson.pmf(x, math.sqrt(lambda1 * lambda4))
    elif grade_1 == 3:
        # Арифметическое среднее: (a + b)/2
        Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    elif grade_1 == 4:
        # Квадратичное среднее: sqrt((a^2 + b^2)/2)
        Pscore1 = lambda x: poisson.pmf(
            x, math.sqrt((lambda1**2 + lambda4**2) / 2)
        )
    elif grade_1 == 5:
        Pscore1 = lambda x: poisson.pmf(x, max(lambda1, lambda4))
    else:
        raise ValueError("Неверное значение grade_1 (0..5)")

    # --- Определяем Pscore2 на основе grade_2 (атака B vs. защита A) ---
    if grade_2 == 0:
        Pscore2 = lambda x: poisson.pmf(x, min(lambda3, lambda2))
    elif grade_2 == 1:
        Pscore2 = lambda x: poisson.pmf(
            x, 2 * lambda3 * lambda2 / (lambda3 + lambda2)
        )
    elif grade_2 == 2:
        Pscore2 = lambda x: poisson.pmf(x, math.sqrt(lambda3 * lambda2))
    elif grade_2 == 3:
        Pscore2 = lambda x: poisson.pmf(x, (lambda3 + lambda2) / 2)
    elif grade_2 == 4:
        Pscore2 = lambda x: poisson.pmf(
            x, math.sqrt((lambda3**2 + lambda2**2) / 2)
        )
    elif grade_2 == 5:
        Pscore2 = lambda x: poisson.pmf(x, max(lambda3, lambda2))
    else:
        raise ValueError("Неверное значение grade_2 (0..5)")

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    return Win1, draw, Win2

def SuperPrime(a, b, grade_1, grade_2, forma_1, forma_2):
    # Приводим индексы a и b к индексации с нуля (так как в Python индексация начинается с 0)
    a -= 1
    b -= 1

    # Проверяем, что индексы a и b находятся в пределах допустимого диапазона
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(
            f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных."
        )

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    # Переводим их в числа (float)
    lambda1, lambda2, lambda3, lambda4 = (
        lambda1[0],
        lambda2[0],
        lambda3[0],
        lambda4[0],
    )

    # Используем нововведние формы команд
    n_forma = 5
    if forma_1 > 0:
        lambda1 = lambda1*(1+(forma_1)/n_forma)
    elif forma_1 < 0:
        lambda2 = lambda2*(1-(forma_1)/n_forma)
    
    if forma_2 > 0:
        lambda3 = lambda3*(1+(forma_2)/n_forma)
    elif forma_2 < 0:
        lambda4 = lambda4*(1-(forma_2)/n_forma)
    

    # --- Определяем Pscore1 на основе grade_1 (атака A vs. защита B) ---
    if grade_1 == 0:
        Pscore1 = lambda x: poisson.pmf(x, min(lambda1, lambda4))
    elif grade_1 == 1:
        # Гармоническое среднее: 2/(1/a + 1/b)
        Pscore1 = lambda x: poisson.pmf(
            x, 2 * lambda1 * lambda4 / (lambda1 + lambda4)
        )
    elif grade_1 == 2:
        # Геометрическое среднее: sqrt(a*b)
        Pscore1 = lambda x: poisson.pmf(x, math.sqrt(lambda1 * lambda4))
    elif grade_1 == 3:
        # Арифметическое среднее: (a + b)/2
        Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    elif grade_1 == 4:
        # Квадратичное среднее: sqrt((a^2 + b^2)/2)
        Pscore1 = lambda x: poisson.pmf(
            x, math.sqrt((lambda1**2 + lambda4**2) / 2)
        )
    elif grade_1 == 5:
        Pscore1 = lambda x: poisson.pmf(x, max(lambda1, lambda4))
    else:
        raise ValueError("Неверное значение grade_1 (0..5)")

    # --- Определяем Pscore2 на основе grade_2 (атака B vs. защита A) ---
    if grade_2 == 0:
        Pscore2 = lambda x: poisson.pmf(x, min(lambda3, lambda2))
    elif grade_2 == 1:
        Pscore2 = lambda x: poisson.pmf(
            x, 2 * lambda3 * lambda2 / (lambda3 + lambda2)
        )
    elif grade_2 == 2:
        Pscore2 = lambda x: poisson.pmf(x, math.sqrt(lambda3 * lambda2))
    elif grade_2 == 3:
        Pscore2 = lambda x: poisson.pmf(x, (lambda3 + lambda2) / 2)
    elif grade_2 == 4:
        Pscore2 = lambda x: poisson.pmf(
            x, math.sqrt((lambda3**2 + lambda2**2) / 2)
        )
    elif grade_2 == 5:
        Pscore2 = lambda x: poisson.pmf(x, max(lambda3, lambda2))
    else:
        raise ValueError("Неверное значение grade_2 (0..5)")

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    return Win1, draw, Win2,lambda1,lambda2,lambda3,lambda4

def Z(lmbda, nu):
    return np.sum([lmbda**n / gamma(n + 1)**nu for n in range(11)])

def MaxwellPoissonDist(x, lmbda, nu):
    x_int = np.round(x).astype(int)  # Округление до целого числа для массива
    return (lmbda**x_int / gamma(x_int + 1)**nu) / Z(lmbda, nu)

def fit_function(x, lmbda, nu, matches):
    return matches * MaxwellPoissonDist(x, lmbda, nu)

def Coef_Max(a, b):
    """Основная функция для вычисления коэффициентов."""
    # Приводим индексы a и b к индексации с нуля
    a -= 1
    b -= 1


    # Извлечение данных
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Фитирование параметров распределения
    x_data = np.arange(len(scored_a))
    try:
        # Фитирование для забитых голов команды a
        params1, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, matches_per_team[a]), 
            x_data, scored_a, p0=[2,2]
        )
        lmbda1, nu1 = params1

        # Фитирование для пропущенных голов команды a
        params2, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, matches_per_team[a]), 
            x_data, skipped_a,p0=[2,2])
        lmbda2, nu2 = params2

        # Фитирование для забитых голов команды b
        params3, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, matches_per_team[b]), 
            x_data, scored_b,p0=[2,2]
        )
        lmbda3, nu3 = params3

        # Фитирование для пропущенных голов команды b
        params4, _ = curve_fit(
            lambda x, l, v :fit_function(x, l, v, matches_per_team[b]), 
            x_data, skipped_b,p0=[2,2]
        )
        lmbda4, nu4 = params4

    except RuntimeError as e:
        raise ValueError("Ошибка фитирования параметров") from e

    # Функции вероятностей
    def Pscore1(x):
        """Вероятность для команды 1."""
        total = 0
        for i in range(21):  # Суммируем от 0 до 20
            arg = 2 * x - i
            if arg >= 0:  # Игнорируем отрицательные значения
                total += fit_function(arg, lmbda1, nu1,matches_per_team[a]) * fit_function(i, lmbda4, nu4,matches_per_team[b])
        return total / (matches_per_team[a]*matches_per_team[b])

    def Pscore2(x):
        """Вероятность для команды 2."""
        total = 0
        for i in range(21):
            arg = 2 * x - i
            if arg >= 0:
                total += fit_function(arg, lmbda2, nu2,matches_per_team[a]) * fit_function(i, lmbda3, nu3,matches_per_team[b])
        return total / (matches_per_team[a]*matches_per_team[b])

    # Кумулятивные функции
    def f1(n):  # Целое число
        return sum(Pscore1(i / 2) for i in range(int(2 * n + 1)))

    def f2(n):
        return sum(Pscore2(i / 2) for i in range(int(2 * n + 1)))

    # Вычисление исходов
    Win1 = sum(Pscore1(i / 2) * f2(i/2-0.5) for i in range(1, 21))  # i: 1-20
    Win2 = sum(Pscore2(i / 2) * f1((i - 1) / 2) for i in range(1, 21))
    draw = sum(Pscore1(i / 2) * Pscore2(i / 2) for i in range(0,21))       # i: 0-20
    Mmatrix = np.array([[Pscore1(i/2) * Pscore2(j/2) for j in range(10)] for i in range(10)])

    # Возвращаем результаты
    return [Win1,draw,Win2]

def NormalPrime(a, b, grade_1, grade_2, forma_1, forma_2):
    a -= 1
    b -= 1

    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(
            f"Индексы команд a={a + 1} или b={b + 1} выходят за пределы доступных данных."
        )

    # Подгонка Пуассона для забитых и пропущенных голов команд a и b
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Находим параметры распределения для команд a и b
    lambda1, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, lambd: poisson_func(x, lambd, matches_per_team[b]), num, skipped_b, p0=[1])

    lambda1, lambda2, lambda3, lambda4 = (
        lambda1[0],
        lambda2[0],
        lambda3[0],
        lambda4[0],
    )
    Summa = lambda1 + lambda2 + lambda3 + lambda4
    print(Summa, lambda1, lambda2, lambda3, lambda4)

    # Initialize lambda_ variables with original values
    lambda_1 = lambda1  # Default to original lambda1
    lambda_2 = lambda2  # Default to original lambda2
    lambda_3 = lambda3  # Default to original lambda3
    lambda_4 = lambda4  # Default to original lambda4

    # Adjust based on forma_1 and forma_2
    n_forma = 5
    if forma_1 > 0:
        lambda_1 = lambda1 * (1 + forma_1 / n_forma)
    elif forma_1 < 0:
        lambda_2 = lambda2 * (1 - forma_1 / n_forma)

    if forma_2 > 0:
        lambda_3 = lambda3 * (1 + forma_2 / n_forma)
    elif forma_2 < 0:
        lambda_4 = lambda4 * (1 - forma_2 / n_forma)

    Summa_forma = lambda_1 + lambda_2 + lambda_3 + lambda_4
    print(Summa_forma, lambda_1, lambda_2, lambda_3, lambda_4)

    # Scale lambdas to maintain the original sum
    lambda1 = lambda_1 * Summa / Summa_forma
    lambda2 = lambda_2 * Summa / Summa_forma
    lambda3 = lambda_3 * Summa / Summa_forma
    lambda4 = lambda_4 * Summa / Summa_forma
    print(lambda1, lambda2, lambda3, lambda4)

    # --- Определяем Pscore1 на основе grade_1 (атака A vs. защита B) ---
    if grade_1 == 0:
        Pscore1 = lambda x: poisson.pmf(x, min(lambda1, lambda4))
    elif grade_1 == 1:
        # Гармоническое среднее: 2/(1/a + 1/b)
        Pscore1 = lambda x: poisson.pmf(
            x, 2 * lambda1 * lambda4 / (lambda1 + lambda4)
        )
    elif grade_1 == 2:
        # Геометрическое среднее: sqrt(a*b)
        Pscore1 = lambda x: poisson.pmf(x, math.sqrt(lambda1 * lambda4))
    elif grade_1 == 3:
        # Арифметическое среднее: (a + b)/2
        Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    elif grade_1 == 4:
        # Квадратичное среднее: sqrt((a^2 + b^2)/2)
        Pscore1 = lambda x: poisson.pmf(
            x, math.sqrt((lambda1**2 + lambda4**2) / 2)
        )
    elif grade_1 == 5:
        Pscore1 = lambda x: poisson.pmf(x, max(lambda1, lambda4))
    else:
        raise ValueError("Неверное значение grade_1 (0..5)")

    # --- Определяем Pscore2 на основе grade_2 (атака B vs. защита A) ---
    if grade_2 == 0:
        Pscore2 = lambda x: poisson.pmf(x, min(lambda3, lambda2))
    elif grade_2 == 1:
        Pscore2 = lambda x: poisson.pmf(
            x, 2 * lambda3 * lambda2 / (lambda3 + lambda2)
        )
    elif grade_2 == 2:
        Pscore2 = lambda x: poisson.pmf(x, math.sqrt(lambda3 * lambda2))
    elif grade_2 == 3:
        Pscore2 = lambda x: poisson.pmf(x, (lambda3 + lambda2) / 2)
    elif grade_2 == 4:
        Pscore2 = lambda x: poisson.pmf(
            x, math.sqrt((lambda3**2 + lambda2**2) / 2)
        )
    elif grade_2 == 5:
        Pscore2 = lambda x: poisson.pmf(x, max(lambda3, lambda2))
    else:
        raise ValueError("Неверное значение grade_2 (0..5)")

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11)]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11)]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    return Win1, draw, Win2,lambda1,lambda2,lambda3,lambda4,TotalGoals