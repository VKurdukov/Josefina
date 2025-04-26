import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gamma
from math import factorial
from scipy.stats import nbinom
import matplotlib.pyplot as plt


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
    #print(Summa, lambda1, lambda2, lambda3, lambda4)

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
    #print(Summa_forma, lambda_1, lambda_2, lambda_3, lambda_4)

    # Scale lambdas to maintain the original sum
    lambda1 = lambda_1 * Summa / Summa_forma
    lambda2 = lambda_2 * Summa / Summa_forma
    lambda3 = lambda_3 * Summa / Summa_forma
    lambda4 = lambda_4 * Summa / Summa_forma
    #print(lambda1, lambda2, lambda3, lambda4)

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

def Coef_Monte(a, b, n_simulations=100000):
    # Приводим индексы к нулевой базе
    a -= 1
    b -= 1

    # Проверка валидности индексов
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Invalid team indices: a={a+1} or b={b+1}")

    # Извлекаем данные
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Функция для фитинга с ошибками
    def fit_poisson(x_data, y_data, matches):
        valid = y_data > 0
        if sum(valid) == 0:
            return 0.0, 0.0
        x_fit = x_data[valid]
        y_fit = y_data[valid]
        sigma = np.sqrt(y_fit)
        sigma[sigma == 0] = 1e-6
        
        try:
            params, cov = curve_fit(
                lambda x, lmbd: poisson_func(x, lmbd, matches),
                x_fit, y_fit,
                p0=[1.0],
                sigma=sigma,
                absolute_sigma=True
            )
            err = np.sqrt(np.diag(cov))[0]
        except:
            params = [np.sum(x_data * y_data) / matches]
            err = 0.0
        return params[0], err

    # Получаем параметры с ошибками
    lambda1, lambda1_err = fit_poisson(num, scored_a, matches_per_team[a])
    lambda2, lambda2_err = fit_poisson(num, skipped_a, matches_per_team[a])
    lambda3, lambda3_err = fit_poisson(num, scored_b, matches_per_team[b])
    lambda4, lambda4_err = fit_poisson(num, skipped_b, matches_per_team[b])

    # Монте-Карло симуляция
    #np.random.seed(42)
    lambdas = [
        np.maximum(np.random.normal(l, l_err, n_simulations), 1e-9)
        for l, l_err in [(lambda1, lambda1_err), (lambda2, lambda2_err),
                        (lambda3, lambda3_err), (lambda4, lambda4_err)]
    ]

    # Расчет вероятностей
    win1, win2, draw, total = [], [], [], []
    
    for i in range(n_simulations):
        l1, l2, l3, l4 = [l[i] for l in lambdas]
        
        mu_a = (l1 + l4)/2
        mu_b = (l2 + l3)/2
        
        goals = np.arange(11)
        p_a = poisson.pmf(goals, mu_a)
        p_b = poisson.pmf(goals, mu_b)
        
        cum_b = np.cumsum(p_b)
        win1.append(np.sum(p_a[i] * cum_b[i-1] for i in range(1, 11)))
        win2.append(np.sum(p_b[i] * np.sum(p_a[:i]) for i in range(1, 11)))
        draw.append(np.sum(p_a[:11] * p_b[:11]))
        total.append(mu_a + mu_b)

    # Возвращаем результаты
    return {
        'P1': (float(np.mean(win1)), float(np.std(win1))),
        'Draw': (float(np.mean(draw)), float(np.std(draw))),
        'P2': (float(np.mean(win2)), float(np.std(win2))),
        'Total': (float(np.mean(total)), float(np.std(total))),
        'Lambdas': {
            'Team1_Attack': (lambda1, lambda1_err),
            'Team1_Defense': (lambda2, lambda2_err),
            'Team2_Attack': (lambda3, lambda3_err),
            'Team2_Defense': (lambda4, lambda4_err)
        }
    }

def SuperPrime_Error(a, b, grade_1, grade_2, forma_1, forma_2, n_simulations=100000):
    a -= 1
    b -= 1

    # Проверка валидности индексов и данных
    if (a >= len(scored) or b >= len(skipped) or 
        a < 0 or b < 0 or
        len(scored[a]) == 0 or len(skipped[a]) == 0 or
        len(scored[b]) == 0 or len(skipped[b]) == 0):
        raise ValueError("Некорректные данные команд")

    # Извлекаем данные
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Улучшенный фитинг Пуассона
    def fit_poisson(x_data, y_data, matches):
        valid = y_data > 0
        if sum(valid) == 0:
            return 0.0, 0.0
        x_fit = x_data[valid]
        y_fit = y_data[valid]
        sigma = np.sqrt(y_fit)
        sigma[sigma == 0] = 1e-6
        
        try:
            params, cov = curve_fit(
                lambda x, lmbd: poisson_func(x, lmbd, matches),
                x_fit, y_fit,
                p0=[1.0],
                sigma=sigma,
                absolute_sigma=True
            )
            err = np.sqrt(np.diag(cov))[0]
        except:
            params = [np.sum(x_data * y_data) / matches]
            err = 0.0
        return params[0], err

    # Получаем параметры
    lambda1, lambda1_err = fit_poisson(num, scored_a, matches_per_team[a])
    lambda2, lambda2_err = fit_poisson(num, skipped_a, matches_per_team[a])
    lambda3, lambda3_err = fit_poisson(num, scored_b, matches_per_team[b])
    lambda4, lambda4_err = fit_poisson(num, skipped_b, matches_per_team[b])

    # Корректировка формы (явное применение)
    n_forma = 5
    if forma_1 != 0:
        if forma_1 > 0:
            lambda1 *= (1 + forma_1/n_forma)
            lambda1_err *= (1 + forma_1/n_forma)
        else:
            lambda2 *= (1 - forma_1/n_forma)
            lambda2_err *= (1 - forma_1/n_forma)
    
    if forma_2 != 0:
        if forma_2 > 0:
            lambda3 *= (1 + forma_2/n_forma)
            lambda3_err *= (1 + forma_2/n_forma)
        else:
            lambda4 *= (1 - forma_2/n_forma)
            lambda4_err *= (1 - forma_2/n_forma)

    # Монте-Карло симуляция
    #np.random.seed(42)
    samples = [
        np.maximum(np.random.normal(l, l_err, n_simulations), 1e-9)
        for l, l_err in [(lambda1, lambda1_err), (lambda2, lambda2_err),
                        (lambda3, lambda3_err), (lambda4, lambda4_err)]
    ]

    # Расчет вероятностей
    win1, win2, draw, total = [], [], [], []
    for i in range(n_simulations):
        l1, l2, l3, l4 = samples[0][i], samples[1][i], samples[2][i], samples[3][i]
        
        # Вычисление средних значений
        def calc_mean(g, a, d):
            if g == 0: return min(a, d)
            elif g == 1: return 2*a*d/(a+d+1e-6)
            elif g == 2: return (a*d)**0.5
            elif g == 3: return (a+d)/2
            elif g == 4: return ((a**2 + d**2)/2)**0.5
            else: return max(a, d)
        
        mu1 = calc_mean(grade_1, l1, l4)
        mu2 = calc_mean(grade_2, l3, l2)
        
        # Расчет вероятностей
        max_goals = 11
        g = np.arange(max_goals)
        p1 = poisson.pmf(g, mu1)
        p2 = poisson.pmf(g, mu2)
        
        
        # Вычисление исходов
        win1.append(np.sum([p1[i] * np.sum(p2[:i]) for i in range(1, max_goals)]))
        win2.append(np.sum([p2[i] * np.sum(p1[:i]) for i in range(1, max_goals)]))
        draw.append(np.sum(p1[:11] * p2[:11]))
        total.append(mu1 + mu2)

    # Возврат результатов
    def calc_stats(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        return round(mean, 4), round(std, 4)
    
    return {
        'Win1': calc_stats(win1),
        'Draw': calc_stats(draw),
        'Win2': calc_stats(win2),
        'TotalGoals': calc_stats(total),
        'Lambdas': [
            (round(lambda1, 4), round(lambda1_err, 4)),
            (round(lambda2, 4), round(lambda2_err, 4)),
            (round(lambda3, 4), round(lambda3_err, 4)),
            (round(lambda4, 4), round(lambda4_err, 4))
        ]
    }

def fit_nb(x, r, p, matches):
    """PMF отрицательного биномиального, умноженная на число матчей."""
    return matches * nbinom.pmf(x, r, p)

def Coef_NB(a, b):
    a -= 1; b -= 1

    scored_a  = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b  = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    x_data = np.arange(len(scored_a))

    try:
        # 1) Забитые голы команды a
        (r1, p1), _ = curve_fit(
            lambda x, r, p: fit_nb(x, r, p, matches_per_team[a]),
            x_data, scored_a,
            p0=[10, 0.5],
            bounds=([0,0], [100, 0.999999]),  # p ≤ 0.99
            method='dogbox',
            max_nfev=5000
        )

        # График для scored_a
        plt.figure()
        plt.scatter(x_data, scored_a, label='scored_a')
        plt.plot(x_data,
                 [fit_nb(x, r1, p1, matches_per_team[a]) for x in x_data],
                 label=f'r={r1:.2f}, p={p1:.2f}')
        plt.xlabel('x (голы)')
        plt.ylabel('Частота')
        plt.title('Fit отрицательного биномиального к scored_a')
        plt.legend()
        plt.show()

        # 2) Пропущенные голы команды a
        (r2, p2), _ = curve_fit(
            lambda x, r, p: fit_nb(x, r, p, matches_per_team[a]),
            x_data, skipped_a,
            p0=[10, 0.5],
            bounds=([0,0], [100,0.999999]),
            method='dogbox',
            max_nfev=5000
        )

        plt.figure()
        plt.scatter(x_data, skipped_a, label='skipped_a')
        plt.plot(x_data,
                 [fit_nb(x, r2, p2, matches_per_team[a]) for x in x_data],
                 label=f'r={r2:.2f}, p={p2:.2f}')
        plt.title('Fit отрицательного биномиального к skipped_a')
        plt.legend()
        plt.show()

        # 3) Забитые голы команды b
        (r3, p3), _ = curve_fit(
            lambda x, r, p: fit_nb(x, r, p, matches_per_team[b]),
            x_data, scored_b,
            p0=[10, 0.5],
            bounds=([0,0], [100, 0.999999]),
            method='dogbox',
            max_nfev=5000
        )

        plt.figure()
        plt.scatter(x_data, scored_b, label='scored_b')
        plt.plot(x_data,
                 [fit_nb(x, r3, p3, matches_per_team[b]) for x in x_data],
                 label=f'r={r3:.2f}, p={p3:.2f}')
        plt.title('Fit scored_b')
        plt.legend()
        plt.show()

        # 4) Пропущенные голы команды b
        (r4, p4), _ = curve_fit(
            lambda x, r, p: fit_nb(x, r, p, matches_per_team[b]),
            x_data, skipped_b,
            p0=[10, 0.5],
            bounds=([0,0], [100,0.999999]),
            method='dogbox',
            max_nfev=5000
        )

        plt.figure()
        plt.scatter(x_data, skipped_b, label='skipped_b')
        plt.plot(x_data,
                 [fit_nb(x, r4, p4, matches_per_team[b]) for x in x_data],
                 label=f'r={r4:.2f}, p={p4:.2f}')
        plt.title('Fit skipped_b')
        plt.legend()
        plt.show()

    except RuntimeError as e:
        raise ValueError("Ошибка фитирования параметров") from e

    # Функции вероятностей для полуматчей
    def Pscore1(x):
        total = 0
        for i in range(21):
            arg = 2*x - i
            if arg >= 0:
                total += fit_nb(arg, r1, p1, matches_per_team[a]) * \
                         fit_nb(i,    r4, p4, matches_per_team[b])
        return total / (matches_per_team[a] * matches_per_team[b])

    def Pscore2(x):
        total = 0
        for i in range(21):
            arg = 2*x - i
            if arg >= 0:
                total += fit_nb(arg, r2, p2, matches_per_team[a]) * \
                         fit_nb(i,    r3, p3, matches_per_team[b])
        return total / (matches_per_team[a] * matches_per_team[b])

    # Кумулятивные функции
    def f1(n):
        return sum(Pscore1(i/2) for i in range(int(2*n + 1)))

    def f2(n):
        return sum(Pscore2(i/2) for i in range(int(2*n + 1)))

    # Итоговые вероятности исходов
    Win1 = sum(Pscore1(i/2) * f2(i/2 - 0.5) for i in range(1, 21))
    Win2 = sum(Pscore2(i/2) * f1((i - 1)/2)     for i in range(1, 21))
    draw = sum(Pscore1(i/2) * Pscore2(i/2)       for i in range(0, 21))

    return [Win1, draw, Win2]
from math import factorial
#Абсолютно бесполезное распределение, но возможно когда нибуль применим
def Coef_Bivariate(a, b, lambda_biv=0.1):
    # Приводим индексы к нулевой базе
    a -= 1
    b -= 1

    # Проверка валидности индексов
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a+1} или b={b+1} недопустимы.")

    # Извлекаем данные
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Фитирование параметров Пуассона
    lambda1, _ = curve_fit(lambda x, l: poisson_func(x, l, matches_per_team[a]), num, scored_a, p0=[1])
    lambda2, _ = curve_fit(lambda x, l: poisson_func(x, l, matches_per_team[a]), num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(lambda x, l: poisson_func(x, l, matches_per_team[b]), num, scored_b, p0=[1])
    lambda4, _ = curve_fit(lambda x, l: poisson_func(x, l, matches_per_team[b]), num, skipped_b, p0=[1])

    # Получаем параметры
    lambda1 = lambda1[0]
    lambda2 = lambda2[0]
    lambda3 = lambda3[0]
    lambda4 = lambda4[0]

    # Рассчитываем параметры Bivariate Poisson
    lambda1_bvp = (lambda1 + lambda4) / 2  # Атака команды A + Защита команды B
    lambda2_bvp = (lambda2 + lambda3) / 2  # Защита команды A + Атака команды B

    def bivariate_pmf(x, y):
        """Функция вероятности для Bivariate Poisson"""
        total = 0.0
        min_k = min(x, y)
        for k in range(min_k + 1):
            term = (lambda1_bvp**(x - k) * lambda2_bvp**(y - k) * lambda_biv**k) 
            term /= (factorial(x - k) * factorial(y - k) * factorial(k))
            total += term
        return total * np.exp(-(lambda1_bvp + lambda2_bvp + lambda_biv))

    # Рассчитываем вероятности исходов
    max_goals = 10
    Win1, Draw, Win2 = 0.0, 0.0, 0.0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = bivariate_pmf(i, j)
            if i > j:
                Win1 += prob
            elif i < j:
                Win2 += prob
            else:
                Draw += prob

    # Общее количество голов
    TotalGoals = lambda1_bvp + lambda2_bvp + lambda_biv

    return Win1, Draw, Win2, TotalGoals

from scipy.optimize import minimize
from scipy.stats import poisson

def zip_pmf(x, lambda_, psi):
    """Zero-Inflated Poisson PMF"""
    if x == 0:
        return psi + (1 - psi) * poisson.pmf(0, lambda_)
    else:
        return (1 - psi) * poisson.pmf(x, lambda_)

def fit_zip(data, matches):
    """Функция для подгонки ZIP-распределения"""
    def neg_log_likelihood(params):
        lambda_, psi = params
        log_lik = 0
        for x, count in enumerate(data):
            if count > 0:  # Игнорируем нулевые счетчики для стабильности
                log_lik += count * np.log(zip_pmf(x, lambda_, psi) + 1e-10)
        return -log_lik
    
    # Начальные параметры: lambda = среднее, psi = доля нулей
    init_params = [np.sum(np.arange(len(data)) * data) / matches, 0.2]
    bounds = [(0.01, None), (0, 0.99)]  # lambda > 0, 0 <= psi < 1
    
    result = minimize(neg_log_likelihood, init_params, bounds=bounds)
    return result.x[0], result.x[1]  # lambda, psi

def Coef_ZIP(a, b):
    # Приводим индексы к нулевой базе
    a -= 1
    b -= 1

    # Проверка валидности индексов
    if a >= len(scored) or b >= len(skipped) or a < 0 or b < 0:
        raise ValueError(f"Индексы команд a={a+1} или b={b+1} недопустимы.")

    # Извлекаем данные
    scored_a = np.array([s[1] for s in scored[a]])
    skipped_a = np.array([s[1] for s in skipped[a]])
    scored_b = np.array([s[1] for s in scored[b]])
    skipped_b = np.array([s[1] for s in skipped[b]])

    # Подгонка ZIP для каждой команды
    lambda1, psi1 = fit_zip(scored_a, matches_per_team[a])
    lambda2, psi2 = fit_zip(skipped_a, matches_per_team[a])
    lambda3, psi3 = fit_zip(scored_b, matches_per_team[b])
    lambda4, psi4 = fit_zip(skipped_b, matches_per_team[b])

    # Вероятности голов с учетом Zero-Inflation
    def Pscore1(x):
        p_zip = zip_pmf(x, (lambda1 + lambda4)/2, (psi1 + psi4)/2)
        return p_zip if not np.isnan(p_zip) else 0

    def Pscore2(x):
        p_zip = zip_pmf(x, (lambda2 + lambda3)/2, (psi2 + psi3)/2)
        return p_zip if not np.isnan(p_zip) else 0

    # Вычисление вероятностей исходов
    Win1 = sum(Pscore1(i) * sum(Pscore2(j) for j in range(i)) for i in range(1, 11))
    Win2 = sum(Pscore2(i) * sum(Pscore1(j) for j in range(i)) for i in range(1, 11))
    draw = sum(Pscore1(i) * Pscore2(i) for i in range(11))
    # Общее количество голов (без учета Zero-Inflation)
    TotalGoals = 0.5 * ((lambda1*(1-psi1) + lambda4*(1-psi4)) + 
                        (lambda2*(1-psi2) + lambda3*(1-psi3)))

    return Win1, draw, Win2, TotalGoals, lambda1, psi1, lambda2, psi2, lambda3, psi3, lambda4, psi4
print(Coef_ZIP(1,2))
print(Coef(1,2))