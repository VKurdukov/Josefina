import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import poisson

# Загружаем данные из Excel (каждый лист соответствует команде)
excel_file = pd.ExcelFile(r"C:\Users\Владимир\Desktop\Josefina\LaLiga1.xlsx")

# Параметры
Matches = 25
num = np.arange(0, 10)

# Создаем списки для scored и skipped
scored = []
skipped = []

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

# Функция Пуассона для подгонки
def poisson_func(x, lambd):
    return Matches * poisson.pmf(x, lambd)

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
    lambda1, _ = curve_fit(poisson_func, num, scored_a, p0=[1])
    lambda2, _ = curve_fit(poisson_func, num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(poisson_func, num, scored_b, p0=[1])
    lambda4, _ = curve_fit(poisson_func, num, skipped_b, p0=[1])

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

    return Win1, draw, Win2, 1/Win1, 1/draw, 1/Win2, TotalGoals, Mmatrix

# Пример использования функции
print(Coef(19,16))


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
    lambda1, _ = curve_fit(poisson_func, num, scored_a, p0=[1])
    lambda2, _ = curve_fit(poisson_func, num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(poisson_func, num, scored_b, p0=[1])
    lambda4, _ = curve_fit(poisson_func, num, skipped_b, p0=[1])

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

    return Win1, draw, Win2, 1/Win1, 1/draw, 1/Win2, TotalGoals, Mmatrix
#print(Coef_fans(5,5,0))

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
    lambda1, _ = curve_fit(poisson_func, num, scored_a, p0=[1])
    lambda2, _ = curve_fit(poisson_func, num, skipped_a, p0=[1])
    lambda3, _ = curve_fit(poisson_func, num, scored_b, p0=[1])
    lambda4, _ = curve_fit(poisson_func, num, skipped_b, p0=[1])

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
    if w1 < 1.6 and w1 * Win1 < 1:
        Win1 = 1.1/w1
        Win2 = (1-Win1)*Win2/(Win2+draw)
        draw = (1-Win1-Win2)
    elif w2 < 1.6 and w2 * Win2 < 1:
        Win2 = 1.1/w2
        Win1 = (1-Win2)*Win1/(Win1+draw)
        draw = (1-Win1-Win2)
    

    TotalGoals = float(0.5 * (lambda1 + lambda2 + lambda3 + lambda4))

    # Матрица вероятностей счета
    Mmatrix = np.array([[Pscore1(i) * Pscore2(j) for j in range(7)] for i in range(7)])

    return Win1, draw, Win2, 1/Win1, 1/draw, 1/Win2, TotalGoals, Mmatrix