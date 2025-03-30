import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import curve_fit

# Функции для вычисления коэффициентов
def Z(lmbda, nu):
    return np.sum([lmbda**n / gamma(n + 1)**nu for n in range(11)])

def MaxwellPoissonDist(x, lmbda, nu):
    x_int = np.round(x).astype(int)  # Округление до целого числа для массива
    return (lmbda**x_int / gamma(x_int + 1)**nu) / Z(lmbda, nu)

def fit_function(x, lmbda, nu, Matches):
    return Matches * MaxwellPoissonDist(x, lmbda, nu)

def Coef_Max(scored_a, skipped_a, scored_b, skipped_b):
    """Основная функция для вычисления коэффициентов."""
    # Количество матчей (может быть разным для команд)
    Matches = np.sum(scored_a)  # Используем сумму забитых голов как количество матчей

    # Фитирование параметров распределения
    x_data = np.arange(len(scored_a))
    try:
        # Фитирование для забитых голов команды a
        params1, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, Matches), 
            x_data, scored_a, p0=[2, 2]
        )
        lmbda1, nu1 = params1

        # Фитирование для пропущенных голов команды a
        params2, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, Matches), 
            x_data, skipped_a, p0=[2, 2]
        )
        lmbda2, nu2 = params2

        # Фитирование для забитых голов команды b
        params3, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, Matches), 
            x_data, scored_b, p0=[2, 2]
        )
        lmbda3, nu3 = params3

        # Фитирование для пропущенных голов команды b
        params4, _ = curve_fit(
            lambda x, l, v: fit_function(x, l, v, Matches), 
            x_data, skipped_b, p0=[2, 2]
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
                total += fit_function(arg, lmbda1, nu1, Matches) * fit_function(i, lmbda4, nu4, Matches)
        return total / Matches**2

    def Pscore2(x):
        """Вероятность для команды 2."""
        total = 0
        for i in range(21):
            arg = 2 * x - i
            if arg >= 0:
                total += fit_function(arg, lmbda2, nu2, Matches) * fit_function(i, lmbda3, nu3, Matches)
        return total / Matches**2

    # Кумулятивные функции
    def f1(n):  # Целое число
        return sum(Pscore1(i / 2) for i in range(int(2 * n + 1)))

    def f2(n):
        return sum(Pscore2(i / 2) for i in range(int(2 * n + 1)))

    # Вычисление исходов
    Win1 = sum(Pscore1(i / 2) * f2(i/2 - 0.5) for i in range(1, 21))  # i: 1-20
    Win2 = sum(Pscore2(i / 2) * f1((i - 1) / 2) for i in range(1, 21))
    draw = sum(Pscore1(i / 2) * Pscore2(i / 2) for i in range(0, 21))  # i: 0-20

    # Возвращаем результаты
    return [Win1, draw, Win2]

# Основной алгоритм
def process_matches(df, debug=False):
    # Словари для хранения данных о командах
    teams = {}
    for team in pd.concat([df['Home Team'], df['Away Team']]).unique():
        teams[team] = {
            'scored': np.zeros(21),  # Распределение забитых голов (0-20)
            'skipped': np.zeros(21),  # Распределение пропущенных голов (0-20)
            'matches_played': 0  # Количество сыгранных матчей
        }

    # Переменная для хранения итогового результата
    itog = 0

    # Список для хранения отладочной информации
    debug_info = []

    # Проходим по всем матчам
    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = row['Home Goals']
        away_goals = row['Away Goals']
        result = row['Result']
        odds_home = row['Bet365 Home Win']
        odds_draw = row['Bet365 Draw']
        odds_away = row['Bet365 Away Win']

        # Проверяем, что обе команды сыграли больше 5 матчей до текущего
        if teams[home_team]['matches_played'] > 10 and teams[away_team]['matches_played'] > 10:
            # Получаем распределения голов до текущего матча
            scored_a = teams[home_team]['scored'].copy()
            skipped_a = teams[home_team]['skipped'].copy()
            scored_b = teams[away_team]['scored'].copy()
            skipped_b = teams[away_team]['skipped'].copy()

            try:
                # Вычисляем коэффициенты
                win1, draw_prob, win2 = Coef_Max(scored_a, skipped_a, scored_b, skipped_b)

                # Умножаем на коэффициенты букмекеров
                value_home = win1 * odds_home
                value_draw = draw_prob * odds_draw
                value_away = win2 * odds_away

                # Выбираем максимальное значение
                max_value = max(value_home, value_draw, value_away)

                # Обновляем итоговый результат
                if max_value == value_home:
                    if result == 'H':
                        itog += odds_home - 1
                    else:
                        itog -= 1
                elif max_value == value_draw:
                    if result == 'D':
                        itog += odds_draw - 1
                    else:
                        itog -= 1
                elif max_value == value_away:
                    if result == 'A':
                        itog += odds_away - 1
                    else:
                        itog -= 1

                # Сохраняем отладочную информацию
                if debug:
                    debug_info.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'win1': win1,
                        'draw': draw_prob,
                        'win2': win2,
                        'scored_a': scored_a,
                        'skipped_a': skipped_a,
                        'scored_b': scored_b,
                        'skipped_b': skipped_b
                    })
                    # Ограничиваем длину списка до последних 10 матчей
                    if len(debug_info) > 10:
                        debug_info.pop(0)
            except ValueError:
                # Пропускаем матч, если фитирование не удалось
                continue

        # Обновляем данные о командах после текущего матча
        teams[home_team]['scored'][home_goals] += 1
        teams[home_team]['skipped'][away_goals] += 1
        teams[home_team]['matches_played'] += 1

        teams[away_team]['scored'][away_goals] += 1
        teams[away_team]['skipped'][home_goals] += 1
        teams[away_team]['matches_played'] += 1

    return itog, debug_info

# Загрузка данных из Excel
def analyze_sheet(sheet_name, debug=False):
    df = pd.read_excel('eredev_seasons_coef.xlsx', sheet_name=sheet_name)
    itog, debug_info = process_matches(df, debug=debug)
    print(f"Итог для листа {sheet_name}: {itog}")
    return itog, debug_info

# Пример использования
if __name__ == '__main__':
    # Список листов для анализа
    sheets = ['2024-2025','2023-2024','2022-2023','2021-2022','2020-2021']  # Добавьте все нужные листы

    # Анализ каждого листа
    total_itog = 0
    debug_info = []  # Инициализируем переменную для отладочной информации
    for sheet in sheets:
        if sheet == '2024-2025':  # Только для первого листа добавляем отладочную информацию
            itog, debug_info = analyze_sheet(sheet, debug=True)
        else:
            itog, _ = analyze_sheet(sheet)
        total_itog += itog

    print(f"Общий итог: {total_itog}")

    # Вывод отладочной информации для последних 10 матчей
    if debug_info:  # Проверяем, что отладочная информация есть
        print("\nОтладочная информация для последних 10 матчей:")
        for match in debug_info:
            print(f"Матч: {match['home_team']} vs {match['away_team']}")
            print(f"Win1: {match['win1']:.4f}, Draw: {match['draw']:.4f}, Win2: {match['win2']:.4f}")
            print(f"scored_a: {match['scored_a']}")
            print(f"skipped_a: {match['skipped_a']}")
            print(f"scored_b: {match['scored_b']}")
            print(f"skipped_b: {match['skipped_b']}")
            print("-" * 50)
    else:
        print("\nОтладочная информация отсутствует.")