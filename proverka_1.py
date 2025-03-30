import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import curve_fit

# Функция для вычисления коэффициентов
def poisson_func(x, lambd):
    return poisson.pmf(x, lambd)

def Coef(scored_a, skipped_a, scored_b, skipped_b, forma_1, forma_2):
    """Основная функция для вычисления коэффициентов."""
    # Количество матчей (может быть разным для команд)
    Matches = np.sum(scored_a)  # Используем сумму забитых голов как количество матчей

    # Находим параметры распределения для команд a и b
    num = np.arange(len(scored_a))  # Массив индексов для фитирования
    try:
        lambda1, _ = curve_fit(poisson_func, num, scored_a, p0=[1])
        lambda2, _ = curve_fit(poisson_func, num, skipped_a, p0=[1])
        lambda3, _ = curve_fit(poisson_func, num, scored_b, p0=[1])
        lambda4, _ = curve_fit(poisson_func, num, skipped_b, p0=[1])
    except RuntimeError:
        # Если фитирование не удалось, используем средние значения
        lambda1, lambda2, lambda3, lambda4 = [1.0, 1.0, 1.0, 1.0]

    lambda1, lambda2, lambda3, lambda4 = lambda1[0], lambda2[0], lambda3[0], lambda4[0]

    # Используем форму команд для корректировки параметров
    n_forma = 5
    if forma_1 > 0:
        lambda1 = lambda1 * (1 + forma_1 / n_forma)
    elif forma_1 < 0:
        lambda2 = lambda2 * (1 - forma_1 / n_forma)

    if forma_2 > 0:
        lambda3 = lambda3 * (1 + forma_2 / n_forma)
    elif forma_2 < 0:
        lambda4 = lambda4 * (1 - forma_2 / n_forma)

    # Вероятности для забитых и пропущенных голов
    Pscore1 = lambda x: poisson.pmf(x, (lambda1 + lambda4) / 2)
    Pscore2 = lambda x: poisson.pmf(x, (lambda2 + lambda3) / 2)

    # Вычисляем шансы на победу, ничью и количество голов
    f1 = lambda n: np.sum([Pscore1(i) for i in range(n + 1)])
    f2 = lambda n: np.sum([Pscore2(i) for i in range(n + 1)])

    Win1 = float(np.sum([Pscore1(i) * f2(i - 1) for i in range(1, 11) if i - 1 >= 0]))
    Win2 = float(np.sum([Pscore2(i) * f1(i - 1) for i in range(1, 11) if i - 1 >= 0]))
    draw = float(np.sum([Pscore1(i) * Pscore2(i) for i in range(11)]))

    # Выводим параметры lambda для отладки
    #print(f"lambda1: {lambda1:.4f}, lambda2: {lambda2:.4f}, lambda3: {lambda3:.4f}, lambda4: {lambda4:.4f}")

    return Win1, draw, Win2

# Основной алгоритм
def process_matches(df, debug=False):
    # Словари для хранения данных о командах
    teams = {}
    for team in pd.concat([df['Home Team'], df['Away Team']]).unique():
        teams[team] = {
            'scored': np.zeros(21),  # Распределение забитых голов (0-20)
            'skipped': np.zeros(21),  # Распределение пропущенных голов (0-20)
            'matches_played': 0,  # Количество сыгранных матчей
            'last_results': []  # Список последних результатов (1, 0, -1)
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

        # Проверяем, что обе команды сыграли больше 10 матчей до текущего
        if teams[home_team]['matches_played'] >= 10 and teams[away_team]['matches_played'] >= 10:
            # Получаем распределения голов до текущего матча (исключая текущий матч)
            scored_a = teams[home_team]['scored'].copy()
            skipped_a = teams[home_team]['skipped'].copy()
            scored_b = teams[away_team]['scored'].copy()
            skipped_b = teams[away_team]['skipped'].copy()

            # Вычисляем форму команд (исключая текущий матч)
            forma_1 = sum(teams[home_team]['last_results'][-10:])  # Форма домашней команды
            forma_2 = sum(teams[away_team]['last_results'][-10:])  # Форма гостевой команды

            try:
                # Вычисляем коэффициенты
                win1, draw_prob, win2 = Coef(scored_a, skipped_a, scored_b, skipped_b, forma_1, forma_2)

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
                        'skipped_b': skipped_b,
                        'forma_1': forma_1,
                        'forma_2': forma_2
                    })
            except ValueError:
                # Пропускаем матч, если фитирование не удалось
                continue

        # Обновляем данные о командах после текущего матча
        if teams[home_team]['matches_played'] >= 10:
            # Удаляем самый старый матч, если сыграно больше 10 матчей
            oldest_goals = np.argmax(teams[home_team]['scored'] > 0)
            teams[home_team]['scored'][oldest_goals] -= 1
            teams[home_team]['skipped'][np.argmax(teams[home_team]['skipped'] > 0)] -= 1
            teams[home_team]['last_results'].pop(0)

        teams[home_team]['scored'][home_goals] += 1
        teams[home_team]['skipped'][away_goals] += 1
        teams[home_team]['matches_played'] += 1
        teams[home_team]['last_results'].append(1 if result == 'H' else (0 if result == 'D' else -1))

        if teams[away_team]['matches_played'] >= 10:
            # Удаляем самый старый матч, если сыграно больше 10 матчей
            oldest_goals = np.argmax(teams[away_team]['scored'] > 0)
            teams[away_team]['scored'][oldest_goals] -= 1
            teams[away_team]['skipped'][np.argmax(teams[away_team]['skipped'] > 0)] -= 1
            teams[away_team]['last_results'].pop(0)

        teams[away_team]['scored'][away_goals] += 1
        teams[away_team]['skipped'][home_goals] += 1
        teams[away_team]['matches_played'] += 1
        teams[away_team]['last_results'].append(1 if result == 'A' else (0 if result == 'D' else -1))

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
    sheets = ['2024-2025', '2023-2024','2022-2023','2021-2022','2020-2021']  # Добавьте все нужные листы

    # Анализ каждого листа
    total_itog = 0
    debug_info = []  # Инициализируем переменную для отладочной информации
    for sheet in sheets:
        if sheet == '2024-2025':  # Только для первого листа добавляем отладочную информацию
            itog, debug_info = analyze_sheet(sheet, debug=False)
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
            print(f"Форма домашней команды: {match['forma_1']}")
            print(f"Форма гостевой команды: {match['forma_2']}")
            print("-" * 50)
    else:
        print("\nОтладочная информация отсутствует.")