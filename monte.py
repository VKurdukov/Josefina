import numpy as np
from scipy.stats import poisson
import time

# Функция для моделирования голов на основе распределения Пуассона
def simulate_goals(lambda_goals, minutes):
    goals = 0
    for _ in range(minutes):
        # Вероятность гола на текущей минуте
        if np.random.rand() < poisson.pmf(1, lambda_goals / minutes):
            goals += 1
    return goals

# Функция для симуляции матча с учётом атаки и обороны
def simulate_match(lambda1, lambda2, lambda3, lambda4, match_duration=90):
    # Рассчитываем итоговые lambda для каждой команды
    lambda_a = (lambda1 + lambda4) / 2  # Атака команды A + оборона команды B
    lambda_b = (lambda3 + lambda2) / 2  # Атака команды B + оборона команды A
    
    # Симулируем головы для каждой команды
    goals_a = simulate_goals(lambda_a, match_duration)
    goals_b = simulate_goals(lambda_b, match_duration)
    
    # Возвращаем итоговый счёт
    return goals_a, goals_b

# Функция для симуляции матча с выводом счёта на каждой минуте
def simulate_match_live(lambda1, lambda2, lambda3, lambda4, match_duration=90):
    # Рассчитываем итоговые lambda для каждой команды
    lambda_a = (lambda1 + lambda4) / 2  # Атака команды A + оборона команды B
    lambda_b = (lambda3 + lambda2) / 2  # Атака команды B + оборона команды A
    
    goals_a, goals_b = 0, 0
    for minute in range(1, match_duration + 1):
        # Симулируем голы на текущей минуте
        if np.random.rand() < poisson.pmf(1, lambda_a / match_duration):
            goals_a += 1
        if np.random.rand() < poisson.pmf(1, lambda_b / match_duration):
            goals_b += 1
        
        # Выводим счёт на текущей минуте
        print(f"Минута {minute}: {goals_a} - {goals_b}")
        time.sleep(0)
    
    return goals_a, goals_b

# Пример использования
lambda1 = 2.5482283913265746 # Среднее количество голов, которые забивает команда A
lambda2 = 1.5744154578585354# Среднее количество голов, которые пропускает команда A
lambda3 = 1.2750430874221201  # Среднее количество голов, которые забивает команда B
lambda4 = 3.308641684179461 # Среднее количество голов, которые пропускает команда B

# Симуляция матча с выводом счёта на каждой минуте
final_score = simulate_match_live(lambda1, lambda2, lambda3, lambda4)
print(f"Итоговый счёт: {final_score[0]} - {final_score[1]}")