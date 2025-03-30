import numpy as np
import matplotlib.pyplot as plt
import math

def compute_Z(lambda_, nu, max_terms=1000):
    """Вычисляет нормирующую константу Z(lambda, nu)."""
    Z = 0.0
    for j in range(max_terms):
        try:
            term = (lambda_ ** j) / (math.factorial(j) ** nu)
            Z += term
            if term < 1e-10:  # Остановка, если слагаемое становится очень маленьким
                break
        except OverflowError:
            # Если возникает переполнение, пропускаем это слагаемое
            break
    return Z

def compute_mean(lambda_, nu, max_terms=1000):
    """Вычисляет математическое ожидание для CMP распределения."""
    Z = compute_Z(lambda_, nu, max_terms)
    mean = 0.0
    for j in range(1, max_terms):  # Начинаем с j=1, так как j=0 дает 0
        try:
            term = (j * (lambda_ ** j)) / ((math.factorial(j)) ** nu)
            mean += term
            if term < 1e-10:  # Остановка, если слагаемое становится очень маленьким
                break
        except OverflowError:
            # Если возникает переполнение, пропускаем это слагаемое
            break
    return mean / Z

def compute_variance(lambda_, nu, max_terms=1000):
    """Вычисляет дисперсию для CMP распределения."""
    Z = compute_Z(lambda_, nu, max_terms)
    mean = compute_mean(lambda_, nu, max_terms)
    variance = 0.0
    for j in range(1, max_terms):  # Начинаем с j=1, так как j=0 дает 0
        try:
            term = (j ** 2 * (lambda_ ** j)) / ((math.factorial(j)) ** nu)
            variance += term
            if term < 1e-10:  # Остановка, если слагаемое становится очень маленьким
                break
        except OverflowError:
            # Если возникает переполнение, пропускаем это слагаемое
            break
    variance = variance / Z - mean ** 2
    return variance

# Диапазон значений lambda и nu
lambda_values = np.arange(0.1, 10.1, 0.1)  # От 0.1 до 10 с шагом 0.1
nu_values = np.arange(0.1, 5.1, 0.1)       # От 0.1 до 5 с шагом 0.1

# Фиксированные значения для построения графиков
fixed_nu = 1.5  # Фиксированное значение nu для графиков от lambda
fixed_lambda = 2.0  # Фиксированное значение lambda для графиков от nu

# Вычисляем математическое ожидание и дисперсию
mean_lambda = [compute_mean(lambda_, fixed_nu) for lambda_ in lambda_values]
variance_lambda = [compute_variance(lambda_, fixed_nu) for lambda_ in lambda_values]

mean_nu = [compute_mean(fixed_lambda, nu) for nu in nu_values]
variance_nu = [compute_variance(fixed_lambda, nu) for nu in nu_values]

# Построение графиков
plt.figure(figsize=(14, 10))

# Математическое ожидание от lambda
plt.subplot(2, 2, 1)
plt.plot(lambda_values, mean_lambda, marker='', linestyle='-', color='b')
plt.xlabel('λ (lambda)')
plt.ylabel('Математическое ожидание')
plt.title(f'Математическое ожидание от λ (ν = {fixed_nu})')
plt.grid(True)

# Математическое ожидание от nu
plt.subplot(2, 2, 2)
plt.plot(nu_values, mean_nu, marker='', linestyle='-', color='r')
plt.xlabel('ν (nu)')
plt.ylabel('Математическое ожидание')
plt.title(f'Математическое ожидание от ν (λ = {fixed_lambda})')
plt.grid(True)

# Дисперсия от lambda
plt.subplot(2, 2, 3)
plt.plot(lambda_values, variance_lambda, marker='', linestyle='-', color='g')
plt.xlabel('λ (lambda)')
plt.ylabel('Дисперсия')
plt.title(f'Дисперсия от λ (ν = {fixed_nu})')
plt.grid(True)

# Дисперсия от nu
plt.subplot(2, 2, 4)
plt.plot(nu_values, variance_nu, marker='', linestyle='-', color='m')
plt.xlabel('ν (nu)')
plt.ylabel('Дисперсия')
plt.title(f'Дисперсия от ν (λ = {fixed_lambda})')
plt.grid(True)

# Показать графики
plt.tight_layout()
plt.show()