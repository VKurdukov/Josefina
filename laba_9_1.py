import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные
V = np.array([0.1, 0.14, 0.23, 0.28, 0.35, 0.43, 0.54, 0.67, 0.77, 0.84, 0.86,
              -0.87, -0.79, -0.65, -0.58, -0.5, -0.46, -0.4, -0.34, -0.29, -0.24, -0.19, -0.14, -0.11, -0.08, -0.02, 0.02])

f = np.array([944.518, 948.4, 949.7, 950.48, 951.3, 952.851, 953.247, 953.027, 953.253, 953.472, 953.466,
              910.196, 910.617, 910.734, 910.827, 911.037, 911.111, 911.309, 911.847, 912.740, 913.784,
              916.079, 919.873, 922.672, 925.649, 932.754, 937.108])

f0 = np.array([955.668, 955.457, 955.710, 955.452, 955.507, 955.516, 955.475, 955.538, 955.7, 955.548, 955.524,
               955.6, 955.614, 955.431, 955.688, 955.517, 955.651, 955.663, 957.637, 955.520, 955.563, 955.562,
               955.576, 955.768, 955.808, 955.837, 955.474])

# Параметры
T_room = 22  # Комнатная температура в °C
thermocouple_coeff = 0.041  # Коэффициент термопары в В/°C
V_shift = 0.00016  # Сдвиг нуля в В
delta_V = 0.00002  # Погрешность измерения напряжения в В

# Перевод V в T
T = T_room + (V - V_shift) / thermocouple_coeff

# Погрешность температуры
delta_T = delta_V / thermocouple_coeff

# Вычисляем X = f^2 / (f0^2 - f^2)
X = f**2 / (f0**2 - f**2)

# Удаляем точки, которые находятся в пределах +-0.3 градуса от 25.5, 32.5, 35, 40.7
T_to_remove = [25.5, 32.5, 35, 40.7]
mask = np.ones_like(T, dtype=bool)  # Изначально все точки включены

for t in T_to_remove:
    mask &= (T < t - 0.3) | (T > t + 0.3)  # Исключаем точки в пределах +-0.3 градуса

# Применяем маску
T = T[mask]
X = X[mask]
delta_T = delta_T  # Погрешность температуры остается той же

# Погрешность X (фиксированная)
delta_X = np.full_like(X, 0.01)

# Определим линейный участок (визуально примерно от -0.35 В до 0.5 В)
linear_indices = (V[mask] >= -0.05) & (V[mask] <= 0.9)
X_linear = X[linear_indices]
T_linear = T[linear_indices]

# Линейная аппроксимация
def linear_fit(x, a, b):
    return a * x + b

# Аппроксимируем X от T
params, covariance = curve_fit(linear_fit, T_linear, X_linear)
a_fit, b_fit = params

# Построение графика
plt.figure(figsize=(8, 6))
plt.errorbar(T, X, xerr=delta_T, yerr=delta_X, fmt='o', markersize=1, label='Экспериментальные данные', capsize=3)
plt.plot(T_linear, linear_fit(T_linear, a_fit, b_fit), 'r-', label=f'Линейная аппроксимация: X = {a_fit:.2f} * T + {b_fit:.2f}')

plt.xlabel('Температура (°C)')
plt.ylabel(r'$X = \frac{f^2}{f_0^2 - f^2}$')
plt.title('Зависимость X от T и линейная аппроксимация')
plt.legend()
plt.grid(True)

plt.show()

# Вывод коэффициентов аппроксимации
a_fit, b_fit


# Применяем маску к f0, чтобы удалить те же точки, что и для T и X
f0_masked = f0[mask]

# Построение графика f0 от T
plt.figure(figsize=(8, 6))
plt.errorbar(T, f0_masked, xerr=delta_T, fmt='o', markersize=1, label='Экспериментальные данные', capsize=3)

plt.xlabel('Температура (°C)')
plt.ylabel(r'$f_0$ (Гц)')
plt.title('Зависимость $f_0$ от температуры')
plt.legend()
plt.grid(True)

plt.show()