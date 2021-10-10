import json

import numpy as np


def save(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=6)


functions = [
    ('cos(pi * x ^ (1/2))', lambda x: np.cos(np.pi * np.sqrt(x)), (0.0, 0.25)),
    ('tan(pi * x / 2)', lambda x: np.tan(np.pi * x / 2.0), (0.0, 0.5)),
    ('2 ^ -x', lambda x: np.power(2.0, -x), (0.0, 1.0)),
    ('arcsin(x * 2 ^ (1/2))', lambda x: np.arcsin(x * np.sqrt(2)), (0.0, 0.595)),
    ('arctan(2 * x)', lambda x: np.arctan(2.0 * x), (0.0, 0.779)),
    ('ln(x)', lambda x: np.log(x), (1.0, 2.718)),
    ('log[2](1 + x)', lambda x: np.log2(1.0 + x), (0.0, 1.0)),
    ('sinh(x)', lambda x: np.sinh(x), (0.0, 0.881)),
    ('log[10](x)', lambda x: np.log10(x), (1.0, 10.0)),
    ('e ^ (-x)', lambda x: np.exp(-x), (0.0, 2.718)),
    ('x ^ (1/2)', lambda x: np.sqrt(x), (0.0, 1.0)),
    ('sin(pi * x)', lambda x: np.sin(np.pi * x), (0.0, 1.0)),
    ('ln(x) / ln(3)', lambda x: np.log(x) / np.log(3.0), (1.0, 3.0)),
    ('x ^ 2', lambda x: np.power(x, 2.0), (0.0, 1.0))
]

ls = 0.01  # скорость обучения

# funs = ["tan(pi*x:2)", "x^(1:2)", "sin(pi*x)", "2^-x", "ln(x)", "e^(-x)"]
# funs = ["tan(pi*x:2)", "x^(1:2)"]
# funs = ["sin(pi*x)", "2^-x"]
funs = ["ln(x)", "e^(-x)"]
sizes = [512, 1024, 2048]
nns = [8, 16, 32]
epochs = [5000, 15000, 30000]

res = {}
for f in funs:
    res[f] = {}
    for s in sizes:
        res[f][s] = {}
        for n in nns:
            res[f][s][n] = {}
            for e in epochs:
                res[f][s][n][e] = 0

for kk, fun in enumerate(funs):
    for fun_tuple in functions:
        if fun == fun_tuple[0].replace("/", ":").replace(" ", ""):
            fun_name, function, limits = fun_tuple[0], fun_tuple[1], fun_tuple[2]
            break

    x_min, x_max = limits[0], limits[1]
    for neurons in nns:
        # задание массива смещений
        b_min = -1
        b_max = 1
        B = np.linspace(b_min, b_max, num=neurons)

        neuron_step = (x_max - x_min) / (neurons - 1)
        for sample_size in sizes:
            X = np.linspace(x_min, x_max, num=sample_size)  # входы
            T = function(X)  # теоретические выходы
            Y = np.zeros(sample_size)  # выходы
            A = np.zeros(neurons)  # выходы 1 - го слоя
            S = np.zeros(neurons)  # сумма
            for max_epoch in epochs:
                trained_flag = False  # признак конца обучения

                sigmoid_coefficient = 0.2  # коэффициент сигмоиды(крутизна)
                target_max_delta = 0.0005  # требуемая максимальная АБСОЛЮТНАЯ ПОГРЕШНОСТЬ

                max_epoch_reached_flag = False  # признак завершения обучения(по числу эпох)

                trained_flag_array = [False for _ in range(sample_size)]  # сброс признаков обучения

                # Задание весов
                W = np.random.random(neurons)
                epoch = 0
                while not (trained_flag or max_epoch_reached_flag):
                    for i in range(sample_size):
                        Y[i] = 0.0
                        for j in range(neurons):
                            S[j] = X[i]
                            A[j] = 1.0 / (1.0 + np.exp(-(S[j] - B[j]) / sigmoid_coefficient))
                            Y[i] += W[j] * A[j]

                        delta = T[i] - Y[i]
                        delta_abs = np.abs(delta)
                        if delta_abs < target_max_delta:  # абсолютная погрешность (1.0 - 100 % )
                            trained_flag_array[i] = True
                        else:
                            # Добавление всех входов к весам
                            trained_flag = False
                            trained_flag_array = [False for _ in range(sample_size)]

                            for j in range(neurons):
                                if j * neuron_step <= X[i] <= (j + 1) * neuron_step:
                                    W[j] += ls * delta * X[i]
                        trained_flag = all(trained_flag_array)

                    epoch = epoch + 1
                    max_epoch_reached_flag = epoch == max_epoch

                for i in range(sample_size):
                    Y[i] = 0.0
                    for j in range(neurons):
                        S[j] = X[i]
                        A[j] = 1.0 / (1.0 + np.exp(-(S[j] - B[j]) / sigmoid_coefficient))
                        Y[i] += W[j] * A[j]
                delta_max = np.max(np.abs(T - Y))

                res[fun][sample_size][neurons][max_epoch] = delta_max
                save(f"{fun}.json", res)
                print(f"dumped {fun}, {neurons}, {sample_size}, {max_epoch}")

            save(f"{fun}.json", res)
            print(f"dumped {fun}, {neurons}, {sample_size}")

        save(f"{fun}.json", res)
        print(f"dumped {fun}, {neurons}")

    save(f"{fun}.json", res)
    print(f"dumped {fun}")

save(f"../../tmp.json", res)
print("dumped everything")



