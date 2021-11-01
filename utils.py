"""
Модуль с различными вспомогательными функциями для выполнения проверок,
обработки и форматирования данных, отрисовки графиков.
"""
import os
from typing import List, Tuple

from PIL import Image
import numpy as np
from numpy import ndarray
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logsumexp


mpl.use('Qt5Agg')


def assert_same_shape(a: ndarray, b: ndarray):
    """
    Функция проверки совпадения форм массивов
    Parameters
    ----------
    a: Первый массив
    b: Второй массив
    """
    assert a.shape == b.shape, \
        "The shapes of the given arrays do not match:\n" \
        f"first array's shape is {tuple(a.shape)} and " \
        f"second array's shape if {tuple(b.shape)}"


def permute_data(a: ndarray,
                 b: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Функция перемешивания двух массивов
    Parameters
    ----------
    a: Первый массив
    b: Второй массив
    Returns
    -------
    Tuple[ndarray]: Перемешанные массивы
    """
    perm = np.random.permutation(a.shape[0])
    return a[perm], b[perm]


def complete_path(path: str) -> str:
    """
    Функция формирования полного пути
    Parameters
    ----------
    path: Путь
    Returns
    -------
    str: Полный путь
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, path)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename


def normalize(a: ndarray) -> ndarray:
    """
    Функция дополняет массив веротностей массивом
    обратных вероятностей q = 1 - p
    Parameters
    ----------
    a: Исходный массив вероятностей
    Returns
    -------
    ndarray: Массив исходных и обратных вероятностей
    """
    other = 1.0 - a
    return np.concatenate([a, other], axis=1)


def unnormalize(a: np.ndarray) -> ndarray:
    """
    Функция возвращает массив исходных вероятностей из массива
    вероятностей, дополненных обратными с помощью функции normalize
    Parameters
    ----------
    a: Массив исходных и обратных вероятностей
    Returns
    -------
    ndarray: Массив исходных вероятностей
    """
    return a[np.newaxis, 0]


def to_2d(a: np.ndarray,
          array_type: str = "col") -> np.ndarray:
    """
    Функция формирования 2-мерного массива из 1-мерного
    Parameters
    ----------
    a: 1-мерный массив
    array_type: Тип 1-мерного массива (строка или столбец)
    Returns
    -------
    ndarray: 2-мерный массив
    """

    if a.ndim == 1:
        if array_type == "col":
            return a.reshape(-1, 1)
        elif array_type == "row":
            return a.reshape(1, -1)
    else:
        return a


def softmax(x: ndarray, axis: int = None) -> ndarray:
    """
    Функция применения Softmax к массиву по указанной оси
    Parameters
    ----------
    x: Входной массив
    axis: Номер оси
    Returns
    -------
    ndarray: Softmax(x)
    """
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def load_image(filename: str, scaler: MinMaxScaler = None) -> ndarray:
    """
    Функция загрузки изображения в градациях серого с приведением его
    к одномерному массиву с диапозоном значений [0; 1]
    Parameters
    ----------
    filename: Путь к файлу с изображением
    scaler: Скейлер для приведения значений к диапозону [0; 1]
    Returns
    -------
    ndarray: Одномерный массив, представляющий изображение в градациях серого
    """
    img = Image.open(filename).convert("L")
    img.load()
    data = np.asarray(img, dtype="double").flatten()
    data = data.reshape((data.size, 1))
    if scaler:
        data = scaler.fit_transform(data)
    return data


def calc_accuracy_model(model,
                        test_set: ndarray,
                        y_test: ndarray) -> str:
    """
    Функция оценки точности модели в количестве правильно предсказанных меток
    Parameters
    ----------
    model: Модель (нейросеть)
    test_set: Набор тестовых входных данных
    y_test: Набор тестовых выходных данных
    Returns
    -------
    str: Точность работы модели
    """
    pred_test = model.forward(test_set, inference=True)
    accuracy = np.equal(np.argmax(pred_test, axis=1),
                        np.argmax(y_test, axis=1)).sum()
    accuracy = accuracy * 100.0 / test_set.shape[0]
    return f"Точность модели: {accuracy:.2f}%"


def batches_generator(x: ndarray,
                      y: ndarray,
                      size: int = 32):
    """
    Генератор пакетов для обучения
    Parameters
    ----------
    x: Входы
    y: Требуемые выходы
    size: Размер пакета
    """
    assert x.shape[0] == y.shape[0], f"""
    Входы и выходы должны иметь одинаковое число строк, но входы имеют
     {x.shape[0]} строк, а выходы - {y.shape[0]}
    """

    n = x.shape[0]
    size = min(size, n)

    for ii in range(0, n, size):
        x_batch, y_batch = x[ii:ii+size], y[ii:ii+size]
        yield x_batch, y_batch


def show_results(losses: List[float],
                 x_test: ndarray,
                 pred_test: ndarray,
                 y_test: ndarray,
                 function_name: str,
                 neurons: List[int]):
    """
    Функия для отрисовки графика по результатам обучения нейросети, 
    аппроксимирующей математические функции одной переменной
    Parameters
    ----------
    losses: Список потерь
    x_test: Массив входных значений
    pred_test: Массив выходных значений
    y_test: Массив требуемых выходных значений
    function_name: Название функции
    neurons: Количество нейронов
    Returns
    -------
    None
    """
    x = x_test
    y = pred_test
    t = y_test
    e = t - y

    fig = plt.figure()
    ax1 = plt.subplot(122)
    ax1.plot(x, y, label='модель')
    ax1.plot(x, t, label='функция')
    ax1.set_title(f'воспроизведено с помощью {neurons[:-1]} нейронов')
    ax1.set(xlabel='x', ylabel=function_name)
    ax1.legend()
    ax1.grid()

    ax2 = plt.subplot(221)
    ax2.plot(range(len(losses)), losses)
    ax2.set_title('функция потерь')
    ax2.set(xlabel='опросы', ylabel='потеря')
    ax2.grid()

    ax3 = plt.subplot(223)
    ax3.plot(x, e)
    ax3.set_title('макс. абс. ошибка')
    ax3.set(xlabel='x', ylabel='MAE(x)')
    ax3.grid()

    return fig, (ax1, ax2, ax3)


def show_results3d(losses: List[float],
                   x_test: ndarray,
                   pred_test: ndarray,
                   y_test: ndarray,
                   function_name: str,
                   neurons: List[int]):
    """
    Функия для отрисовки графика по результатам обучения нейросети,
    аппроксимирующей математические функции одной переменной
    Parameters
    ----------
    losses: Список потерь
    x_test: Массив входных значений
    pred_test: Массив выходных значений
    y_test: Массив требуемых выходных значений
    function_name: Название функции
    neurons: Количество нейронов
    Returns
    -------
    None
    """

    x = x_test
    y = pred_test
    t = y_test
    e = t - y

    fig = plt.figure()
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter3D(x[:, 0], x[:, 1], y, label='модель')
    ax1.scatter3D(x[:, 0], x[:, 1], t, label='функция')
    ax1.set_title(f'воспроизведено с помощью {neurons[:-1]} нейронов')
    ax1.set(xlabel="x1", ylabel="x2", zlabel=f"{function_name}")
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(221)
    ax2.plot(range(len(losses)), losses)
    ax2.set_title('функция потерь')
    ax2.set(xlabel='опросы', ylabel='потеря')
    ax2.grid()

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter3D(x[:, 0], x[:, 1], e)
    ax3.set_title('макс. абс. ошибка')
    ax3.set(xlabel="x1", ylabel="x2", zlabel="MAE(x1, x2)")
    ax3.grid()

    return fig, (ax1, ax2, ax3)


def mnist_labels_to_y(labels: ndarray) -> ndarray:
    """
    Функция преобразования метки в набор набор признаков
    Parameters
    ----------
    labels: Массив меток
    Returns
    -------
    ndarray: Массив признаков
    """
    labels -= np.min(labels)
    classes = int(np.max(labels) + 1)
    y = np.zeros((labels.size, classes))
    for (ii, label) in enumerate(labels):
        y[ii][label] = 1
    return y


def replace_chars(a: str) -> str:
    """
    Функция для удаления пробелов и замены '/' на ':' в строках
    Parameters
    ----------
    a: Входная строка
    Returns
    -------
    str: Выходная строка с заменёнными символами
    """
    return a.replace("/", ":").replace(" ", "")


def cartesian(arrays: Tuple[ndarray, ...]) -> ndarray:
    """
    Вычисление декартового произведения массивов
    Parameters
    ----------
    arrays: 1-D массивы
    Returns
    -------
    ndarray: 2-D массив формы (M, len(arrays)) с декартовым произведением
             исходных массивов
    Examples
    --------
    >>> cartesian((np.asarray([1, 2, 3]), np.asarray([4, 5]), np.asarray([6, 7])))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j * m: (j + 1) * m, k + 1:] = out[0: m, k + 1:]
    return out
