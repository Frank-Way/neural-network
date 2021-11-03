"""
Модуль с описаниями класса, реализующего логику работы основного окна, и классов
, используемых для реализации обучения в отдельном треде, что позволяет
предоствратить "замерзание" GUI
"""
import sys
import traceback
from queue import Queue
from typing import List, Any, Tuple, Callable

import matplotlib
import numpy as np
import sympy
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtWidgets import QMessageBox, QComboBox, QMdiSubWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from matplotlib.figure import Figure

from config import Configuration
from forms.UI_MainWindow import Ui_MainWindow
from losses import MeanSquaredError, SoftmaxCrossEntropy
from operations import Linear, Sigmoid, Tanh, ReLU, LeakyReLU
from optimizers import SGD, SGDMomentum
from utils import show_function, show_function3d

matplotlib.use('Qt5Agg')  # настройка matplotlib на использование движка Qt


class MainWindowSlots(Ui_MainWindow):
    """
    Абстрактный класс, описывающий логику работы основного окна
    """
    # словари для перевода текста в соответствующие классы
    # функции потерь
    _LOSSES = {"MSE": MeanSquaredError,
               "Softmax перекрёстная энтропия": SoftmaxCrossEntropy}
    # оптимизаторы
    _OPTIMIZERS = {"Стохастический градиентный спуск": SGD,
                   "Стохастический градиентный спуск с инерцией": SGDMomentum}
    # способы снижения скорости обучения
    _DECAY_TYPES = {"Линейно": "lin",
                    "Экспоненциально": "exp"}
    # функции активации
    _ACTIVATIONS = {"Линейная": Linear,
                    "Сигмоида": Sigmoid,
                    "Гиперб. тангенс": Tanh,
                    "ReLU": ReLU,
                    "Leaky ReLU": LeakyReLU}
    # декодировщик значений из конфигурации в соответствующий текст
    _DECODER = {"sigmoid": "Сигмоида", "linear": "Линейная",
                "leakyrelu": "Leaky ReLU", "relu": "ReLU",
                "tanh": "Гиперб. тангенс", "mse": "MSE",
                "sce": "Softmax перекрёстная энтропия",
                "sgd": "Стохастический градиентный спуск",
                "sgdm": "Стохастический градиентный спуск с инерцией",
                "lin": "Линейно", "exp": "Экспоненциально"}
    # кодировщик, обратный декодировщику decoder
    _ENCODER = dict(zip(_DECODER.values(), _DECODER.keys()))

    _thread_pool: QThreadPool  # пул тредов для предотвращения "заморзки" GUI
    _results_queue: Queue  # результаты обучения
    _count: int  # кол-во завершившихся тредов (исп-ся для обраб-ки рез-ов)

    _PATH: str  # путь к файлу с конфигурацией
    _config: Configuration  # конфигурация
    _values: dict  # считанные с формы параметры конфигурации

    _inputs: int  # количество входов нейросети
    _layers: int  # количество слоёв нейросети
    _str_expression: str  # строковое представление моделируемой функции
    _function: Any  # моделируемая функция
    _limits: List[List[float]]  # границы входных переменных

    # поля для ввода границ входных значений
    _limit_edits: List[List[QtWidgets.QLineEdit]]
    _neuron_edits: List[QtWidgets.QLineEdit]  # поля для ввода кол-ва нейронов
    _dropout_edits: List[QtWidgets.QLineEdit]  # поля для ввода dropout'а
    # выпадающие списки для выбора функции активации
    _activation_comboboxes: List[QtWidgets.QComboBox]

    def init(self) -> None:
        """
        Создаёт необходимые элементы, загружает значения в
        выпадающие списки, подписывает таблицы
        """
        # загрузка полей для ввода
        self._init_limit_edits()
        self._init_layer_edits_and_comboboxes()

        # установка заголовков таблиц
        self.inputsMinMaxTable. \
            setHorizontalHeaderLabels("x_min;x_max".split(";"))
        self.layersTable. \
            setHorizontalHeaderLabels("Нейронов;"
                                      "Активация;"
                                      "Drop-out".split(";"))

        # загрузка значений в выпадающие списки
        self.lossComboBox.addItems(list(self._LOSSES.keys()))
        self.decayComboBox.addItems(list(self._DECAY_TYPES.keys()))
        self.optimizerComboBox.addItems(list(self._OPTIMIZERS.keys()))

        self._thread_pool = QThreadPool()  # инициализация пула тредов
        """т.к. обучение имеет не thread-safe реализацию, то одновременное 
        выполнение нескольких попыток обучения не возможно. задание 
        максимального числа тредов позволяет перенести ответственность за 
        слежение за количеством выполняемых тредов, организацию тредов в очередь
        и т.д. на сам пул тредов"""
        self._thread_pool.setMaxThreadCount(1)
        self._results_queue = Queue()  # инициализация очереди
        self._count = 0  # инициализация кол-ва завершившихся тредов

        self._limits = None
        self._function = None

    def load(self, path: str) -> None:
        """
        Загрузка конфигуарци и обновление окна в соответствии с ней
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        # загрузка конфигурации
        self._PATH = path
        self._config = Configuration(path)
        self._config.load()

        # считывание и установка количества входов
        self._inputs = self._config.inputs["count"]
        self.inputsSpinBox.setValue(self._inputs)
        # заполнение полей для ввода границ входных значений
        for ii in range(self._inputs):
            for jj in range(2):
                self._limit_edits[ii][jj].setText(
                    str(self._config.inputs["limits"][ii][jj]))
        # заполнение различных полей, выбор нужных значений в выпадающих списках
        self.functionTextEdit.setText(self._config.inputs["function"])
        self.sampleSizeEdit.setText(str(self._config.inputs["size"]))
        self.testSizeEdit.setText(str(self._config.inputs["p_to_test"]))
        self.extendEdit.setText(str(self._config.inputs["p_to_extend"]))
        self.lrEdit.setText(str(self._config.train["lr"]))
        self.lrFinalEdit.setText(str(self._config.train["final_lr"]))
        self._set_combobox_item(self.decayComboBox, "decay_type")
        self.epochsEdit.setText(str(self._config.train["epochs"]))
        self.queryEdit.setText(str(self._config.train["query_times"]))
        self.batchSizeEdit.setText(str(self._config.train["batch_size"]))
        self.stoppingCheckBox.setChecked(self._config.train["early_stopping"])
        self.printCheckBox.setChecked(self._config.train["print_results"])
        self.plotsCheckBox.setChecked(self._config.train["show_plots"])
        self._set_combobox_item(self.lossComboBox, "loss")
        self._set_combobox_item(self.optimizerComboBox, "optimizer")
        self.momentumEdit.setText(str(self._config.train["momentum"]))
        self.restartsSpinBox.setValue(self._config.train["restarts"])
        # считывание и установка количества слоёв
        self._layers = self._config.layers["count"]
        self.layersSpinBox.setValue(self._layers)
        # заполнение полей для ввода количества нейронов, dropout'а и установка
        # значений в выпадающие списки для выбора функции активации
        for ii in range(self._layers):
            self._neuron_edits[ii].setText(
                str(self._config.layers["config"][ii]["neurons"]))
            self._activation_comboboxes[ii]. \
                setCurrentIndex(self._activation_comboboxes[ii].
                                findText(self._DECODER[self._config.
                                         layers["config"][ii]["activation"]]))
            self._dropout_edits[ii].setText(
                str(self._config.layers["config"][ii]["dropout"]))

    def inputs_count_changed(self) -> None:
        """
        Обработка изменения количества входов нейросети.
        Создаются соответствующие поля для ввода границ входных значений,
        изменяется подпись к полю для ввода функции.
        """
        self._limits = None
        self._function = None
        self._init_limit_edits()
        if self._inputs == 1:
            self.inputsMinMaxLabel.setText("F(X1) =")
        elif self._inputs == 2:
            self.inputsMinMaxLabel.setText("F(X1, X2) =")
        else:
            self.inputsMinMaxLabel.setText(f"F(X1, ..., X{self._inputs}) =")

    def layers_count_changes(self) -> None:
        """
        Обработка изменения количества слоёв нейросети.
        Создаются соответствующие поля для ввода количества нейронов и
        dropout'a и выпадающие списки для выбора функции активации
        """
        self._init_layer_edits_and_comboboxes()

    def validate_function(self) -> None:
        """
        Обработчик клика по кнопке проверки функции. Результат проверки
        выводится во всплывающем окне QMessageBox
        """
        self._read_function()  # чтение функции

        valid = self._is_function_valid()  # проверка функции

        if not valid:  # если функция не корректна
            QMessageBox.about(self.centralwidget, "Ошибка",
                              "Не удалось прочитать функцию")
        else:  # есди функция корректна
            QMessageBox.about(self.centralwidget, "Успех",
                              "Функция успешно прочитана")

    def plot_function(self) -> None:
        """
        Построение графика введённой функции в заданных границах
        """
        # считывание количества входов функции
        try:
            self._inputs = self.inputsSpinBox.value()
        except ValueError:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать количество входов")
            return
        # проверка количества входов
        if self._inputs > 2:  # доступны 2- и 3-мерные графики
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Для указанного количества входов "
                                "построение графков не доступно")
            return
        # чтение функции
        if self._function is None:
            self._read_function()
        # чтение границ изменения входных переменных
        try:
            self._read_limits()
        except ValueError:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать границы входных "
                                "переменных")
            return
        # проверка наличия границ
        if self._limits is None:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать границы для "
                                "входных переменных")
            return
        # построение графика функции одной переменной
        if self._inputs == 1:
            fig, _ = show_function(self._function, self._limits)
        # построение графика функции двух переменных
        elif self._inputs == 2:
            fig, _ = show_function3d(self._function, self._limits)
        # отображение графика
        self._plot(fig)

    def run(self) -> None:
        """
        Обработчик клика по кнопке запуска обучения запускает обучение заданное
        количество раз в отдельных последовательно выполняемых тредах.
        """
        self._read()  # чтение данных с формы
        msg = self._validate()  # валидация формы
        if msg is not None:  # если были ошибки
            # выводится сообщение об ошибках
            QMessageBox.warning(self.centralwidget, "Ошибка", msg)
        else:  # если ошибок нет
            # деактивация кнопки запуска обучения
            self.startButton.setEnabled(False)
            # обновление конфигурации в соответствии со считанными данными
            self._config.update(self._values)
            self._config.save()  # сохранение конфигурации
            data = self._config.get_data()  # получение входных данных
            trainer = self._config.get_trainer()  # получение тренера
            # получение аргументов для вызова функции обучения
            fit_params = self._config.get_fit_params(data)

            # цикл по кол-ву перезапусков
            for ii in range(self._config.train["restarts"]):
                # создание рабочего треда для выполнения обучения с заданными
                # параметрами
                worker = _Worker(fn=trainer.fit,  # функция, выполняемая в треде
                                 kwargs=fit_params)  # параметры функции
                # при формировании результата тредом вызывается функция его
                # сохранения
                worker.signals.result.connect(self._save_result)
                # при завершении работы треда вызывается функция обработки
                # результатов его работы
                worker.signals.finished.connect(self._process_results)

                self._thread_pool.start(worker)  # запуск треда

    def clear_output(self) -> None:
        """
        Обработчик клика по кнопке очистки вывода
        """
        self.outputEdit.clear()

    def _init_limit_edits(self) -> None:
        """
        Создание полей для ввода границ входных значений
        """
        self._inputs = int(self.inputsSpinBox.value())  # чтение кол-ва входов
        self.inputsMinMaxTable.clear()  # очистка таблицы
        # установка количества строк таблицы
        self.inputsMinMaxTable.setRowCount(self._inputs)
        # заготовка под список полей для ввода
        self._limit_edits = [[None for jj in range(2)]
                             for ii in range(self._inputs)]
        for ii in range(self._inputs):  # цикл по количеству входов
            for jj in range(2):  # цикл по количеству границ
                # создание и настройка поля для ввода
                edit = QtWidgets.QLineEdit(self.inputsMinMaxTable)
                edit.setObjectName(f"limitEdit{ii + 1}_{jj + 1}")
                edit.setClearButtonEnabled(True)
                edit.setPlaceholderText(str(float(jj)))
                self._limit_edits[ii][jj] = edit  # сохранение поля для ввода
                # добавление поля для ввода в таблицу
                self.inputsMinMaxTable.setCellWidget(ii, jj, edit)

    def _init_layer_edits_and_comboboxes(self) -> None:
        """
        Создание полей для ввода количества нейронов и dropout'a и выпадающих
        списков для выбора функции активации
        """
        self._layers = int(self.layersSpinBox.value())  # чтение кол-ва слоёв
        self.layersTable.clearContents()  # очистка таблицы
        self.layersTable.setRowCount(self._layers)  # установка кол-ва строк
        # заготовка списков для хранения элементов
        self._neuron_edits = []
        self._activation_comboboxes = []
        self._dropout_edits = []
        for ii in range(self._layers):  # цикл по количеству слоёв
            # создание и настройка поля для ввода кол-ва нейронов
            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"neuronEdit{ii + 1}")
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("8")
            self._neuron_edits.append(edit)  # сохранение поля для ввода
            # отображение поля для ввода в таблице
            self.layersTable.setCellWidget(ii, 0, edit)

            # создание и настройка выпадающего списка для выбора активации
            cb = QtWidgets.QComboBox(self.layersTable)
            cb.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
            cb.setObjectName(f"activationComboBox{ii + 1}")
            cb.addItems(list(self._ACTIVATIONS.keys()))
            self._activation_comboboxes.append(cb)  # сохранение списка
            # отображение выпадающего списка в таблице
            self.layersTable.setCellWidget(ii, 1, cb)

            # создание и настройка поля для ввода dropout'а
            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"dropoutEdit{ii + 1}")
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("1.0")
            self._dropout_edits.append(edit)  # сохранение поля для ввода
            # отображение поля для ввода в таблице
            self.layersTable.setCellWidget(ii, 2, edit)

    def _set_combobox_item(self, cb: QComboBox, property_name: str) -> None:
        """
        Установка значения в выпадающем списке в соответствии со значением в
        конфигурауции
        Parameters
        ----------
        cb: Выпадающий список
        property_name: Название параметра
        """
        cb. \
            setCurrentIndex(
                cb.findText(self._DECODER[self._config.train[property_name]]))

    def _read_limits(self) -> None:
        """
        Чтение границ входных переменных
        """
        self._limits = [[None for jj in range(2)] for ii in range(self._inputs)]
        for ii in range(self._inputs):
            for jj in range(2):
                self._limits[ii][jj] = float(self._limit_edits[ii][jj].text())

    def _read_function(self) -> None:
        """
        Чтение функции
        """
        # чтение пользовательского ввода, преобразование к нижнему регистру
        function = str(self.functionTextEdit.toPlainText()).lower()
        # получение символов x1, x2, ..., xn
        input_symbols = [sympy.Symbol(f"x{ii + 1}")
                         for ii in range(self._inputs)]
        # упрощение выражения
        simplified_expression = sympy.simplify(function)
        self._str_expression = str(simplified_expression)
        # получение вызываемой функции
        self._function = sympy.lambdify(input_symbols,
                                        simplified_expression)

    def _is_function_valid(self) -> bool:
        """
        Проверка функции на корректность
        """
        valid = True
        try:  # попытка вычислить значение функции от случайных аргументов
            res = self._function(*(10 * np.random.random(self._inputs) - 5))
        # если функция задана не правильно, function генерирует это исключение
        except NameError:
            valid = False  # функция задана не правильно

        # проверка типа возвращаемого значения
        if valid and type(res) != np.float64:  # если было возвращено не число
            valid = False  # функция задана не правильно

        return valid

    def _read(self) -> None:
        """
        Чтение данных с формы в словарь self.values
        """
        read_failure = False  # пока не было ошибок при чтении
        try:  # попытка чтения
            self._inputs = abs(int(self.inputsSpinBox.value()))
            self._read_function()
            self._read_limits()
            size = abs(int(self.sampleSizeEdit.text()))
            batch_size = min(size, abs(int(self.batchSizeEdit.text())))
            test_size = abs(float(self.testSizeEdit.text()))
            extending = abs(float(self.extendEdit.text()))
            lr = abs(float(self.lrEdit.text()))
            lr_final = min(lr, abs(float(self.lrFinalEdit.text())))
            decay_type = self._ENCODER[self.decayComboBox.currentText()]
            loss = self._ENCODER[self.lossComboBox.currentText()]
            optimizer = self._ENCODER[self.optimizerComboBox.currentText()]
            epochs = abs(int(self.epochsEdit.text()))
            queries = abs(int(self.queryEdit.text()))
            early_stopping = bool(self.stoppingCheckBox.isChecked())
            printing = bool(self.printCheckBox.isChecked())
            plotting = bool(self.plotsCheckBox.isChecked())
            momentum = 1.0
            if optimizer == "sgdm":
                momentum = abs(float(self.momentumEdit.text()))
            restarts = abs(int(self.restartsSpinBox.value()))
            self._layers = abs(int(self.layersSpinBox.value()))
            layers_config = []
            for ii in range(self._layers):
                neurons = abs(int(self._neuron_edits[ii].text()))
                activation = self._ENCODER[self.
                    _activation_comboboxes[ii].currentText()]
                dropout = abs(float(self._dropout_edits[ii].text()))
                layers_config.append((neurons, activation, dropout))
        # если какое-либо поле не было заполнено, то генерируется исключение
        # ValueError, после чего появляется всплывающее окно с сообщением об
        # ошибке, а считанные значения не сохраняются
        except ValueError:
            read_failure = True
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Заполните все поля корректными значениями")
        self._values = None
        if not read_failure:  # если чтение прошло успешно
            # считанные значения заносятся в соответствующий словарь
            self._values = {"inputs": {"count": self._inputs,
                                       "path": "",
                                       "data_loader": "ap",
                                       "function": self._str_expression,
                                       "limits": self._limits,
                                       "size": size,
                                       "scale_inputs": False,
                                       "p_to_test": test_size,
                                       "p_to_extend": extending},
                            "train": {"seed": 0,
                                      "use_seed": False,
                                      "lr": lr,
                                      "final_lr": lr_final,
                                      "decay_type": decay_type,
                                      "epochs": epochs,
                                      "query_times": queries,
                                      "batch_size": batch_size,
                                      "early_stopping": early_stopping,
                                      "print_results": printing,
                                      "show_plots": plotting,
                                      "momentum": momentum,
                                      "loss": loss,
                                      "optimizer": optimizer,
                                      "trainer": "t",
                                      "restarts": restarts},
                            "layers": {"count": self._layers,
                                       "config": [{
                                           "neurons": layers_config[ii][0],
                                           "class": "dense",
                                           "activation": layers_config[ii][1],
                                           "dropout": layers_config[ii][2],
                                           "weight_init": "glorot"
                                       } for ii in range(self._layers)]}}

    def _validate(self) -> str:
        """
        Проверка корректности считанных настроек. Если всё верно, то
        возвращается None, иначе возвращается сообщение об обнаруженных
        ошибках
        Returns
        -------
        str: Сообщение об ошибках (при их наличии)
        """
        message = ""  # сообщение об ошибках
        # если были ошибки при чтении, проверка не выполняется
        if self._values is None:
            return "Чтение параметров не было выполнено"

        # проверка различных значений
        valid_inputs = self._inputs != 0
        message += "" if valid_inputs else "Число входов должно быть" \
                                           " отлично от нуля\n"

        valid_function = self._is_function_valid()
        message += "" if valid_function else "Функция задана неверно\n"

        valid_limits = all(map(lambda limit: limit[0] < limit[1],
                               self._values["inputs"]["limits"]))
        message += "" if valid_limits else "Левая граница для входа должна " \
                                           "быть строго меньше правой\n"

        valid_size = self._values["inputs"]["size"] != 0
        message += "" if valid_size else "Размер выборки должен быть" \
                                         " отличен от нуля\n"

        valid_batch = self._values["train"]["batch_size"] != 0
        message += "" if valid_batch else "Размер пакета должен быть" \
                                          " отличен от нуля\n"

        valid_test = self._values["inputs"]["p_to_test"] != 0.0
        message += "" if valid_test else "Размер тестовой выборки должен " \
                                         "быть отличен от нуля\n"

        valid_lr = 0.0 < self._values["train"]["lr"] <= 2
        message += "" if valid_lr else "Начальная скорость обучения должна " \
                                       "быть в пределах (0; 2]\n"

        valid_final_lr = 0.0 < self._values["train"]["final_lr"] <= 2
        message += "" if valid_final_lr else "Конечная скорость обучения " \
                                             "должна быть в пределах (0; 2]\n"

        valid_decay = self._values["train"]["decay_type"] is not None
        message += "" if valid_decay else "Не выбран способ снижения " \
                                          "скорости обучения\n"

        valid_loss = self._values["train"]["loss"] is not None
        message += "" if valid_loss else "Не выбрана функция потерь\n"

        valid_optimizer = self._values["train"]["optimizer"] is not None
        message += "" if valid_optimizer else "Не выбран оптимизатор\n"

        valid_epochs = self._values["train"]["epochs"] > 1
        message += "" if valid_epochs else "Количество эпох обучения " \
                                           "должно быть больше 1\n"

        valid_queries = self._values["train"]["query_times"] > 0
        message += "" if valid_queries else "Недостаточное количество эпох " \
                                            "обучения для реализации " \
                                            "заданного числа опросов\n"

        valid_momentum = True
        if self._values["train"]["optimizer"] == "sgdm":
            valid_momentum = 0.0 < self._values["train"]["momentum"] <= 1.0
            message += "" if valid_momentum else "Инертность должна быть" \
                                                 " в пределах (0; 1]\n"

        valid_restarts = self._values["train"]["restarts"] != 0
        message += "" if valid_restarts else "Количество запусков должно " \
                                             "быть больше 0"

        valid_layers = self._layers != 0
        message += "" if valid_layers else "Число слоёв должно быть" \
                                           " отлично от нуля\n"

        valid_layers_config_neurons = all(map(lambda cfg:
                                              cfg["neurons"] != 0,
                                              self._values["layers"]["config"]))
        if not valid_layers_config_neurons:
            message += "Число нейронов должно быть отлично от нуля\n"

        valid_layers_config_activations = all(
            map(lambda cfg: cfg["activation"] is not None,
                self._values["layers"]["config"]))
        if not valid_layers_config_activations:
            message += "Не выбрана функция активации\n"

        valid_layers_config_dropout = all(
            map(lambda cfg: 0.0 < cfg["dropout"] <= 1.0,
                self._values["layers"]["config"]))
        if not valid_layers_config_dropout:
            message += "Drop-out должен быть в пределах (0; 1]\n"

        # объединение всех проверок по И
        valid = all((valid_restarts, valid_queries, valid_epochs, valid_inputs,
                     valid_size, valid_lr,
                     valid_test, valid_layers, valid_final_lr, valid_function,
                     valid_optimizer, valid_momentum, valid_limits, valid_loss,
                     valid_layers_config_neurons, valid_layers_config_dropout,
                     valid_layers_config_activations, valid_batch, valid_decay))
        if not valid:  # при наличии ошибок
            return message  # возвращается сообщение об ошибках

    def _save_result(self, result: Tuple) -> None:
        """
        Сохранение результата обучения в очередь
        Parameters
        ----------
        result: Результат обучения
        """
        self._results_queue.put_nowait(result)

    def _plot(self, fig: Figure) -> None:
        """
        Отображение графика в окне MDI-виджета
        Parameters
        ----------
        fig: График
        """
        # создание дочернего окна под график
        sub = QMdiSubWindow(self.mdiArea)
        canvas = FigureCanvasQTAgg(fig)  # создание обёртки для графика
        toolbar = NavigationToolbar2QT(canvas, sub)  # создание панели
        sub.layout().addWidget(toolbar)  # добавление навигационной панели
        sub.layout().addWidget(canvas)  # добавление обёртки с графиком
        sub.show()  # отображение дочернего окна

    def _process_results(self) -> None:
        """
        Обработка результатов опирается на переменную self.count, отвечающей за
        количество завершившихся тредов. При каждом вызове функции происходит
        инкрементирование переменной. Когда её значение сравняется с заданым
        количеством перезапуском обучения, то переменная сбрасывается и
        происходит обработка результатов. Если значение переменной меньше числа
        перезапусков, обработка не выполняется.
        """
        self._count += 1  # увеличение кол-ва завершившихся тредов
        # если завершились не все треды
        if self._count < self._config.train["restarts"]:
            return  # обработки не происходит
        # когда завершились все треды
        self._count = 0  # сбрасывается счётчик
        # выбор лучшего результата
        best_result = self._results_queue.get(timeout=1.0)
        while self._results_queue.qsize() > 0:
            result = self._results_queue.get()
            if result[1] < best_result[1]:
                best_result = result
        # вывод лучшего результата
        print(self._config.get_str_results(best_result))  # текстом
        graph_results = self._config.get_graph_results(best_result)  # графиками
        # config.get_graph_results возвращает None, если графики не требуются
        if graph_results is not None:
            graph_function, graph_params = graph_results
            fig, _ = graph_function(**graph_params)  # получение фигуры
            self._plot(fig)  # отображение графика
        self.startButton.setEnabled(True)  # разблок-ка кнопки запуска обучения


class _WorkerSignals(QObject):
    """
    Класс для описания сигналов, генерируемых тредом, выполняющего обучение
    """
    finished = pyqtSignal()  # признак завершения работы треда
    error = pyqtSignal(tuple)  # ошибка при работе треда
    result = pyqtSignal(object)  # результат работы треда


class _Worker(QRunnable):
    """
    Класс-оболочка для QThread, реализующий вызов заданной функции с заданными
    аргументами
    """

    def __init__(self, fn: Callable, kwargs: dict):
        """
        Конструктор треда
        Parameters
        ----------
        fn: Функция, выполняемая в треде
        kwargs: Аргументы функции
        """
        super(_Worker, self).__init__()  # вызов конструктора QRunnable()
        self.fn = fn  # сохранение функции
        self.kwargs = kwargs  # сохранение аргументов
        self.signals = _WorkerSignals()  # инициализация сигналов

    @pyqtSlot()  # run - Qt-слот
    def run(self) -> None:
        """
        Запуск функции
        """
        try:  # попытка запуска функции
            result = self.fn(**self.kwargs)  # получение результата
        except (NameError, ValueError, TypeError):  # при исключении
            traceback.print_exc()  # вывод трассировки
            # получение параметров исключения
            exctype, value = sys.exc_info()[:2]
            # формирование сигнала ошибки
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:  # если ошибки не возникло
            self.signals.result.emit(result)  # формируется сигнал с результатом
        finally:  # в любом случае
            self.signals.finished.emit()  # формируется сигнал завершения работы
