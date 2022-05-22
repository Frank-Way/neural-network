"""
Модуль с описаниями класса, реализующего логику работы основного окна, и классов
, используемых для реализации обучения в отдельном треде, что позволяет
предоствратить "замерзание" GUI
"""
import sys
import traceback
from copy import deepcopy
from queue import Queue
from typing import List, Any, Tuple, Callable

import matplotlib
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtWidgets import QMessageBox, QComboBox, QMdiSubWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from matplotlib.figure import Figure

from config import ConfigHandler, InputsConfig, TrainConfig, LayersConfig, LayerConfig, Config, ConfigUtils
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
    _models_queue: Queue  # обученные модели
    _count: int  # кол-во завершившихся тредов (исп-ся для обраб-ки рез-ов)

    _PATH: str  # путь к файлу с конфигурацией
    _config_handler: ConfigHandler  # обработчик конфигураций
    _config: Config  # конфигурация
    _unvalidated_config: Config  # непроверенная конфигурация
    _best_model: Any  # лучшая обученная модель

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

        self._thread_pool = QThreadPool().globalInstance()  # инициализация пула
        self._results_queue = Queue()  # инициализация очереди для результатов
        self._models_queue = Queue()  # инициализация очереди для моделей
        self._count = 0  # инициализация кол-ва завершившихся тредов

        self._best_model = None  # инициализация модели

    def load(self, path: str) -> None:
        """
        Загрузка конфигуарци и обновление окна в соответствии с ней
        Parameters
        ----------
        path: str
            Путь к файлу с настройками
        """
        # загрузка конфигурации
        self._PATH = path
        config = Config.load(path)
        self._config_handler = ConfigHandler(path, config)

        # считывание и установка количества входов
        inputs = config.inputs.count
        layers = config.layers.count
        self.inputsSpinBox.setValue(inputs)
        # заполнение полей для ввода границ входных значений
        for ii in range(inputs):
            for jj in range(2):
                self._limit_edits[ii][jj].setText(
                    str(config.inputs.limits[ii][jj]))
        # заполнение различных полей, выбор нужных значений в выпадающих списках
        self.functionTextEdit.setText(config.inputs.function)
        self.sampleSizeEdit.setText(str(config.inputs.size))
        self.testSizeEdit.setText(str(config.inputs.p_to_test))
        self.extendEdit.setText(str(config.inputs.p_to_extend))
        self.lrEdit.setText(str(config.train.lr))
        self.lrFinalEdit.setText(str(config.train.final_lr))
        self._set_combobox_item(self.decayComboBox, "decay_type")
        self.epochsEdit.setText(str(config.train.epochs))
        self.queryEdit.setText(str(config.train.query_times))
        self.batchSizeEdit.setText(str(config.train.batch_size))
        self.stoppingCheckBox.setChecked(config.train.early_stopping)
        self.printCheckBox.setChecked(config.train.print_results)
        self.plotsCheckBox.setChecked(config.train.show_plots)
        self._set_combobox_item(self.lossComboBox, "loss")
        self._set_combobox_item(self.optimizerComboBox, "optimizer")
        self.momentumEdit.setText(str(config.train.momentum))
        self.restartsSpinBox.setValue(config.train.restarts)
        # считывание и установка количества слоёв

        self.layersSpinBox.setValue(layers)
        # заполнение полей для ввода количества нейронов, dropout'а и установка
        # значений в выпадающие списки для выбора функции активации
        for ii in range(layers):
            self._neuron_edits[ii].setText(
                str(config.layers.layers_config[ii].neurons))
            self._activation_comboboxes[ii]. \
                setCurrentIndex(self._activation_comboboxes[ii].
                                findText(self._DECODER[config.layers.layers_config[ii].activation]))
            self._dropout_edits[ii].setText(
                str(config.layers.layers_config[ii].dropout))

    def inputs_count_changed(self) -> None:
        """
        Обработка изменения количества входов нейросети.
        Создаются соответствующие поля для ввода границ входных значений,
        изменяется подпись к полю для ввода функции.
        """
        inputs = self._read_inputs_count()
        if inputs is None: return
        self._init_limit_edits()
        if inputs == 1:
            self.functionLabel.setText("F(X1) =")
            self.inputsMinMaxLabel.setText("Границы X1")
        elif inputs == 2:
            self.functionLabel.setText("F(X1, X2) =")
            self.inputsMinMaxLabel.setText("Границы X1, X2")
        else:
            self.functionLabel.setText(f"F(X1, ..., X{inputs}) =")
            self.inputsMinMaxLabel.setText(f"Границы X1, ..., X{inputs}")

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
        inputs = self._read_inputs_count()
        if inputs is None: return
        _, fn = self._read_function()
        if fn is None: return
        valid = ConfigUtils.is_function_valid(fn, inputs)
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
        inputs = self._read_inputs_count()
        if inputs is None: return 
        simplified_function, fn = self._read_function()  # чтение функции
        if (simplified_function is None) or (fn is None): return
        # чтение границ изменения входных переменных
        limits = self._read_limits()
        if inputs == 1:  # построение графика функции одной переменной
            figs_axs = show_function(simplified_function, fn, limits)
        elif inputs == 2:  # построение графика функции двух переменных
            figs_axs = show_function3d(simplified_function, fn, limits)
        else:
            QMessageBox.information(self.centralwidget, "Внимание",
                                    "Для указанного количества входов "
                                    "построение графков доступно только при "
                                    "X1=X2=...=Xn\nГраницы при этом будут "
                                    "взяты для X1")
            for ii in range(2, inputs + 1):
                simplified_function = simplified_function.replace(f"x{ii}", "x1")
            fn = ConfigUtils.lambidify_function(simplified_function, 1)
            if (simplified_function is None) or (fn is None): return
            figs_axs = show_function(simplified_function, fn, limits)
        for fig, _ in figs_axs:
            self._plot(fig)  # отображение графика

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
            return
        # если ошибок нет
        # деактивация кнопки запуска обучения
        self.startButton.setEnabled(False)
        self.exportButton.setEnabled(False)
        if self._config_handler is None:
            self._config_handler = ConfigHandler(
                self._PATH, deepcopy(self._unvalidated_config))
        else:
            self._config_handler.config = deepcopy(self._unvalidated_config)
        self._config_handler.config.save(self._PATH)
        data = self._config_handler.get_data()  # получение входных данных
        self.progressLabel.setText(f"{self._count} / "
                                   f"{self._config_handler.config.train.restarts}")
        # цикл по кол-ву перезапусков
        for ii in range(self._config_handler.config.train.restarts):
            # создание копии конфигуарции для запуска обучения в нескольких
            # потоках
            cfg = deepcopy(self._config_handler)
            trainer = cfg.get_trainer()  # получение тренера
            # получение аргументов для вызова функции обучения
            fit_params = cfg.get_fit_params(data)
            self._models_queue.put_nowait(cfg.model)  # сохранение модели
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

    def export_model(self) -> None:
        """
        Экспорт модели в окно вывода
        """
        if self._best_model is None:
            QMessageBox.warning(self.centralwidget,
                                "Ошибка", "Нет обученной модели для экспорта")
            return
        self._config_handler.model = self._best_model
        print(self._config_handler.export_model())

    def tabs_help(self) -> None:
        """
        Обработчик клика по справке о назначении вкладок
        """
        msg = "Вкладка 'Настройки' используется для задания различных настроек" \
              " (входные данные, параметры обучения, конфигурация сети).\n" \
              "Вклдадка 'Вывод' используется для вывода результатов обучения" \
              " в виде текста.\n" \
              "Вкладка 'Графики' используется для вывода результатов обучения" \
              " в виде графиков (если для указанного количества входов" \
              " возможно построение графиков). На вкладке могут быть открыты" \
              " одновременно несколько разных графиков, график являются" \
              " интерактивными."
        QMessageBox.about(self.centralwidget, "Помощь", msg)

    def buttons_help(self) -> None:
        """
        Обработчик клика по справке о назначении кнопок
        """
        msg = "Кнопка 'Запуск' используется для запуска обучения нейронной" \
              " сети с указанными настройками. Перед запуском обучения" \
              " необходимо задать настройки обучения.\n" \
              "Кнопка 'Проверка' используется для проверки введённой" \
              " функции на корректность. Проверяется соответствие количества" \
              " введённых переменных, осуществляется попытка вычисления" \
              " функции от случайных выходов. Подробности см. в справке о" \
              " программе.\n" \
              "Кнопка 'График' используется для построения графика введённой" \
              " функции в заданных границах входных переменных. Графики могут" \
              " быть построены, если число входов равно 1 или 2. Перед" \
              " построением графика необходимо заполнить границы входных" \
              " переменных.\n" \
              "Кнопка 'Экспорт' используется для экспорта обученной нейронной" \
              " сети на вкладку 'Вывод' в текстовом виде. Экспорт возможен" \
              " только при наличии обученной модели (т.е. сначала необходимо" \
              " запустить обучение).\n" \
              "Кнопка 'Выход' используется для выхода из приложения.\n" \
              "Кнопка 'Очистить' используется для очистки поля на вкладке" \
              " 'Вывод'."
        QMessageBox.about(self.centralwidget, "Помощь", msg)

    def guide_help(self) -> None:
        """
        Обработчик клика по справке с инструкцией
        """
        msg = "Рекомендуемая последовательность работы с приложением:\n" \
              "0) ознакомиться со справками 'О настройках', 'О назначении" \
              " вкладок', 'О назначении кнопок';\n" \
              "1) установить количество входных переменных;\n" \
              "2) ввести функцию;\n" \
              "3) проверить введённую функцию (кнопка 'Проверка')," \
              " исправить при необходимости;\n" \
              "4) заполнить левые и правые границы изменения входных" \
              " переменных (внимание - левая граница должна быть строго меньше" \
              " правой);\n" \
              "5) *опционально* построить график функции (кнопка 'График')," \
              " ознакомиться с графиком (на вкладке 'Графики');\n" \
              "6) заполнить другие поля;\n" \
              "7) выбрать способ снижения скорости обучения, способ" \
              " оптимизации, функцию потерь;\n" \
              "8) установить количество слоёв нейронной сети;\n" \
              "9) задать настройки для каждого слоя:\n" \
              "9.1) установить количество нейронов в слое;\n" \
              "9.2) выбрать функцию активации;\n" \
              "9.3) устновить вероятность исключения нейрона (drop-out);\n" \
              "10) запустить обучение;\n" \
              "11) ознакомиться с результатами обучения (вкладки 'Вывод' и," \
              " опционально, 'Графики');\n" \
              "12) завершить работу с приложением или повторить обучение с" \
              " новыми параметрами.\n" \
              "Примечание - на длительность обучения влияет размер выборки," \
              " размер тестовой выборки, количество эпох обучения, количество" \
              " и размеры слоёв. Если занимает слишком много времени," \
              " то уменьшите один или несколько из вышеприведённых параметров."
        QMessageBox.about(self.centralwidget, "Помощь", msg)

    def about_help(self) -> None:
        """
        Обработчик клика по справке о программе
        """
        msg = "Программа разработана в рамках выпускной квалификационной" \
              " работы студентом четвёртого курса бакалавриата Рязанского" \
              " государственного радиотехнического университета им. В. Ф." \
              " Уткина факультета автоматики и информационных технологий в" \
              " управлении кафедры автоматизированных систем управления" \
              " направления 'Информационные системы и технологии' 09.03.02" \
              " группы 835 Анашкиным В. С.\n\n" \
              "Назначение программы - обучение нейронных сетей произвольной" \
              " структуры для воспроизведения математических зависимостей" \
              " любого количества входных переменных.\n" \
              "Подробнее о процессе задания функциональной зависимости для" \
              " воспроизведения:" \
              " https://docs.sympy.org/latest/modules/utilities/lambdify.html"
        QMessageBox.about(self.centralwidget, "Помощь", msg)

    def settings_help(self) -> None:
        """
        Обработчик клика по справке о настройках
        """
        msg = "Назначения настроек:\n" \
              "  Кол-во входов - количество входов воспроизводимой функции" \
              " и обучаемой модели;\n" \
              "  F(...) = - воспроизводимая функция (примечание - переменные" \
              " обозначаются 'Xi', где i изменяется от 1 до n, где n -" \
              " количество входов функции);\n" \
              "  Ограничения входов - набор левых и правых границ изменения" \
              " каждой из входных переменных, начиная с X1;\n" \
              "  Размер выборки - количество отсчётов, которые будут взяты" \
              " из каждого диапазона входных переменных (если для трёх входов" \
              " установить размер выборки 100, то резульитрующая выборка" \
              " будет представлять собой декартово произведение размером" \
              " 100*100*100=1000000; обучение такой выборки займёт бОльшее" \
              " время и даст лучший результат);\n" \
              "  Тестовая часть - часть от выборки, которая будет взята для" \
              " тестов и не будет представляться при обучении (например," \
              " при размере выборки 100 и тестовой части 0.3 для тестов будут" \
              " взяты 30 дополнительных отсчётов из диапозона входных значений" \
              ", на которых и будет выполняться проверка точности" \
              " воспроизведения зависимости;\n" \
              "  Расширение - часть, на которую будет расширена обучающая" \
              " выборка (например, диапазон [0; 1] при расширении 0.2 будет" \
              " расширен до диапазона [-0.2; 1.2]), что позволяет значительно" \
              " снизить ошибку воспроизведения зависимости на границах" \
              " диапазонов входных значений;\n" \
              "  Нач. скор. - начальная скорость обучения;\n" \
              "  Кон. скор. - конечная скорость обучения;\n" \
              "  Снижение - способ снижения скорости обучения (постепенное" \
              " снижение скорости позволяет ускорить процесс обучения и" \
              " повысить точность за счёт быстрых корректировок в начале" \
              " обучения и более точных корректировок в конце обучения;\n" \
              "  Эпох - количество эпох обучения (чем больше эпох, тем" \
              " дольше обучение и лучше результат);\n" \
              "  Опросов - количество тестовых опросов сети (оценка её" \
              " точности);\n" \
              "  Пакет - размер пакета при обучении (выборка разбивается на" \
              " пакеты, на каждом из которых сеть обучается независимо)," \
              " не стоит задавать большие размеры пакетов;\n" \
              "  Предв. ост. - флаг предварительной остановки (если да, то" \
              " при обучении будут отслеживаться ситуации, когда точность" \
              " начинает снижаться, и тогда обучение будет прекращено);\n" \
              "  Пром. рез. - флаг вывода промежуточных результатов обучения" \
              " в тестовом виде;\n" \
              "  Графики - флаг построения графиков о результатах обучения" \
              " (построение графиков возможно для функций 1 и 2 переменных;\n" \
              "  Потеря - функция потерь (для задач регрессии более" \
              " подходящей является среднеквадратичная функция потерь MSE);\n" \
              "  Оптимиз. - оптимизатор (обычный градиентный спуск или" \
              " градиентный спуск с инерцией);" \
              "  Инерция - величина инертности, используемая градиентным" \
              " спуском с инерцией (использование инерции позволяет в" \
              " некоторых случаях обходить недостаток градиентного спуска," \
              " связанный с локальными минимумами); чем выше, тем выше" \
              " инертность;\n" \
              "  Количество слоёв - количество слоёв нейронной сети, для задач" \
              " аппроксимации необходимо минимум 2 слоя, причём выходной слой" \
              " должен иметь 1 нейрон и линейную функцию активации (в таком" \
              " случае слой выполняет роль сумматора сигналов, полученных на" \
              " предыдущем слое;\n" \
              "  Нейронов - количество нейронов в слое;\n" \
              "  Активация - функция активации слоя;\n" \
              "  Drop-out - вероятность того, что нейрон не будет исключен" \
              " при очередной итерации обучения (для выходного слоя" \
              " рекомендуется задавать 1, для скрытых слоёв при количестве" \
              " слоёв >2 - от 0.5 до 1.0)"
        QMessageBox.about(self.centralwidget, "Помощь", msg)

    def _init_limit_edits(self) -> None:
        """
        Создание полей для ввода границ входных значений
        """
        inputs = self._read_inputs_count()
        if inputs is None: return 
        self.inputsMinMaxTable.clear()  # очистка таблицы
        # установка количества строк таблицы
        self.inputsMinMaxTable.setRowCount(inputs)
        # заготовка под список полей для ввода
        self._limit_edits = [[None for jj in range(2)]
                             for ii in range(inputs)]
        for ii in range(inputs):  # цикл по количеству входов
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
        layers = self._read_layers_count()
        if layers is None: return 
        self.layersTable.clearContents()  # очистка таблицы
        self.layersTable.setRowCount(layers)  # установка кол-ва строк
        # заготовка списков для хранения элементов
        self._neuron_edits = []
        self._activation_comboboxes = []
        self._dropout_edits = []
        for ii in range(layers):  # цикл по количеству слоёв
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
        cb: QComboBox
            Выпадающий список
        property_name: str
            Название параметра
        """
        cb. \
            setCurrentIndex(
                cb.findText(self._DECODER[self._config_handler.config.train.__getattribute__(property_name)]))

    def _read_limits(self) -> List[List[float]]:
        """
        Чтение границ входных переменных
        """
        inputs = self._read_inputs_count()
        if inputs is None: return 
        limits = [[None for jj in range(2)] for ii in range(inputs)]
        for ii in range(inputs):
            for jj in range(2):
                try:
                    limits[ii][jj] = float(self._limit_edits[ii][jj].text())
                except ValueError:
                    return
        return limits
    
    def _read_inputs_count(self) -> int:
        try:
            inputs = int(self.inputsSpinBox.value())
            if inputs < 1:
                raise ValueError
            return inputs
        except ValueError:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать количество входов")
    
    def _read_layers_count(self) -> int:
        try:
            layers = int(self.layersSpinBox.value())
            if layers < 1:
                raise ValueError
            return layers
        except ValueError:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать количество слоёв")


    def _read_function(self) -> Tuple[str, Callable]:
        simplified_function, fn = None, None
        try:
            str_function = str(self.functionTextEdit.toPlainText()).lower()
            inputs = self._read_inputs_count()
        except ValueError:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось считать количество входов или функцию")
        try:
            simplified_function = ConfigUtils.simplify_function(str_function)
            fn = ConfigUtils.lambidify_function(simplified_function, inputs)
        except Exception:
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Не удалось интерпретировать функцию")
        return simplified_function, fn


    def _read(self) -> None:
        """
        Чтение данных с формы в словарь self.values
        """
        read_failure = False  # пока не было ошибок при чтении
        try:  # попытка чтения
            inputs = self._read_inputs_count()
            if inputs is None: return
            str_function = str(self.functionTextEdit.toPlainText()).lower()
            limits = self._read_limits()
            size = abs(int(self.sampleSizeEdit.text()))
            batch_size = min(size, abs(int(self.batchSizeEdit.text())))
            test_size = abs(float(self.testSizeEdit.text()))
            extending = abs(float(self.extendEdit.text()))
            inputs_config = InputsConfig(
                inputs, "", "ap", ConfigUtils.simplify_function(str_function),
                limits, size, False, test_size, extending)

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
            train_config = TrainConfig(
                0, False, lr, lr_final, decay_type, epochs, queries, batch_size,
                early_stopping, printing, plotting, momentum, loss, optimizer,
                "t", restarts)

            layers = self._read_layers_count()
            if layers is None: return
            layer_config_list = []
            for ii in range(layers):
                neurons = abs(int(self._neuron_edits[ii].text()))
                activation = \
                    self._ENCODER[self._activation_comboboxes[ii].currentText()]
                dropout = abs(float(self._dropout_edits[ii].text()))
                layer_config_list.append(LayerConfig(neurons, "dense",
                                                     activation, dropout, "glorot"))
            layers_config = LayersConfig(layers, layer_config_list)
        # если какое-либо поле не было заполнено, то генерируется исключение
        # ValueError, после чего появляется всплывающее окно с сообщением об
        # ошибке, а считанные значения не сохраняются
        except ValueError:
            read_failure = True
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Заполните все поля корректными значениями")
        if not read_failure:  # если чтение прошло успешно
            # считанные значения запоминаются
            self._unvalidated_config = Config(
                inputs=inputs_config, train=train_config, layers=layers_config)

    def _validate(self) -> str:
        """
        Проверка корректности считанных настроек. Если всё верно, то
        возвращается None, иначе возвращается сообщение об обнаруженных
        ошибках
        Returns
        -------
        str
            Сообщение об ошибках (при их наличии)
        """
        return ConfigUtils.validate(self._unvalidated_config)

    def _save_result(self, result: Tuple) -> None:
        """
        Сохранение результата обучения в очередь
        Parameters
        ----------
        result: Tuple
            Результат обучения
        """
        self._results_queue.put_nowait(result)

    def _plot(self, fig: Figure) -> None:
        """
        Отображение графика в окне MDI-виджета
        Parameters
        ----------
        fig: Figure
            График
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
        self.progressLabel.setText(f"{self._count} / "
                                   f"{self._config_handler.config.train.restarts}")
        # если завершились не все треды
        if self._count < self._config_handler.config.train.restarts:
            return  # обработки не происходит
        # когда завершились все треды
        self._count = 0  # сбрасывается счётчик
        # выбор лучшего результата
        best_result = self._results_queue.get(timeout=1.0)
        self._best_model = self._models_queue.get(timeout=1.0)
        while self._results_queue.qsize() > 0:
            result = self._results_queue.get()
            model = self._models_queue.get()
            if result[1] < best_result[1]:
                best_result = result
                self._best_model = model
        # вывод лучшего результата
        print(self._config_handler.get_str_results(best_result))  # текстом
        graph_results = self._config_handler.get_graph_results(best_result)  # графиками
        # config.get_graph_results возвращает None, если графики не требуются
        if graph_results is not None:
            graph_function, graph_params = graph_results
            figs_axs = graph_function(**graph_params)  # получение фигур
            for fig, _ in figs_axs:
                self._plot(fig)  # отображение графика
        self.startButton.setEnabled(True)  # разблок-ка кнопки запуска обучения
        self.exportButton.setEnabled(True)


class _Worker(QRunnable):
    """
    Класс-оболочка для QThread, реализующий вызов заданной функции с заданными
    аргументами
    """
    class _WorkerSignals(QObject):
        """
        Класс для описания сигналов, генерируемых тредом, выполняющего обучение
        """
        finished = pyqtSignal()  # признак завершения работы треда
        error = pyqtSignal(tuple)  # ошибка при работе треда
        result = pyqtSignal(object)  # результат работы треда

    def __init__(self, fn: Callable, kwargs: dict):
        """
        Конструктор треда
        Parameters
        ----------
        fn: Callable
            Функция, выполняемая в треде
        kwargs: dict
            Аргументы функции
        """
        super(_Worker, self).__init__()  # вызов конструктора QRunnable()
        self.fn = fn  # сохранение функции
        self.kwargs = kwargs  # сохранение аргументов
        self.signals = self._WorkerSignals()  # инициализация сигналов

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
