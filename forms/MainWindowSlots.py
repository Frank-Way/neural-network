import sys
import traceback
from os.path import exists
from typing import List

import numpy as np
import sympy
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtWidgets import QMessageBox, QComboBox, QGridLayout, QMdiSubWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQT, FigureCanvasQTAgg, FigureCanvasAgg, NavigationToolbar2QT
import matplotlib
from matplotlib.backends.backend_template import FigureCanvas

from forms.UI_MainWindow import Ui_MainWindow
from config import Configuration
from losses import MeanSquaredError, SoftmaxCrossEntropy
from operations import Linear, Sigmoid, Tanh, ReLU, LeakyReLU
from optimizers import SGD, SGDMomentum


matplotlib.use('Qt5Agg')


class MainWindowSlots(Ui_MainWindow):
    losses = {"MSE": MeanSquaredError,
              "Softmax перекрёстная энтропия": SoftmaxCrossEntropy}
    optimizers = {"Стохастический градиентный спуск": SGD,
                  "Стохастический градиентный спуск с инерцией": SGDMomentum}
    decay_types = {"Линейно": "lin",
                   "Экспоненциально": "exp"}
    activations = {"Линейная": Linear,
                   "Сигмоида": Sigmoid,
                   "Гиперб. тангенс": Tanh,
                   "ReLU": ReLU,
                   "Leaky ReLU": LeakyReLU}
    decoder = {"sigmoid": "Сигмоида", "linear": "Линейная",
               "leakyrelu": "Leaky ReLU", "relu": "ReLU",
               "tanh": "Гиперб. тангенс", "mse": "MSE",
               "sce": "Softmax перекрёстная энтропия",
               "sgd": "Стохастический градиентный спуск",
               "sgdm": "Стохастический градиентный спуск с инерцией",
               "lin": "Линейно", "exp": "Экспоненциально"}
    encoder = dict(zip(decoder.values(), decoder.keys()))

    def init(self):
        self._init_limit_edits()
        self._init_layer_edits_and_comboboxes()
        self.inputsMinMaxTable. \
            setHorizontalHeaderLabels("x_min;x_max".split(";"))
        self.layersTable. \
            setHorizontalHeaderLabels("Нейронов;"
                                      "Активация;"
                                      "Drop-out".split(";"))
        self.load_losses(self.losses.keys())
        self.load_decay_types(self.decay_types.keys())
        self.load_optimizers(self.optimizers.keys())

        self.threadpool = QThreadPool()

    def load(self, path: str):
        self.path = path
        self.config = Configuration(path)
        if exists(path):
            self.config.load()
            self.inputs = self.config.inputs["count"]
            self.inputsSpinBox.setValue(self.inputs)
            for ii in range(self.inputs):
                for jj in range(2):
                    self.limit_edits[ii][jj].setText(str(self.config.inputs["limits"][ii][jj]))
            self.functionTextEdit.setText(self.config.inputs["function"])
            self.sampleSizeEdit.setText(str(self.config.inputs["size"]))
            self.testSizeEdit.setText(str(self.config.inputs["p_to_test"]))
            self.extendEdit.setText(str(self.config.inputs["p_to_extend"]))
            self.lrEdit.setText(str(self.config.train["lr"]))
            self.lrFinalEdit.setText(str(self.config.train["final_lr"]))
            self._set_combobox_item(self.decayComboBox, "decay_type")
            self.epochsEdit.setText(str(self.config.train["epochs"]))
            self.queryEdit.setText(str(self.config.train["query_times"]))
            self.batchSizeEdit.setText(str(self.config.train["batch_size"]))
            self.stoppingCheckBox. \
                setChecked(int(self.config.train["early_stopping"]))
            self.printCheckBox. \
                setChecked(int(self.config.train["print_results"]))
            self.plotsCheckBox.setChecked(int(self.config.train["show_plots"]))
            self._set_combobox_item(self.lossComboBox, "loss")
            self._set_combobox_item(self.optimizerComboBox, "optimizer")
            self.momentumEdit.setText(str(self.config.train["momentum"]))
            self.layers = self.config.layers["count"]
            self.layersSpinBox.setValue(self.layers)
            for ii in range(self.layers):
                self.neuron_edits[ii].setText(str(self.config.layers["config"][ii]["neurons"]))
                self.activation_comboboxes[ii]. \
                    setCurrentIndex(self.activation_comboboxes[ii].
                                    findText(self.decoder[self.config.
                                             layers["config"][ii]["activation"]]))
                self.dropout_edits[ii].setText(str(self.config.layers["config"][ii]["dropout"]))

    def load_decay_types(self, decay_types: List[str]) -> None:
        self.decayComboBox.addItems(decay_types)

    def load_losses(self, losses: List[str]) -> None:
        self.lossComboBox.addItems(losses)

    def load_optimizers(self, optimizers: List[str]) -> None:
        self.optimizerComboBox.addItems(optimizers)

    def load_activations(self, activations: List[str], cb: QComboBox) -> None:
        cb.addItems(activations)

    def inputs_count_changed(self):
        self._init_limit_edits()
        if self.inputs == 1:
            self.inputsMinMaxLabel.setText("F(X1) =")
        elif self.inputs == 2:
            self.inputsMinMaxLabel.setText("F(X1, X2) =")
        else:
            self.inputsMinMaxLabel.setText(f"F(X1, ..., X{self.inputs}) =")

    def layers_count_changes(self):
        self._init_layer_edits_and_comboboxes()

    def validate_function(self):
        self._read_function()

        valid = self._is_function_valid()

        if not valid:
            QMessageBox.about(self.centralwidget, "Ошибка",
                              "Не удалось прочитать функцию")
        else:
            QMessageBox.about(self.centralwidget, "Успех",
                              "Функция успешно прочитана")

    def plot_function(self):
        if self.function is None:
            self._read_function()
        pass

    def read(self):
        read_failure = False
        try:
            self.inputs = abs(int(self.inputsSpinBox.value()))
            self._read_function()
            self.limits = [[None for jj in range(2)] for ii in range(self.inputs)]
            for ii in range(self.inputs):
                for jj in range(2):
                    self.limits[ii][jj] = float(self.limit_edits[ii][jj].text())
            self.size = abs(int(self.sampleSizeEdit.text()))
            self.batch_size = min(self.size, abs(int(self.batchSizeEdit.text())))
            self.test_size = abs(float(self.testSizeEdit.text()))
            self.extending = abs(float(self.extendEdit.text()))
            self.lr = abs(float(self.lrEdit.text()))
            self.lr_final = min(self.lr, abs(float(self.lrFinalEdit.text())))
            self.decay_type = self.encoder[self.decayComboBox.currentText()]
            self.loss = self.encoder[self.lossComboBox.currentText()]
            self.optimizer = self.encoder[self.optimizerComboBox.currentText()]
            self.epochs = abs(int(self.epochsEdit.text()))
            self.queries = abs(int(self.queryEdit.text()))
            self.early_stopping = bool(self.stoppingCheckBox.isChecked())
            self.printing = bool(self.printCheckBox.isChecked())
            self.plotting = bool(self.plotsCheckBox.isChecked())
            self.momentum = 1.0
            if self.optimizer == "sgdm":
                self.momentum = abs(float(self.momentumEdit.text()))
            self.layers = abs(int(self.layersSpinBox.value()))
            self.layers_config = [None for ii in range(self.layers)]
            for ii in range(self.layers):
                neurons = abs(int(self.neuron_edits[ii].text()))
                activation = self.encoder[self.activation_comboboxes[ii].currentText()]
                dropout = abs(float(self.dropout_edits[ii].text()))
                self.layers_config[ii] = (neurons, activation, dropout)
        except ValueError:
            read_failure = True
            QMessageBox.warning(self.centralwidget, "Ошибка",
                                "Заполните все поля корректными значениями")
        self.values = None
        if not read_failure:
            self.values = {"inputs":{"count": self.inputs,
                                     "path": "",
                                     "data_loader": "ap",
                                     "function": self.str_expression,
                                     "limits": self.limits,
                                     "size": self.size,
                                     "scale_inputs": False,
                                     "p_to_test": self.test_size,
                                     "p_to_extend": self.extending},
                           "train":{"seed": 0,
                                    "use_seed": False,
                                    "lr": self.lr,
                                    "final_lr": self.lr_final,
                                    "decay_type": self.decay_type,
                                    "epochs": self.epochs,
                                    "query_times": self.queries,
                                    "batch_size": self.batch_size,
                                    "early_stopping": self.early_stopping,
                                    "print_results": self.printing,
                                    "show_plots": self.plotting,
                                    "momentum": self.momentum,
                                    "loss": self.loss,
                                    "optimizer": self.optimizer,
                                    "trainer": "t"},
                           "layers":{"count": self.layers,
                                     "config":[{
                                         "neurons": self.layers_config[ii][0],
                                         "class": "dense",
                                         "activation": self.layers_config[ii][1],
                                         "dropout": self.layers_config[ii][2],
                                         "weight_init": "glorot"
                                     } for ii in range(self.layers)]}}

    def validate(self) -> str:
        message = ""
        if self.values is None:
            return "Чтение параметров не было выполнено"
        valid_inputs = self.inputs != 0
        message += "" if valid_inputs else "Число входов должно быть" \
                                           " отлично от нуля\n"

        valid_function = self._is_function_valid()
        message += "" if valid_function else "Функция задана неверно\n"

        valid_limits = all(map(lambda limit: limit[0] < limit[1], self.limits))
        message += "" if valid_limits else "Левая граница для входа должна " \
                                           "быть строго меньше правой\n"

        valid_size = self.size != 0
        message += "" if valid_size else "Размер выборки должен быть" \
                                         " отличен от нуля\n"

        valid_batch = self.batch_size != 0
        message += "" if valid_batch else "Размер пакета должен быть" \
                                          " отличен от нуля\n"

        valid_test = self.test_size != 0.0
        message += "" if valid_test else "Размер тестовой выборки должен " \
                                         "быть отличен от нуля\n"

        valid_lr = 0.0 < self.lr <= 2
        message += "" if valid_lr else "Начальная скорость обучения должна " \
                                       "быть в пределах (0; 2]\n"

        valid_final_lr = 0.0 < self.lr_final <= 2
        message += "" if valid_final_lr else "Конечная скорость обучения " \
                                             "должна быть в пределах (0; 2]\n"

        valid_decay = self.decay_type is not None
        message += "" if valid_decay else "Не выбран способ снижения " \
                                          "скорости обучения\n"

        valid_loss = self.loss is not None
        message += "" if valid_loss else "Не выбрана функция потерь\n"

        valid_optimizer = self.optimizer is not None
        message += "" if valid_optimizer else "Не выбран оптимизатор\n"

        valid_epochs = self.epochs > 1
        message += "" if valid_epochs else "Количество эпох обучения " \
                                           "должно быть больше 1\n"

        valid_queries = self.queries > 0
        message += "" if valid_queries else "Недостаточное количество эпох " \
                                            "обучения для реализации заданного" \
                                            " числа опросов\n"

        valid_momentum = True
        if self.optimizer == "sgdm":
            valid_momentum = 0.0 < self.momentum <= 1.0
            message += "" if valid_momentum else "Инертность должна быть" \
                                                 " в пределах (0; 1]\n"

        valid_layers = self.layers != 0
        message += "" if valid_layers else "Число слоёв должно быть" \
                                           " отлично от нуля\n"

        valid_layers_config_neurons = all(map(lambda cfg:
                                              cfg[0] != 0,
                                              self.layers_config))
        if not valid_layers_config_neurons:
            message += "Число нейронов должно быть отлично от нуля\n"

        valid_layers_config_activations = all(map(lambda cfg:
                                                  cfg[1] is not None,
                                                  self.layers_config))
        if not valid_layers_config_activations:
            message += "Не выбрана функция активации\n"

        valid_layers_config_dropout = all(map(lambda cfg:
                                              0.0 < cfg[2] <= 1.0,
                                              self.layers_config))
        if not valid_layers_config_dropout:
            message += "Drop-out должен быть в пределах (0; 1]\n"

        valid = all((valid_queries, valid_epochs, valid_test,  valid_size, valid_lr,
                     valid_inputs, valid_layers, valid_final_lr,valid_function,
                     valid_optimizer, valid_momentum, valid_limits, valid_loss,
                     valid_layers_config_neurons, valid_layers_config_dropout,
                     valid_layers_config_activations, valid_batch, valid_decay))
        if not valid:
            return message

    def run(self):
        self.read()
        msg = self.validate()
        if msg is not None:
            QMessageBox.warning(self.centralwidget, "Ошибка", msg)
        else:
            self.startButton.setEnabled(False)
            self.config.update(self.values)
            self.config.save()
            data = self.config.get_data()
            trainer = self.config.get_trainer()
            fit_params = self.config.get_fit_params(data)

            # Pass the function to execute
            worker = _Worker(fn=trainer.fit,
                             kwargs=fit_params)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self._output_results)
            worker.signals.finished.connect(lambda: self.startButton.setEnabled(True))

            # Execute
            self.threadpool.start(worker)

    def clear_output(self):
        self.outputEdit.clear()

    def _init_limit_edits(self):
        self.inputs = int(self.inputsSpinBox.value())
        self.inputsMinMaxTable.clear()
        self.inputsMinMaxTable.setRowCount(self.inputs)

        self.limit_edits = [[None for jj in range(2)] for ii in range(self.inputs)]
        for ii in range(self.inputs):
            for jj in range(2):
                edit = QtWidgets.QLineEdit(self.inputsMinMaxTable)
                edit.setObjectName(f"limitEdit{ii + 1}_{jj + 1}")
                edit.setClearButtonEnabled(True)
                edit.setPlaceholderText(str(float(jj)))
                self.limit_edits[ii][jj] = edit
                self.inputsMinMaxTable.setCellWidget(ii, jj, edit)

    def _init_layer_edits_and_comboboxes(self):
        self.layers = int(self.layersSpinBox.value())
        self.layersTable.clearContents()
        self.layersTable.setRowCount(self.layers)
        self.neuron_edits = []
        self.activation_comboboxes = []
        self.dropout_edits = []
        for ii in range(self.layers):
            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"neuronEdit{ii + 1}")
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("8")
            self.neuron_edits.append(edit)
            self.layersTable.setCellWidget(ii, 0, edit)

            cb = QtWidgets.QComboBox(self.layersTable)
            cb.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
            cb.setObjectName(f"activationComboBox{ii + 1}")
            self.load_activations(self.activations.keys(), cb)
            self.activation_comboboxes.append(cb)
            self.layersTable.setCellWidget(ii, 1, cb)

            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"dropoutEdit{ii + 1}")
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("1.0")
            self.dropout_edits.append(edit)
            self.layersTable.setCellWidget(ii, 2, edit)

    def _set_combobox_item(self, cb: QComboBox, property_name: str) -> None:
        cb. \
            setCurrentIndex(cb.
                            findText(self.
                                     decoder[self.config.train[property_name]]))

    def _read_function(self):
        function = str(self.functionTextEdit.toPlainText()).lower()
        input_symbols = [sympy.Symbol(f"x{ii + 1}")
                         for ii in range(self.inputs)]
        simplified_expression = sympy.simplify(function)
        self.str_expression = str(simplified_expression)

        self.function = sympy.lambdify(input_symbols,
                                       simplified_expression)

    def _is_function_valid(self) -> bool:
        valid = True
        try:
            res = self.function(*np.random.random(self.inputs))
        except NameError:
            valid = False

        if type(res) != np.float64:
            valid = False

        return valid

    def _output_results(self, results):
        print(self.config.get_str_results(results))
        graph_results = self.config.get_graph_results(results)
        if graph_results is not None:
            graph_function, graph_params = graph_results
            fig, axes = graph_function(**graph_params)

            sub = QMdiSubWindow(self.mdiArea)
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, sub)
            sub.layout().addWidget(toolbar)
            sub.layout().addWidget(canvas)
            sub.show()



class _WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class _Worker(QRunnable):
    def __init__(self, fn, kwargs):
        super(_Worker, self).__init__()
        self.fn = fn
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(**self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
