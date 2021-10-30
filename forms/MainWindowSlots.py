from os.path import exists
from typing import List

import numpy as np
import sympy
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QComboBox, QTableWidgetItem

from forms.UI_MainWindow import Ui_MainWindow
from config import Configuration
from losses import MeanSquaredError, SoftmaxCrossEntropy
from operations import Linear, Sigmoid, Tanh, ReLU, LeakyReLU
from optimizers import SGD, SGDMomentum


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
        self.inputsMinMaxTable.\
            setHorizontalHeaderLabels("x[i]_min;x[i]_max".split(";"))
        self.layersTable.\
            setHorizontalHeaderLabels("Нейронов;"
                                      "Активация;"
                                      "Drop-out".split(";"))
        self.load_losses(self.losses.keys())
        self.load_decay_types(self.decay_types.keys())
        self.load_optimizers(self.optimizers.keys())

    def load(self, path: str):
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
            self.stoppingCheckBox.\
                setChecked(int(self.config.train["early_stopping"]))
            self.printCheckBox.\
                setChecked(int(self.config.train["print_results"]))
            self.plotsCheckBox.setChecked(int(self.config.train["show_plots"]))
            self._set_combobox_item(self.lossComboBox, "loss")
            self._set_combobox_item(self.optimizerComboBox, "optimizer")
            self.layers = self.config.layers["count"]
            self.layersSpinBox.setValue(self.layers)
            for ii in range(self.layers):
                self.neuron_edits[ii].setText(str(self.config.layers["config"][ii]["neurons"]))
                self.activation_comboboxes[ii].\
                    setCurrentIndex(self.activation_comboboxes[ii].
                                    findText(self.decoder[self.config.
                                             layers["config"][ii]["activation"]]))
                self.dropout_edits[ii].setText(str(self.config.layers["config"][ii]["dropout"]))



    def _set_combobox_item(self, cb: QComboBox, property_name: str) -> None:
        cb.\
            setCurrentIndex(cb.
                            findText(self.
                                     decoder[self.config.train[property_name]]))

    def read(self):
        pass

    def validate(self):
        pass
    def run(self):
        pass
    def exit(self):
        pass

    def inputs_count_changed(self):
        self.inputs = int(self.inputsSpinBox.value())
        self.inputsMinMaxTable.clear()
        self.inputsMinMaxTable.setRowCount(self.inputs)

        self.limit_edits = [[None for jj in range(2)] for ii in range(self.inputs)]
        for ii in range(self.inputs):
            for jj in range(2):
                edit = QtWidgets.QLineEdit(self.inputsMinMaxTable)
                edit.setObjectName(f"limitEdit{ii + 1}_{jj + 1}")
                self.limit_edits[ii][jj] = edit
                self.inputsMinMaxTable.setCellWidget(ii, jj, edit)
        if self.inputs == 1:
            self.inputsMinMaxLabel.setText("F(X1) =")
        elif self.inputs == 2:
            self.inputsMinMaxLabel.setText("F(X1, X2) =")
        else:
            self.inputsMinMaxLabel.setText(f"F(X1, ..., X{self.inputs}) =")

    def _read_function(self):
        function = str(self.functionTextEdit.toPlainText()).lower()
        input_symbols = [sympy.Symbol(f"x{ii + 1}")
                         for ii in range(self.inputs)]
        simplified_expression = sympy.simplify(function)
        self.str_expression = str(simplified_expression)

        self.function = sympy.lambdify(input_symbols,
                                       simplified_expression)

    def validate_function(self):
        self._read_function()

        valid = True
        try:
            res = self.function(*np.random(self.inputs))
        except NameError:
            valid = False

        if type(res) not in ('int', 'float'):
            valid = False
        if not valid:
            QMessageBox.about(self, "Ошибка", "Не удалось прочитать функцию")
        else:
            QMessageBox.about(self, "Успех", "Функция успешно прочитана")

    def plot_function(self):
        if self.function is None:
            self._read_function()
        pass

    def load_decay_types(self, decay_types: List[str]) -> None:
        self.decayComboBox.addItems(decay_types)

    def load_losses(self, losses: List[str]) -> None:
        self.lossComboBox.addItems(losses)

    def load_optimizers(self, optimizers: List[str]) -> None:
        self.optimizerComboBox.addItems(optimizers)

    def layers_count_changes(self):
        self.layers = int(self.layersSpinBox.value())
        self.layersTable.clearContents()
        self.layersTable.setRowCount(self.layers)
        self.neuron_edits = []
        self.activation_comboboxes = []
        self.dropout_edits = []
        for ii in range(self.layers):
            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"neuronEdit{ii + 1}")
            self.neuron_edits.append(edit)
            self.layersTable.setCellWidget(ii, 0, edit)

            cb = QtWidgets.QComboBox(self.layersTable)
            cb.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
            cb.setObjectName(f"activationComboBox{ii+1}")
            self.load_activations(self.activations.keys(), cb)
            self.activation_comboboxes.append(cb)
            self.layersTable.setCellWidget(ii, 1, cb)

            edit = QtWidgets.QLineEdit(self.layersTable)
            edit.setObjectName(f"dropoutEdit{ii + 1}")
            self.dropout_edits.append(edit)
            self.layersTable.setCellWidget(ii, 2, edit)


    def load_activations(self, activations: List[str], cb: QComboBox) -> None:
        cb.addItems(activations)

    def clear_output(self):
        self.outputEdit.clear()
