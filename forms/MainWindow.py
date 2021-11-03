"""
Модуль с описанием класса, представляющего интерфейс главного окна
"""
from os.path import join

from PyQt5 import QtGui
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QTextEdit

from forms.MainWindowSlots import MainWindowSlots


class MainWindow(MainWindowSlots):
    """
    Класс с описанием интерфейса главного окна
    """
    def __init__(self, form):
        """
        Конструктор формы
        Parameters
        ----------
        form: Форма
        """
        # конфигурация интерфейса методом из базового класса Ui_MainWindow
        self.setupUi(form)
        self.connect_slots()  # подключение слотов к виджетам
        self.init()  # подготовка окна

        # задание параметров файла с настройками
        CONFIG_DIR = "settings"
        CONFIG_NAME = "config"
        CONFIG_EXTENSION = 'json'
        CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
        path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)

        self.load(path_to_config)  # загрузка настроек

    def connect_slots(self) -> None:
        """
        Подключение слотов к сигналам
        """
        self.inputsSpinBox.valueChanged.connect(self.inputs_count_changed)
        self.layersSpinBox.valueChanged.connect(self.layers_count_changes)
        self.validateFunctionButton.clicked.connect(self.validate_function)
        self.startButton.clicked.connect(self.run)
        self.exitButton.clicked.connect(QCoreApplication.instance().quit)
        self.clearOutputButton.clicked.connect(self.clear_output)
        self.plotFunctionButton.clicked.connect(self.plot_function)


class OutLog:
    """
    Клосс с описанием логгера, который перенаправляет вывод в заданный элемент
    QTextEdit
    """
    def __init__(self, edit: QTextEdit) -> None:
        """
        При создании назначениется элемент для вывода в него тестовой информации
        Parameters
        ----------
        edit: Поля для вывода
        """
        self.edit = edit

    def write(self, msg) -> None:
        """
        Запись вывода в файл
        Parameters
        ----------
        msg: Сообщение
        """
        # перемещение курсора в конец
        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(msg)  # запись сообщния

    def flush(self) -> None:
        """
        Очищение вывода
        """
        self.edit.clear()
