"""
Модуль с описанием класса, представляющего интерфейс главного окна
"""
from os.path import join
from queue import Queue

from PyQt5 import QtGui
from PyQt5.QtCore import QCoreApplication, QObject, pyqtSignal
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
        self.exportButton.clicked.connect(self.export_model)
        self.tabsAction.triggered.connect(self.tabs_help)
        self.buttonsAction.triggered.connect(self.buttons_help)
        self.guideAction.triggered.connect(self.guide_help)
        self.aboutAction.triggered.connect(self.about_help)
        self.settingsAction.triggered.connect(self.settings_help)


class OutLog:
    """
        Клосс с описанием логгера, который перенаправляет вывод в заданный элемент
        QTextEdit
    """
    class LogSignals(QObject):
        """
        Сигналы логгера
        """
        new_msg = pyqtSignal()

    _MSG_QUEUE: Queue

    def __init__(self, edit: QTextEdit) -> None:
        """
        При создании назначениется элемент для вывода в него тестовой информации
        Parameters
        ----------
        edit: Поля для вывода
        """
        self.edit = edit
        self._MSG_QUEUE = Queue()
        self.signals = self.LogSignals()
        self.signals.new_msg.connect(self._update)

    def write(self, msg) -> None:
        """
        Запись вывода в поле
        Parameters
        ----------
        msg: Сообщение
        """
        self._MSG_QUEUE.put_nowait(msg)
        self.signals.new_msg.emit()

    def _update(self):
        # перемещение курсора в конец
        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(self._MSG_QUEUE.get())  # запись сообщния

    def flush(self) -> None:
        """
        Очищение вывода
        """
        self.edit.clear()
