from os.path import join

from forms.MainWindowSlots import MainWindowSlots


class MainWindow(MainWindowSlots):
    # При инициализации класса нам необходимо выпонить некоторые операции
    def __init__(self, form):
        # Сконфигурировать интерфейс методом из базового класса Ui_Form
        self.setupUi(form)
        # Подключить созданные нами слоты к виджетам
        self.connect_slots()
        # Подготовка окна
        self.init()

        CONFIG_DIR = "settings"
        CONFIG_NAME = "config"
        CONFIG_EXTENSION = 'json'
        CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
        path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)

        self.load(path_to_config)

    # Подключаем слоты к виджетам
    def connect_slots(self):
        self.inputsSpinBox.valueChanged.connect(self.inputs_count_changed)
        self.layersSpinBox.valueChanged.connect(self.layers_count_changes)

