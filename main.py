import sys
from os.path import join

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow

from config import Configuration
from forms.MainWindow import MainWindow

# CONFIG_DIR = "settings"
# CONFIG_NAME = "config"
# CONFIG_EXTENSION = 'json'
# CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
# path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)
# config = Configuration(path_to_config)
#
# data = config.get_data()
# trainer = config.get_trainer()
# fit_params = config.get_fit_params(data)
# results = trainer.fit(**fit_params)
#
# print(config.get_str_results(results))
#
# graph_results = config.get_graph_results(results)
# if graph_results is not None:
#     graph_function, graph_params = graph_results
#     graph_function(**graph_params)

if __name__ == '__main__':
    # Создаём экземпляр приложения
    app = QApplication(sys.argv)
    # Создаём базовое окно, в котором будет отображаться наш UI
    window = QMainWindow()
    # Создаём экземпляр нашего UI
    ui = MainWindow(window)
    # Отображаем окно
    window.show()
    # Обрабатываем нажатие на кнопку окна "Закрыть"
    sys.exit(app.exec_())
