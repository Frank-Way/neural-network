"""
Основной модуль для запуска программного средства для обучения нейронных сетей
для решения задач аппроксимации математических функций
"""
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from forms.MainWindow import MainWindow, OutLog

if __name__ == '__main__':
    app = QApplication(sys.argv)  # создание экземпляра приложения
    window = QMainWindow()  # создание базового окна для отображения UI
    ui = MainWindow(window)  # создание экземпляра UI
    window.show()  # отображение окна
    # перенаправление стандартного вывода на виджет outputEdit
    sys.stdout = OutLog(ui.outputEdit)
    sys.exit(app.exec_())  # обработка нажания на кнопку окна "Закрыть"
