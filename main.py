"""
Основной модуль для запуска программного средства для обучения нейронных сетей
для решения задач аппроксимации математических функций
"""
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from forms.MainWindow import MainWindow, OutLog

if __name__ == '__main__':
    # Создаём экземпляр приложения
    app = QApplication(sys.argv)
    # Создаём базовое окно, в котором будет отображаться наш UI
    window = QMainWindow()
    # Создаём экземпляр нашего UI
    ui = MainWindow(window)
    # Отображаем окно
    window.show()
    # Переназначаем стандартный вывод на виджет outputEdit
    sys.stdout = OutLog(ui.outputEdit)
    # Обрабатываем нажатие на кнопку окна "Закрыть"
    sys.exit(app.exec_())
