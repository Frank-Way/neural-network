"""
Модуль с описанием классов исключений
"""
from data_loaders import DataLoader


class ConfigNotFoundException(Exception):
    """Не найден файл с настройками"""
    def __init__(self, path: str):
        super().__init__(f"Не найден файл настроек '{path}'")


class OptimizerNotFoundException(Exception):
    """Нет указанного оптимизатора"""
    def __init__(self, optimizer: str):
        super().__init__(f"Не найден оптимизатор '{optimizer}'")


class DataLoaderNotFoundException(Exception):
    """Нет указанного загрузчика данных"""
    def __init__(self, data_loader: str):
        super().__init__(f"Не найден загрузчик '{data_loader}'")


class TooManyInputsException(Exception):
    """Слишком много входов для построения графиков"""
    def __init__(self, inputs: int):
        super().__init__(f"Построение графиков не доступно для {inputs}-D входов")


class NotImplementedDataLoaderGraphException(Exception):
    """Нет возможности отрисовать графики"""
    def __init__(self, data_loader: DataLoader):
        super().__init__(f"Отрисовка графиков не доступна")
