"""
Модуль с описанием классов исключений
"""


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
        super().__init__(f"Построение графиков не доступно для"
                         f" {inputs}-D входов")


class NotImplementedDataLoaderGraphException(Exception):
    """Нет возможности отрисовать графики"""
    def __init__(self, msg: str = None):
        super().__init__(f"Отрисовка графиков не доступна" +
                         f"\n{msg}" if msg is not None else "")


class NoPathToConfigSpecifiedException(Exception):
    """Не задан путь к файлу с настройками"""
    def __init__(self, msg: str = None):
        super().__init__(f"Не указан путь к файлу с настройками" +
                         f"\n{msg}" if msg is not None else "")
