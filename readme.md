# Нейросетевой аппроксиматор

### Краткое описание

Проект разработан в рамках практической части ВКР на тему "Разработка 
программного средства для обучения искусственной нейронной сети для 
преобразования частоты в код". 

Основная задача проекта - обучение нейронных 
сетей для аппроксимации различных аналитических математических функций.
Проект имеет графический пользовательский интерфейс для задания  функций, 
настроек обучения и структуры сети. Обучение реализуется без использования 
специализированных библиотек для ML и DL - бОльшая часть функционала 
реализуется с помощью **NumPy**. Структура проекта вдохновлена фреймворком 
PyTorch.

### Ключевые особенности

- **гибкость** - проект предоставляет возможность обучения сетей различной 
структуры
- **производительность** - основная часть "сложной" математики реализована на 
NumPy, к тому же имеется возможность параллельного запуска нескольких попыток
обучения в нескольких потоках
- **разделение логики и настроек** - выполнение кода управляется 
конфигурационным файлом, т.е. изменение параметров обучения не требует изменения
 кода
- **масштабируемость** - структура проекта позволяет без особых усилий добавлять 
новые источники данных, функции активаций и т.д.
- **ввод функции** - для задания функций может быть использован достаточно 
широкий набор операторов, предоставляемый библиотекой **SymPy**

### Установка и запуск

Для установки и запуска проекта необходимо:
0. Загрузить Python версии 3.8 или новее
1. Создать директорию проекта

        mkdir neural_approximator
        cd neural_approximator
2. Создать виртуальную среду

        python -m venv env
        source env/bin/activate  # на Windows 'env\Scripts\activate'
3. Загрузить зависимости проекта

        pip install -r requirements.txt
4. Запустить проект с помощью **main.py**

        python main.py