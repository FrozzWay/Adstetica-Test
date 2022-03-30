# Тестовое задание Adstetica

### Исходные данные

1. Макет страницы в Figma [(drive.google)](https://drive.google.com/file/d/1X2LCAKs-CNRZLYHcTkZ9JTtmVW0PWKLB/view?usp=sharing)
2. Размеченный набор текстов на русском языке для обучения алгоритмов [(drive.google)](https://drive.google.com/file/d/1NMj989x2KzlNAl12fXQ1YO9Szm_4yEi-/view?usp=sharing)

### Постановка задачи
Разработать веб-сервис для автоматического определения языка (русский/английский) и темы любого загруженного текста.

Для этого необходимо:
- выбрать архитектуру алгоритма классификации текста по теме
- реализовать и обучить алгоритм классификации текста по теме
- подобрать и реализовать алгоритм определения языка текста (только 2
языка: русский, английский)
- разработать интерфейсы
- подключить интерфейсы к алгоритмам и протестировать работу алгоритмов

#### Требования
1. Интерфейсы сервиса должны полностью соответствовать макету
2. Алгоритм должен определять тему как текста на русском, так и текста на английском
3. Точность определения темы текста должна быть не ниже 70%


------------



### Разработка алгоритма классификации

Подготовка данных для обучения и само обучение велось в среде Google Colab.
[Ноутбук доступен по ссылке](https://colab.research.google.com/drive/16zH8-1ag7SU9znab0y9gT993DQQq177Z?usp=sharing)

##### Выбранная архитектура нейронной сети
<img src="https://i.imgur.com/kvxJWgV.png"/>

##### Полученная точность на тестовом наборе данных
<img src="https://i.imgur.com/xlHNXaz.png"/>



------------


### Алгоритм развёртывания

В проекте использован пакетный менеджер *pipenv.*
Для установки необходимых пакетов, среди которых занимающий много памяти tensorflow: `pipenv install`
Запуск из под системы Windows через .bat файл **start (pipenv).bat**

Веса нейронной сети и ещё некоторое количество необходимых файлов будут загружены в процессе первого запуска.


## Конечный результат
<img src= "https://i.imgur.com/XTKMrlm.png"  width="700"/>

