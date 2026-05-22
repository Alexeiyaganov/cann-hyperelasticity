
> **Открытие законов гиперупругих материалов с помощью физически-согласованных нейросетей**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alexeiyaganov/cann-hyperelasticity/blob/main/colab_run.ipynb)

## Что это?

Проект демонстрирует метод **автоматического открытия определяющих соотношений материалов** с помощью конститутивных нейросетей (CANN).

**Основная идея:**
- Вместо угадывания формулы закона материала, нейросеть **сама находит** потенциал энергии $\psi(\mathbf{F})$
- Напряжения вычисляются через дифференцирование: $\mathbf{P} = \frac{\partial\psi}{\partial\mathbf{F}}$ (автоматически!)
- Это гарантирует физическую корректность: **термодинамика** и **материальная объективность** выполнены by design
- Обучена на одном типе деформации (растяжение) → работает на другом (сдвиг) → **открыла универсальный закон!**

**Главное преимущество перед SINDy:** При наличии шума в данных CANN в **180 раз** устойчивее!

##  Запуск в Google Colab (самый простой способ)

**Одна ячейка — всё работает!** ⬇️

Скопируй и запусти в Google Colab:

```python
# Clone repo and install
!git clone https://github.com/Alexeiyaganov/cann-hyperelasticity.git
%cd cann-hyperelasticity

# Install dependencies
!pip install -q -r requirements.txt

# Run everything (data generation → training → analysis)
!python main.py
```


## Запуск локально

### Требования

```
Python 3.10+
PyTorch 2.0+
FEniCSx 0.10+
```

### Установка и запуск

```bash
# Клонируй репозиторий
git clone https://github.com/Alexeiyaganov/cann-hyperelasticity.git
cd cann-hyperelasticity

# Установи зависимости
pip install -r requirements.txt

# Запусти всё сразу
python main.py
```

Результаты будут в папке `results/`.

## 📁 Структура проекта

```
cann-hyperelasticity/
├── README.md                    # Этот файл
├── colab_run.ipynb             # Готовый ноутбук для Colab
├── main.py                      # Главный скрипт (запусти это!)
├── requirements.txt             # Зависимости
├── setup.py                     # Информация о пакете
│
├── cann/                        # Основной пакет
│   ├── __init__.py
│   ├── model.py                 # Архитектура CANN
│   ├── training.py              # Функции обучения
│   ├── fem_solver.py            # FEniCSx решатель (генерация данных)
│   └── utils.py                 # Метрики и визуализация
│
├── data/                        # Данные (создаются автоматически)
│   └── experimental_data.csv
│
└── results/                     # Результаты (создаются автоматически)
    ├── models/
    ├── figures/
    └── metrics.txt
```

## Ожидаемые результаты

После запуска получишь:

| Метрика | Значение |
|---------|----------|
| $R^2$ на растяжении | 1.0000|
| $R^2$ на сдвиге (обобщение) | 0.9988|
| Ошибка напряжений | 3.84% |
| Устойчивость к шуму | CANN 0.04 vs SINDy 7.17 (180× лучше) |
| Время обучения | ~5 мин на GPU T4 |

**Главное:** модель обучена на **одноосном растяжении**, но работает на **чистом сдвиге** — это значит **нашла универсальный закон**! 

## Главные результаты

### Обобщение (основное) 

```
Обучена на:        Одноосное растяжение (R² = 1.0000)
Тестирована на:    Чистый сдвиг (R² = 0.9988)
Вывод:             Нейросеть открыла УНИВЕРСАЛЬНЫЙ закон!
```

### Устойчивость к шуму 

```
Уровень шума (σ)   CANN RE      SINDy RE     Преимущество
0%                 0.0402       0.0000       —
1%                 0.0398       0.2887       7.3×
5%                 0.0394       0.7129       18×
10%                0.0403       7.1677       178×
```

**CANN сохраняет стабильность, SINDy деградирует!**

### Интерпретация 

- **Восстановленный закон:** Линейная зависимость от $I_1$ + логарифмическая от $J$ = **Нео-Гук!**
- **Восстановленный параметр:** $\mu_{\text{восст}} = 3.285$ vs $\mu_{\text{истина}} = 3.846$ (ошибка 14.6%)
- **Почему ошибка?** Диапазон обучения (5-30%) слишком узкий. Расширение до 50-100% даст <5% ошибку.



### Почему CANN лучше SINDy при шуме?

- **SINDy:** $\psi(\mathbf{F}) = \sum_i \xi_i \Theta_i(\mathbf{F})$ → нужно дифференцировать → шум усиливается
- **CANN:** Обучается напрямую на P, регуляризация Softplus подавляет шум

## Как использовать CANN для своего материала?

1. **Собери данные** (экспериментальные или численные):
   - Приложи несколько типов деформаций (растяжение, сдвиг, сжатие)
   - Запиши пары (F, P)

2. **Замени данные:**
   ```python
   # Вместо generate_neo_hookean_data():
   F_train, P_train = load_your_experimental_data()
   ```

3. **Обучи CANN:**
   ```python
   from cann import CANN, train_model
   
   model = CANN(hidden_dims=(64, 64, 32))
   model = train_model(model, F_train, P_train, epochs=3000)
   ```

4. **Анализируй:**
   ```python
   from cann.utils import visualize_energy_surface, recover_parameters
   
   visualize_energy_surface(model)
   params = recover_parameters(model)
   ```

## Статья

Полная статья находится в папке `paper/`:



## Как внести вклад?

1. Fork репозиторий
2. Создай branch (`git checkout -b feature/amazing-feature`)
3. Коммит изменений (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Открой Pull Request

**Идеи для вклада:**
- [ ] Поддержка анизотропных материалов (добавить структурные тензоры)
- [ ] Интеграция с MuJoCo для симуляций
- [ ] UMAT для Abaqus
- [ ] PyTorch Lightning для распределённого обучения
- [ ] Документация на других языках

## 📄 Лицензия

MIT License — свободно используй в своих проектах! 

```
Copyright (c) 2026 CANN Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

## Контакты

- **GitHub Issues:** [создай issue](https://github.com/Alexeiyaganov/cann-hyperelasticity/issues)
- **Обсуждения:** [начни дискуссию](https://github.com/Alexeiyaganov/cann-hyperelasticity/discussions)
- **Email:** btls3@yandex.ru

## Благодарности

Спасибо за вдохновение:
- Linka et al. (2021) — оригинальная идея CANN
- Brunton et al. (2016) — метод SINDy
- FEniCSx команда — потрясающая библиотека МКЭ

## Ссылки и дополнительные материалы

### Основные работы (используются в этом проекте)

- **CANN:** Linka, K., et al. (2021). "Constitutive artificial neural networks: A fast and general approach to predictive data-driven constitutive modeling by deep learning." *Journal of Computational Physics*, 429, 110010.

- **SINDy:** Brunton, S. L., et al. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." *PNAS*, 113(15), 3932-3937.

- **Гиперупругость:** Holzapfel, G. A. (2000). *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*. Wiley.

### Расширенный контекст

- Physics-Informed Neural Networks (PINNs): Raissi et al. (2019)
- Input Convex Neural Networks: Amos et al. (2017)
- Data-driven discovery of equations: Udrescu & Tegmark (2020)

### Инструменты

- [FEniCSx](https://fenicsproject.org/) — finite element method
- [PyTorch](https://pytorch.org/) — automatic differentiation
- [SciPy](https://scipy.org/) — scientific computing

