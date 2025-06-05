# Система Автоматизированного Тестирования для Движка Анализа Воронок

Комплексная система тестирования для проверки корректности логики расчета метрик продуктовых воронок.

## 📋 Обзор

Эта тестовая система обеспечивает полное покрытие функциональности движка анализа воронок, включая:

- ✅ **Базовые сценарии** - линейные воронки, нулевая и 100% конверсия
- ⏱️ **Окна конверсии** - временные ограничения для прохождения воронки
- 🔢 **Методы подсчета** - unique_users, event_totals, unique_pairs
- 🔄 **Режимы повторного входа** - first_only, optimized_reentry
- ⚠️ **Пограничные случаи** - пустые данные, некорректные значения, производительность
- 🎯 **Сегментация** - фильтрация по свойствам событий и пользователей

## 🏗️ Структура Тестов

```
tests/
├── conftest.py                    # Фикстуры и общие настройки
├── test_basic_scenarios.py        # Базовые сценарии (Happy Path)
├── test_conversion_window.py      # Тестирование окон конверсии
├── test_counting_methods.py       # Методы подсчета
├── test_edge_cases.py             # Пограничные случаи
├── test_segmentation.py           # Сегментация и фильтрация
└── README.md                      # Данная документация
```

## 🚀 Запуск Тестов

### Базовая установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск всех тестов
python run_tests.py

# Или напрямую через pytest
python -m pytest tests/ -v
```

### Специфические категории тестов

```bash
# Базовые сценарии
python run_tests.py --basic

# Тестирование окон конверсии
python run_tests.py --conversion-window

# Методы подсчета
python run_tests.py --counting-methods

# Пограничные случаи
python run_tests.py --edge-cases

# Сегментация
python run_tests.py --segmentation

# Тесты производительности
python run_tests.py --performance
```

### Дополнительные опции

```bash
# Параллельный запуск
python run_tests.py --parallel

# С отчетом покрытия
python run_tests.py --coverage

# Быстрый smoke test
python run_tests.py --smoke

# Комплексный отчет
python run_tests.py --report

# По маркерам
python run_tests.py --marker basic
python run_tests.py --marker edge_case
```

## 📊 Маркеры Тестов

Тесты организованы с помощью pytest маркеров:

- `@pytest.mark.basic` - Базовые сценарии
- `@pytest.mark.conversion_window` - Тестирование окон конверсии
- `@pytest.mark.counting_method` - Методы подсчета
- `@pytest.mark.edge_case` - Пограничные случаи
- `@pytest.mark.segmentation` - Сегментация
- `@pytest.mark.performance` - Тесты производительности

## 🧪 Описание Тест-кейсов

### 1. Базовые Сценарии (`test_basic_scenarios.py`)

**Цель**: Проверка корректности работы в стандартных условиях

- `test_linear_funnel_calculation` - Стандартная линейная воронка с постепенным отсевом
- `test_zero_conversion_scenario` - Сценарий с нулевой конверсией после первого шага
- `test_perfect_conversion_scenario` - 100% прохождение всех шагов воронки
- `test_single_step_funnel` - Обработка воронки с одним шагом
- `test_two_step_minimal_funnel` - Минимальная валидная воронка (2 шага)
- `test_funnel_with_noise_events` - Игнорирование событий, не входящих в воронку

**Ожидаемые результаты**: Для тестовых данных 1000→800→600→400 пользователей должны получаться конверсии 100%→80%→60%→40%

### 2. Окна Конверсии (`test_conversion_window.py`)

**Цель**: Проверка корректности временных ограничений

- `test_events_within_conversion_window` - События в пределах окна засчитываются
- `test_events_outside_conversion_window` - События за пределами окна не засчитываются
- `test_events_at_conversion_window_boundary` - Граничные условия (≤ boundary)
- `test_sequential_conversion_windows` - Окно B→C отсчитывается от события B, не A
- `test_different_conversion_window_sizes` - Разные размеры окон дают разные результаты

**Ожидаемые результаты**: События в 1-часовом окне засчитываются, за пределами - нет

### 3. Методы Подсчета (`test_counting_methods.py`)

**Цель**: Проверка различных алгоритмов подсчета

- `test_unique_users_method` - Каждый пользователь засчитывается один раз на шаг
- `test_event_totals_method` - Подсчет общего количества событий
- `test_unique_pairs_method` - Пошаговые конверсии между соседними шагами
- `test_counting_method_comparison_same_data` - Разные методы дают разные результаты

**Ожидаемые результаты**: 
- Unique users: 2 пользователя = [2, 2]
- Event totals: 8 событий = [3, 5]

### 4. Пограничные Случаи (`test_edge_cases.py`)

**Цель**: Проверка устойчивости к некорректным данным

- `test_empty_dataset` - Пустой набор данных
- `test_events_with_same_timestamp` - События с одинаковым временем
- `test_out_of_order_events` - События в неправильном порядке
- `test_missing_user_id` - Отсутствующие user_id
- `test_missing_event_name` - Отсутствующие event_name
- `test_very_large_dataset_performance` - Производительность на больших данных

**Ожидаемые результаты**: Система должна корректно обрабатывать все граничные случаи без сбоев

### 5. Сегментация (`test_segmentation.py`)

**Цель**: Проверка фильтрации по свойствам

- `test_segmentation_by_event_property` - Сегментация по свойствам событий
- `test_segmentation_by_user_property` - Сегментация по свойствам пользователей
- `test_statistical_significance_between_segments` - Статистическая значимость различий
- `test_multiple_properties_same_event` - Корректная фильтрация при множественных свойствах

**Ожидаемые результаты**: Мобильные пользователи: [3,3,3], десктопные: [2,2,0]

## 📈 Ожидаемые Результаты

### Эталонные Данные

**Простая линейная воронка** (1000 пользователей):
```python
expected_results = {
    'steps': ['Sign Up', 'Email Verification', 'First Login', 'First Purchase'],
    'users_count': [1000, 800, 600, 400],
    'conversion_rates': [100.0, 80.0, 60.0, 40.0],
    'drop_offs': [0, 200, 200, 200],
    'drop_off_rates': [0.0, 20.0, 25.0, 33.33]
}
```

**Окно конверсии 1 час**:
- user_001 (события через 30 мин): ✅ засчитывается
- user_002 (события через 90 мин): ❌ не засчитывается  
- user_003 (события ровно через 60 мин): ✅ засчитывается

**Сегментация по платформе**:
- Mobile: 3 пользователя, 100% конверсия
- Desktop: 2 пользователя, конверсия на первых двух шагах

## 🔧 Конфигурация

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
markers =
    basic: Basic funnel calculation scenarios
    conversion_window: Tests for conversion window functionality
    counting_method: Tests for different counting methods
    edge_case: Edge cases and boundary conditions
    segmentation: Segmentation and filtering tests
addopts = -v --strict-markers --cov=app --cov-report=html
```

### Фикстуры (conftest.py)

Основные фикстуры для тестов:

- `base_timestamp` - Базовое время для всех тестов
- `default_config` - Стандартная конфигурация воронки
- `simple_linear_funnel_data` - Эталонные данные линейной воронки
- `conversion_window_test_data` - Данные для тестирования окон
- `calculator_factory` - Фабрика для создания калькуляторов
- `expected_simple_linear_results` - Ожидаемые результаты

## 📋 Критерии Успешности

✅ **Покрытие кода**: Минимум 85% покрытия тестами
✅ **Все сценарии**: Все описанные в ТЗ сценарии покрыты тестами
✅ **Автоматизация**: Тесты запускаются без ручного вмешательства
✅ **CI/CD готовность**: Поддержка запуска в CI/CD пайплайнах
✅ **Производительность**: Тесты выполняются в разумное время (<30 сек)
✅ **Читаемость**: Названия тестов четко описывают проверяемые сценарии

## 🚨 Обработка Ошибок

Система тестирования проверяет корректную обработку ошибок:

- **Валидация данных**: Проверка требуемых колонок
- **Пустые наборы**: Корректная обработка пустых DataFrame
- **Некорректные типы**: Устойчивость к смешанным типам данных
- **Отсутствующие значения**: Игнорирование записей с null значениями

## 📊 Отчетность

### HTML отчет покрытия
```bash
python run_tests.py --coverage
# Результат: htmlcov/index.html
```

### XML отчеты для CI/CD
```bash
python run_tests.py --report
# Результаты: coverage.xml, test-results.xml
```

### Подробный вывод
```bash
python -m pytest tests/ -v --tb=long
```

## 🔄 Интеграция с CI/CD

```yaml
# Пример для GitHub Actions
- name: Run Tests
  run: |
    pip install -r requirements.txt
    python run_tests.py --coverage --parallel
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 🤝 Разработка Тестов

### Добавление новых тестов

1. Определите категорию теста
2. Добавьте соответствующий маркер
3. Создайте эталонные данные
4. Опишите ожидаемые результаты
5. Задокументируйте проверяемый сценарий

### Соглашения по именованию

- Тесты: `test_описание_сценария`
- Фикстуры: `описательное_имя_data/config`  
- Классы: `TestФункциональность`

## 📚 Дополнительные Ресурсы

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Pandas Testing](https://pandas.pydata.org/docs/reference/general_functions.html#testing-functions) 