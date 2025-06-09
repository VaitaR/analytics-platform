# Тесты для обнаружения и исправления Polars Fallback

Эти тесты созданы для выявления ситуаций, когда оптимизированная реализация Polars "тихо" падает и переключается на стандартную реализацию Pandas, что приводит к снижению производительности без явных ошибок.

## Обзор проблемы

В текущей реализации FunnelCalculator используется три уровня вычислений с автоматическим переключением:

1. **Оптимизированная Polars реализация** (быстрая) - первая попытка
2. **Стандартная Polars реализация** (средняя) - первый fallback
3. **Pandas реализация** (медленная) - последний fallback

Текущие тесты просто проверяют, что функции "работают", но не проверяют, что они используют оптимизированную реализацию. Это приводит к ситуации, когда тесты проходят успешно, но код фактически выполняется медленнее, чем мог бы.

## Набор тестов

### 1. Обнаружение Fallback (test_polars_fallback_detection.py)

Эти тесты проверяют, что оптимизированные реализации действительно используются, а не происходит тихий fallback:

- `test_path_analysis_fallback_detection` - Проверяет, что path analysis не делает fallback
- `test_cohort_analysis_fallback_detection` - Проверяет, что cohort analysis не делает fallback
- `test_problematic_lazy_frame_path_analysis` - Специально проверяет проблему с LazyFrame
- `test_detect_polars_to_pandas_fallback_combinations` - Проверяет все комбинации настроек для выявления тех, что вызывают fallback
- `test_lazy_frame_in_path_analysis` - Эмулирует проблему с LazyFrame через monkey patching

### 2. Специфические тесты ошибки с LazyFrame (test_lazy_frame_bug.py)

- `test_reproduce_lazy_frame_error` - Воспроизводит ошибку с LazyFrame в path analysis
- `test_fix_lazy_frame_error` - Показывает как исправить проблему
- `test_lazy_frame_error_fix_suggestion` - Демонстрирует предложенное решение из сообщения об ошибке

### 3. Исправление ошибки с LazyFrame (test_path_analysis_fix.py)

- `test_fixed_implementation_with_regular_dataframes` - Проверяет, что исправленная реализация работает с обычными DataFrame
- `test_fixed_implementation_with_lazy_frames` - Проверяет, что исправленная реализация правильно обрабатывает LazyFrame
- `test_integration_with_calculator` - Проверяет интеграцию исправления с FunnelCalculator
- `test_error_handling` - Проверяет корректную обработку ошибок
- `test_suggested_code_fix` - Демонстрирует различные подходы к решению проблемы

### 4. Полное тестирование всех комбинаций (test_fallback_comprehensive.py)

Этот набор тестов проверяет все 12 возможных комбинаций параметров FunnelCalculator:
- funnel_order: ORDERED, UNORDERED (2 значения)
- reentry_mode: FIRST_ONLY, OPTIMIZED_REENTRY (2 значения)  
- counting_method: UNIQUE_USERS, EVENT_TOTALS, UNIQUE_PAIRS (3 значения)

Тесты:
- `test_no_fallback_in_funnel_calculation` - Проверяет отсутствие fallback во всех комбинациях
- `test_component_specific_fallback_detection` - Проверяет каждый компонент отдельно (path_analysis, time_to_convert, cohort_analysis)
- `test_path_analysis_int64_uint32_error` - Тестирует конкретную ошибку "type Int64 is incompatible with expected type UInt32"
- `test_path_analysis_nested_object_types_error` - Тестирует ошибку "not yet implemented: Nested object types"
- `test_lazy_frame_error` - Воспроизводит и проверяет ошибку с LazyFrame
- `test_documentation_of_fallback_patterns` - Генерирует отчет о том, какие комбинации вызывают fallback

Эти тесты специально фейлятся, если обнаруживают fallback, что помогает выявить проблемы с производительностью.

## Запуск тестов

Для запуска различных типов тестов можно использовать следующие команды:

```bash
# Запуск тестов обнаружения fallback
python run_tests.py --fallback-detection

# Запуск тестов для проверки исправления
python run_tests.py --path-analysis-fix

# Запуск стандартных тестов на fallback
python run_tests.py --fallbacks

# Запуск полного тестирования всех комбинаций
python run_tests.py --comprehensive-fallback
```

## Исправление проблемы с LazyFrame

Основная проблема связана с тем, что в методе `_calculate_path_analysis_polars_optimized` происходит попытка использовать LazyFrame в операции `is_in()`, что приводит к ошибке:

```
cannot create expression literal for value of type LazyFrame.
Hint: Pass `allow_object=True` to accept any value and create a literal of type Object.
```

Исправление заключается в следующем:

1. Проверять тип входящих параметров в начале функции
2. Преобразовывать LazyFrame в DataFrame с помощью `.collect()`
3. Использовать полученные DataFrame вместо LazyFrame

Пример исправления:

```python
def _calculate_path_analysis_polars_optimized(self, 
                                 segment_funnel_events_df, 
                                 funnel_steps,
                                 full_history_for_segment_users
                                ) -> PathAnalysisData:
    # Ensure we're working with DataFrames, not LazyFrames
    if hasattr(segment_funnel_events_df, 'collect'):
        segment_funnel_events_df = segment_funnel_events_df.collect()
        
    if hasattr(full_history_for_segment_users, 'collect'):
        full_history_for_segment_users = full_history_for_segment_users.collect()
    
    # Остальной код остается прежним
    # ...
```

## Другие обнаруженные проблемы

Помимо проблемы с LazyFrame, были обнаружены и другие причины fallback:

1. **Проблема с типами Int64/UInt32**: 
   ```
   type Int64 is incompatible with expected type UInt32
   ```
   Возникает при преобразовании типов в path analysis.

2. **Проблема с вложенными объектами**:
   ```
   not yet implemented: Nested object types
   ```
   Возникает при попытке обработать сложные структуры данных.

## Рекомендации

1. **Не использовать тихий fallback**: Лучше явно сообщать о проблемах, чем тихо переключаться на более медленную реализацию
2. **Проверять случаи fallback в тестах**: Тесты должны проверять не только "работает/не работает", но и "работает оптимально"
3. **Добавить проверку типов**: Проверять и преобразовывать входные параметры в начале функций
4. **Явно обрабатывать LazyFrame**: Всегда явно преобразовывать LazyFrame в DataFrame перед использованием
5. **Исправить проблемы с типами данных**: Обеспечить корректное преобразование типов Int64/UInt32
6. **Упростить обработку вложенных структур**: Избегать сложных вложенных объектов или добавить их поддержку 