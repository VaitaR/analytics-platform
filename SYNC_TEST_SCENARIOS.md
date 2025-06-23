# 🧪 Тестовые сценарии синхронизации систем

## 📋 Обзор

Этот документ описывает сценарии для проверки того, как изменения в одном файле влияют на работу всех систем.

## 🎯 Тестовые сценарии

### Сценарий 1: Изменение конфигурации MyPy в pyproject.toml

**Что меняем:**
```toml
[tool.mypy]
check_untyped_defs = true  # Меняем с false на true
```

**Ожидаемое поведение:**
- ✅ Локально: `make check` использует новую настройку
- ✅ GitHub: workflows используют ту же настройку через `make check`
- ✅ Автоматическая синхронизация: ДА

**Проверочные команды:**
```bash
make check                    # Должен применить новую настройку
ruff check . --output-format=github  # Должен работать как в CI
```

### Сценарий 2: Добавление новой версии Python

**Что меняем:**
```toml
# pyproject.toml
requires-python = ">=3.8"
target-version = "py38"
```

**И в workflows:**
```yaml
# .github/workflows/tests.yml
python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
```

**Ожидаемое поведение:**
- ⚠️ Требует изменений в 2 местах
- ✅ После изменения: полная синхронизация
- ❌ Автоматическая синхронизация: НЕТ

### Сценарий 3: Изменение команды в Makefile

**Что меняем:**
```makefile
lint:
    ruff check . --select E,W,F  # Добавляем селектор
    mypy . --strict              # Добавляем --strict
```

**Ожидаемое поведение:**
- ✅ Локально: `make lint` использует новые параметры
- ✅ GitHub: `make check` (который вызывает `make lint`) использует новые параметры
- ✅ Автоматическая синхронизация: ДА

### Сценарий 4: Добавление новой зависимости

**Что меняем:**
```txt
# requirements-dev.txt
bandit>=1.7.0  # Добавляем новую зависимость для security check
```

**Ожидаемое поведение:**
- ✅ Локально: `make install-dev` установит новую зависимость
- ✅ GitHub: workflows установят через `pip install -r requirements-dev.txt`
- ✅ Автоматическая синхронизация: ДА

### Сценарий 5: Изменение команды тестирования

**Что меняем:**
```python
# run_tests.py
def run_pytest(...):
    cmd.extend(["--maxfail=1"])  # Добавляем новый параметр
```

**Ожидаемое поведение:**
- ✅ Локально: все команды `python run_tests.py` используют новый параметр
- ✅ GitHub: все шаги workflows используют новый параметр
- ✅ Автоматическая синхронизация: ДА

## 📊 Матрица синхронизации

| Файл изменения | Локальная разработка | GitHub Actions | Автосинхронизация |
|----------------|---------------------|----------------|-------------------|
| `pyproject.toml` (linter config) | ✅ | ✅ | ✅ |
| `pyproject.toml` (Python version) | ✅ | ⚠️ | ❌ |
| `Makefile` (команды) | ✅ | ✅ | ✅ |
| `requirements*.txt` | ✅ | ✅ | ✅ |
| `run_tests.py` | ✅ | ✅ | ✅ |
| `.github/workflows/*.yml` | ❌ | ✅ | ❌ |

## 🚨 Критические точки рассинхронизации

### 1. Версии Python
**Проблема:** Нужно обновлять в 2 местах
```toml
# pyproject.toml
requires-python = ">=3.X"
target-version = "pyXX"
```
```yaml
# .github/workflows/tests.yml
python-version: ["3.X", ...]
```

**Решение:** Проверочная команда
```bash
make validate-python-versions  # TODO: добавить в Makefile
```

### 2. Прямые команды в workflows
**Проблема:** Если в workflow есть прямые команды вместо `make`
```yaml
# ❌ Плохо - может рассинхрониться
- run: ruff check . --select E,W,F

# ✅ Хорошо - всегда синхронизировано
- run: make check
```

### 3. Пути к файлам и исключения
**Проблема:** Разные пути в разных системах
```toml
# pyproject.toml
exclude = ["tests/conftest.py"]
```
```yaml
# workflow
--exclude tests/conftest.py
```

## 🛠️ Рекомендации по поддержанию синхронизации

### 1. Используйте make команды в workflows
```yaml
# ✅ Хорошо
- run: make check
- run: make test

# ❌ Плохо
- run: ruff check .
- run: python -m pytest
```

### 2. Централизуйте конфигурации
```toml
# pyproject.toml - единый источник истины для всех инструментов
[tool.ruff]
[tool.mypy]
[tool.pytest.ini_options]
```

### 3. Добавьте проверки валидации
```makefile
validate-sync:
    @echo "Проверка синхронизации конфигураций..."
    # Проверить версии Python
    # Проверить соответствие команд
    # Проверить зависимости
```

## 🧪 Команды для тестирования синхронизации

```bash
# Проверить локальную среду
make ci-check

# Проверить GitHub Actions формат
ruff check . --output-format=github

# Проверить совместимость команд
python run_tests.py --validate

# Симулировать CI среду
make clean && make install-dev && make check && make test
```

## 🎯 Заключение

**Автоматически синхронизируются:**
- ✅ Конфигурации linter'ов (через pyproject.toml)
- ✅ Команды тестирования (через run_tests.py)
- ✅ Make команды (через Makefile)
- ✅ Зависимости (через requirements*.txt)

**Требуют ручной синхронизации:**
- ⚠️ Версии Python (pyproject.toml + workflows)
- ⚠️ Прямые команды в workflows (если не используют make)

**Рекомендация:** Всегда используйте `make` команды в GitHub workflows для максимальной синхронизации!
