# LLM Bench — CRUXEval + CRUXEval-X + MMLU-CS

Тестирование локальных LLM на реальных бенчмарках с короткими ответами.

## Датасеты

| Suite | Источник | Языки | Формат ответа | Кол-во |
|---|---|---|---|---|
| `cruxeval` | [crux-eval/cruxeval](https://huggingface.co/datasets/crux-eval/cruxeval) | Python | exact value | ~1600 (800×2) |
| `cruxeval_x_cpp` | [CRUXEval-X](https://huggingface.co/datasets/xinyiwan/CRUXEval-X) | C++ | exact value | ~600 |
| `cruxeval_x_c` | CRUXEval-X | C | exact value | ~600 |
| `cruxeval_x_python` | CRUXEval-X | Python | exact value | ~600 |
| `mmlu_cs` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | — | A/B/C/D | ~100 |
| `mmlu_machine_learning` | cais/mmlu | — | A/B/C/D | ~100 |

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```bash
# Ollama (по умолчанию http://localhost:11434/v1)
python bench.py --model llama3

# Все суиты
python bench.py --model qwen2.5-coder:7b \
    --suites cruxeval cruxeval_x_cpp cruxeval_x_c mmlu_cs

# Быстрый тест — 50 вопросов на суит
python bench.py --model llama3 --limit 50

# llama.cpp / vLLM / LM Studio
python bench.py --model my-model --base-url http://localhost:8080/v1

# Параллельные запросы (если сервер тянет)
python bench.py --model llama3 --workers 4

# Подробный вывод ошибок
python bench.py --model llama3 --verbose

# Список доступных суитов
python bench.py --model x --list-suites
```

## Совместимые серверы

Любой OpenAI-compatible API:
- **Ollama**: `ollama serve` → `--base-url http://localhost:11434/v1`
- **llama.cpp**: `./llama-server -m model.gguf` → `--base-url http://localhost:8080/v1`
- **vLLM**: `vllm serve model` → `--base-url http://localhost:8000/v1`
- **LM Studio**: запустить сервер → `--base-url http://localhost:1234/v1`

## Результаты

Автоматически сохраняются в JSON с детализацией по каждому вопросу.
