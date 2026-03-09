# Assignment 1 — Digital Signal Processing

Реализация кастомного слоя LogMelFilterBanks и обучение CNN-классификатора на датасете Google Speech Commands (бинарная классификация: `yes` / `no`).

---

## Структура проекта:

```
.
├── melbanks.py          # кастомный слой LogMelFilterBanks
├── main.py              # точка входа, обучение модели
├── src/
│   ├── model.py         # архитектура MelNet (Conv1d)
│   ├── train_module.py  # LightningModule
│   └── data.py          # датамодуль
├── experiments/         # логи tensorboard по экспериментам
├── staff/               # графики
└── dataset/             # скачивается автоматически
```

---

## Запуск обучения:

```bash
python main.py --n_mels 80 --n_groups 1
```

Датасет скачается автоматически в папку `dataset/`.

### Просмотр логов:

```bash
tensorboard --logdir ./experiments
```

Открыть в браузере: `http://localhost:6006`

---

Отчет оформлен в виде файла  REPORT.md