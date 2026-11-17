# Nuraidar Final Programming

Final project: Tiny Transformer for legal act classification (Kazakh text).

**Language / Язык:** [English](#english) | [Русский](#русский)

---

<a id="english"></a>

# English

## Description

The project classifies Kazakh legal acts by their form (`act_form`, e.g. *Шешім* = decision, *Қаулы* = resolution, *Бұйрық* = order) using only the **title** of the act.
Dataset: [KazakhLawCorpus-clean](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean) (215,889 documents).
It is downloaded automatically from Hugging Face inside the notebook with `load_dataset`, so no data files are stored in this repository.

Three approaches are compared on a **temporal split** (train on older acts, test on the newest ones):

1. TF-IDF (word and character n-grams) + Logistic Regression (baseline)
2. A small Transformer trained from scratch with a custom BPE tokenizer (up to 10M parameters; the actual model has 4.7M)
3. Fine-tuned `intfloat/multilingual-e5-small` (118M parameters)

Author: Mambetaly Nuraidar, ADA-2602M.

## Notebook structure

0. Setup
1. Data loading
2. EDA (missing values, target, duplicates, title length, distribution over time)
3. Preprocessing and temporal train/val/test split
4. Data leakage checks (exact duplicates, near-duplicates via cosine similarity on character n-grams)
5. SQLite experiment log and shared functions
6. Baseline: TF-IDF + Logistic Regression (learning curve, choice of C)
7. Common training loop for transformers
8. Tiny Transformer from scratch with BPE tokenizer
9. Fine-tuning multilingual-e5-small
10. Model comparison
11. Error analysis
12. Predictions for custom input text
13. Interactive input

## Experimental setup

| Item | Value |
|---|---|
| Documents | 215,889 |
| Target | The `act_form` field often lists several values (e.g. "Өзгерту, Шешім"), and the first one is often not the form of the act itself. So the label is the main act form taken from a fixed list of forms (*Шешім*, *Қаулы*, *Бұйрық*, *Заң*, *Жарлық*, *Өкім*, *Нормативтік қаулы*, etc.); if none is present, the first value is used. This gives 53 distinct labels |
| Classes used | **7**: *Шешім*, *Қаулы*, *Бұйрық*, *Өкім*, *Жарлық*, *Заң*, *Нормативтік қаулы*. Rarer classes are dropped (1.0% of train, 0.1% of val, 0.08% of test) |
| Split | Temporal: train up to 2023-03-25, validation 2023-03-27 to 2024-08-27, test 2024-08-28 to 2026-07-20 |
| Sizes after leakage cleaning | train 130,871 / val 18,231 / test 16,869 / strict test 14,368 |
| Main metric | **Macro-F1** (accuracy is reported too, but it is dominated by the biggest classes) |
| Training | Class-weighted loss for transformers, GPU (CUDA); e5 is trained on a 60K stratified subset of train (3 epochs) |

**Why macro-F1?** The classes are imbalanced: *Шешім* is about 60% of test and *Заң* or *Нормативтік қаулы* only about 1% or less. A model that always predicts the majority class gets high accuracy but a low macro-F1. Macro-F1 averages F1 over classes equally, so rare forms count as much as common ones.

**Why a temporal split?** A random split would put near-identical acts (same template, different district or year) on both sides and inflate the scores. Splitting by adoption date mimics real use: the model sees only the past and is evaluated on future acts.

## Data leakage checks

- No `law_id` appears in more than one split.
- However, **15.6% of val and 11.4% of test titles exactly match a title in train** (many acts are templates such as "On establishing a public easement"). These were removed from val/test, and duplicate (title, label) pairs were removed from train (about 40K rows).
- A cosine-similarity search on character n-grams shows that **42.8% of test titles have a near-duplicate in train** (similarity >= 0.8). So an extra **strict test** set (without near-duplicates) is reported next to the normal test.
- The label of the nearest train neighbour matches the test label in about 95% of cases when similarity is at least 0.8 (92.5% for exact copies), and a plain 1-nearest-neighbour classifier already gets 0.915 accuracy on the whole test set. Only 798 distinct titles in the dataset carry more than one label, so label noise is small but not zero.

## Results

Test = acts adopted after 2024-08-28 (cleaned of exact duplicates). Strict test = additionally without near-duplicates of train.

| Model | Params | Size, MB | Val F1-macro | **Test F1-macro** | Strict test F1-macro | Test accuracy | CPU, s / 1000 texts | Train time, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | 1.7M | 24.5 | 0.902 | **0.905** | 0.888 | 0.958 | 0.46 | 467 |
| Tiny Transformer (from scratch) | 4.7M | 18.6 | 0.847 | **0.852** | 0.827 | 0.951 | 7.9 | 364 |
| multilingual-e5-small (fine-tuned) | 117.7M | 465.2 | 0.915 | **0.894** | 0.879 | 0.956 | 90.8 | 572 |

(TF-IDF has 242,165 features; its training time includes fitting the vectorizers. Times are from one GPU run in Colab, a single seed. The e5 model is trained on 60K examples, the other two on all 130,871.)

### What the results mean

- **The simple baseline is the best model.** TF-IDF reaches the highest test F1 (0.905) against 0.894 for fine-tuned e5, while being about 19x smaller on disk, about 200x faster on CPU and quicker to train. The gap to e5 is small and only one seed was used, so the safe conclusion is "e5 is not better", not "e5 is clearly worse". For this task TF-IDF is the better engineering choice.
- **e5 wins on validation but not on test** (0.915 vs 0.894). The test period is newer and has a different class mix, and e5 was trained on a smaller subset (60K of 131K), which may also play a role.
- **The tiny transformer is about 5 F1 points behind but very small.** At 18.6 MB and 4.7M parameters it is the smallest model, 25x smaller than e5 and 11x faster on CPU. It is the weakest of the three on rare classes (e.g. *Нормативтік қаулы* F1 0.64 versus 0.83 for TF-IDF and 0.88 for e5), because it has no pre-training and few examples of them.
- **The strict test is lower by 1.5-2.5 points** (0.888 vs 0.905 for TF-IDF; 0.879 vs 0.894 for e5; 0.827 vs 0.852 for the tiny transformer). Near-duplicates of train do make the test a little easier, but the ranking of the models stays the same, and the scores on genuinely new titles are still high.
- **Accuracy (0.95-0.96) is higher than macro-F1 (0.85-0.90)** because all models are excellent on the big classes and weaker on the small ones.
- **Validation and test scores are close** for TF-IDF (0.902 and 0.905) and the tiny transformer (0.847 and 0.852). The class mix does change over time (*Шешім* grows from 35% of train to 60% of test, *Қаулы* falls from 42% to 20%), but with 7 classes this shift does not hurt much.

### Learning curve

TF-IDF validation macro-F1 grows from 0.737 (2K training examples) to 0.766 (5K), 0.804 (10K), 0.816 (20K), 0.832 (40K), 0.840 (80K) and 0.899 (all 130,871). Gains are slow between 20K and 80K, and then there is a visible jump with the full data, so more data of the same kind still helps. This is why the e5 subset of 60K is a compromise made for training time, not a point where the curve is flat. The final C (16) was chosen from {1, 4, 16} by validation F1 (0.8905, 0.8992, 0.9016).

## Error analysis

Done for the best model by validation score (fine-tuned e5; 4.4% of test titles are misclassified).

- **Per class (F1 for e5 / TF-IDF / tiny):** *Шешім* 0.98 / 0.98 / 0.98, *Заң* 0.98 / 0.97 / 0.97, *Бұйрық* 0.95 / 0.95 / 0.94, *Қаулы* 0.91 / 0.91 / 0.89, *Өкім* 0.79 / 0.88 / 0.80, *Жарлық* 0.78 / 0.81 / 0.75, *Нормативтік қаулы* 0.88 / 0.83 / 0.64. The weak classes (*Өкім* 120, *Жарлық* 125, *Нормативтік қаулы* 60 test examples) are small, so their F1 is unstable.
- **Most frequent confusions (e5):** *Шешім* predicted as *Қаулы* (210 cases), *Қаулы* as *Шешім* (148), *Қаулы* as *Бұйрық* (136), *Бұйрық* as *Қаулы* (91). These are the three biggest classes. The title alone often does not show which body issued the act (e.g. a local council issues a "Шешім", a government a "Қаулы"), so some of these cases are ambiguous.
- **Confidence is informative.** The error rate is 55% when the top-class probability is at most 0.5, 40% for 0.5-0.7, 2.4% for 0.7-0.9 and 0% above 0.9 (171 cases). So low-confidence predictions can be sent to a human reviewer, and high-confidence ones can be trusted.
- **Models agree most of the time.** All three are correct on 93.0% of test titles and all three are wrong on 2.4%. An oracle that picks the best model per example would reach 97.6% accuracy, so an ensemble could give a small gain.
- **Short titles (1-5 words) are the hardest** (32% errors), but there are only 56 of them; other length groups have 2-8% errors, with the lowest error rate for long titles (21+ words: about 2%). Errors grow over time (3.2% in 2024, 4.4% in 2025, 7.9% in 2026), which suggests some drift. On the strict test the error rate is 4.7% against 2.4% for the near-duplicates.

## Conclusions

1. The form of a Kazakh legal act can be predicted from its title with macro-F1 of about 0.90 and accuracy of about 0.96 on a temporal test set.
2. A strong, cheap TF-IDF + Logistic Regression baseline is the best model here; pre-trained e5 is slightly worse on test and far larger and slower, and the tiny transformer trained from scratch is clearly worse but 25x smaller than e5 and runs without a GPU.
3. Always evaluate on a temporal split and check for duplicates: 43% of test titles have a near-duplicate in train, and removing them lowers the scores by 1.5-2.5 points.
4. Possible improvements: ensembling, more training data for e5 (all 131K instead of 60K), and metadata available at prediction time (issuing body, region).

Limitations: a single run per model (no confidence intervals), only 7 of 53 classes are used, the best C is at the edge of the searched grid (1, 4, 16), e5 is trained on 60K examples instead of all 131K, and the title is the only input.

## How to run

The notebook runs in Google Colab, VS Code or Jupyter. A GPU is recommended for section 9 (training on CPU is much slower).

```
pip install datasets tokenizers transformers scikit-learn
```

Other libraries used: numpy, pandas, matplotlib, joblib, torch. Then open `Nuraidar_FINAL.ipynb` and run all cells. The last cell waits for keyboard input, so it blocks "Run all" at the very end; everything important is saved before it.

In the notebook, trained artifacts (models, tokenizer, `experiments.db`, cached features) are saved to a results folder on Google Drive and reloaded on the next run, so re-running does not retrain everything. The heavy ones are not stored in this repository; see the Google Drive link in Files.

## Files

- `Nuraidar_FINAL.ipynb` - full project with code, outputs and results
- `README.md` - this file
- `results/` - small result files of the training runs:
  - `experiments.db` - SQLite experiment log (tables `experiments`, `per_class`, `predictions`)
  - `learning_curve.csv` - TF-IDF learning curve
  - `tfidf_best_c.json`, `tfidf_meta.json`, `tiny_meta.json` - best C, training time and sequence length

**Additional files on Google Drive** (trained models such as `tiny_transformer.pt`, `tfidf_lr.joblib` and the fine-tuned e5 `e5_small_cls`, tokenizer `tiny_bpe.json`, `near_dup.parquet`, etc.): [Google Drive folder](https://drive.google.com/drive/folders/1AWQdl28N11Vrg7Mm_RUY9kg-AE_D2cX_?usp=sharing)

---

<a id="русский"></a>

# Русский

## Описание

Проект классифицирует нормативные правовые акты Казахстана по их форме (`act_form`, например *Шешім* = решение, *Қаулы* = постановление, *Бұйрық* = приказ), используя только **название** акта.
Датасет: [KazakhLawCorpus-clean](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean) (215 889 документов).
Он автоматически загружается с Hugging Face внутри ноутбука через `load_dataset`, поэтому файлы с данными в репозитории не хранятся.

Три подхода сравниваются на **временном разбиении** (обучение на более старых актах, тест на самых новых):

1. TF-IDF (словные и символьные n-граммы) + Logistic Regression (базовая модель)
2. Небольшой трансформер, обученный с нуля, с собственным BPE-токенизатором (до 10 млн параметров; фактически 4,7 млн)
3. Дообученная `intfloat/multilingual-e5-small` (118 млн параметров)

Автор: Мамбеталы Нурайдар, ADA-2602M.

## Структура ноутбука

0. Окружение
1. Загрузка данных
2. EDA (пропуски, целевая переменная, дубликаты, длина названий, распределение по времени)
3. Предобработка и временное разбиение train/val/test
4. Проверка утечки данных (точные дубликаты, почти-дубликаты через косинусное сходство по символьным n-граммам)
5. Журнал экспериментов в SQLite и общие функции
6. Baseline: TF-IDF + Logistic Regression (кривая обучения, подбор C)
7. Общий цикл обучения для трансформеров
8. Компактный трансформер с нуля и BPE-токенизатор
9. Дообучение multilingual-e5-small
10. Сравнение моделей
11. Анализ ошибок
12. Предсказание для введённого текста
13. Интерактивный ввод

## Постановка эксперимента

| Параметр | Значение |
|---|---|
| Документов | 215 889 |
| Целевая переменная | В поле `act_form` часто перечислено несколько значений (например «Өзгерту, Шешім»), и первое из них нередко не является формой самого акта. Поэтому метка это основная форма акта из заданного списка форм (*Шешім*, *Қаулы*, *Бұйрық*, *Заң*, *Жарлық*, *Өкім*, *Нормативтік қаулы* и др.); если ни одной из них нет, берётся первое значение. Получается 53 различные метки |
| Используемые классы | **7**: *Шешім*, *Қаулы*, *Бұйрық*, *Өкім*, *Жарлық*, *Заң*, *Нормативтік қаулы*. Более редкие классы отброшены (1,0% train, 0,1% val, 0,08% test) |
| Разбиение | Временное: train до 2023-03-25, validation с 2023-03-27 по 2024-08-27, test с 2024-08-28 по 2026-07-20 |
| Размеры после очистки от утечек | train 130 871 / val 18 231 / test 16 869 / strict test 14 368 |
| Основная метрика | **Macro-F1** (accuracy тоже приведена, но она определяется самыми крупными классами) |
| Обучение | Взвешенная по классам функция потерь для трансформеров, GPU (CUDA); e5 обучается на стратифицированной подвыборке train в 60 тыс. (3 эпохи) |

**Почему macro-F1?** Классы несбалансированы: *Шешім* составляет около 60% test, а *Заң* и *Нормативтік қаулы* около 1% и меньше. Модель, всегда предсказывающая самый частый класс, получит высокую accuracy, но низкий macro-F1. Macro-F1 усредняет F1 по классам с равным весом, поэтому редкие формы важны так же, как частые.

**Почему временное разбиение?** При случайном разбиении почти одинаковые акты (один шаблон, другой район или год) попадали бы и в train, и в test, и метрики были бы завышены. Разбиение по дате принятия имитирует реальное использование: модель видит только прошлое, а оценивается на будущих актах.

## Проверка утечки данных

- Ни один `law_id` не встречается более чем в одной части.
- Однако **15,6% названий в val и 11,4% в test точно совпадают с названием из train** (много шаблонных актов, например «Об установлении публичного сервитута»). Они удалены из val/test, а дубликаты (название, метка) удалены из train (около 40 тыс. строк).
- Поиск по косинусному сходству символьных n-грамм показывает, что у **42,8% названий из test есть почти-дубликат в train** (сходство >= 0,8). Поэтому дополнительно считается **strict test** (без почти-дубликатов) рядом с обычным test.
- Метка ближайшего соседа из train совпадает с меткой в test примерно в 95% случаев при сходстве не ниже 0,8 (92,5% для точных копий), а простой классификатор 1-ближайшего соседа уже даёт accuracy 0,915 на всём test. Только 798 различных названий во всём датасете имеют больше одной метки, поэтому шум разметки небольшой, но не нулевой.

## Результаты

Test = акты, принятые после 2024-08-28 (очищены от точных дубликатов). Strict test = дополнительно без почти-дубликатов train.

| Модель | Параметры | Размер, МБ | Val F1-macro | **Test F1-macro** | Strict test F1-macro | Test accuracy | CPU, с / 1000 текстов | Время обучения, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | 1,7 млн | 24,5 | 0,902 | **0,905** | 0,888 | 0,958 | 0,46 | 467 |
| Tiny Transformer (с нуля) | 4,7 млн | 18,6 | 0,847 | **0,852** | 0,827 | 0,951 | 7,9 | 364 |
| multilingual-e5-small (дообучена) | 117,7 млн | 465,2 | 0,915 | **0,894** | 0,879 | 0,956 | 90,8 | 572 |

(У TF-IDF 242 165 признаков; время обучения включает обучение векторизаторов. Время получено в одном запуске на GPU в Colab, один seed. e5 обучена на 60 тыс. примеров, остальные две модели на всех 130 871.)

### Что означают результаты

- **Простая базовая модель оказалась лучшей.** TF-IDF даёт самый высокий test F1 (0,905) против 0,894 у дообученной e5, при этом примерно в 19 раз меньше на диске, примерно в 200 раз быстрее на CPU и быстрее обучается. Разрыв с e5 небольшой, а использован один seed, поэтому надёжный вывод это «e5 не лучше», а не «e5 явно хуже». Для этой задачи TF-IDF выгоднее с инженерной точки зрения.
- **e5 выигрывает на validation, но не на test** (0,915 против 0,894). Тестовый период новее и с другим составом классов, и e5 обучалась на меньшей подвыборке (60 тыс. из 131 тыс.), что тоже могло повлиять.
- **Маленький трансформер хуже примерно на 5 пунктов F1, но очень компактный.** При 18,6 МБ и 4,7 млн параметров это самая маленькая модель: в 25 раз меньше e5 и в 11 раз быстрее на CPU. Он самый слабый из трёх на редких классах (например, у *Нормативтік қаулы* F1 0,64 против 0,83 у TF-IDF и 0,88 у e5), потому что у него нет предобучения, а примеров таких классов мало.
- **Strict test ниже на 1,5-2,5 пункта** (0,888 против 0,905 у TF-IDF; 0,879 против 0,894 у e5; 0,827 против 0,852 у маленького трансформера). Почти-дубликаты train действительно немного облегчают test, но порядок моделей не меняется, и качество на действительно новых названиях остаётся высоким.
- **Accuracy (0,95-0,96) выше macro-F1 (0,85-0,90)**, потому что все модели отлично справляются с крупными классами и слабее с мелкими.
- **Оценки на validation и test близки** у TF-IDF (0,902 и 0,905) и маленького трансформера (0,847 и 0,852). Состав классов со временем действительно меняется (*Шешім* растёт с 35% train до 60% test, *Қаулы* падает с 42% до 20%), но при 7 классах этот сдвиг сильно не вредит.

### Кривая обучения

Val macro-F1 у TF-IDF растёт от 0,737 (2 тыс. примеров) до 0,766 (5 тыс.), 0,804 (10 тыс.), 0,816 (20 тыс.), 0,832 (40 тыс.), 0,840 (80 тыс.) и 0,899 (все 130 871). Между 20 и 80 тыс. прирост медленный, а на полных данных виден заметный скачок, поэтому больше данных того же типа ещё помогает. Выбор подвыборки в 60 тыс. для e5 это компромисс по времени обучения, а не точка, где кривая уже плоская. Итоговое C (16) выбрано из {1, 4, 16} по val F1 (0,8905, 0,8992, 0,9016).

## Анализ ошибок

Выполнен для лучшей модели по validation (дообученная e5; неверно классифицировано 4,4% названий test).

- **По классам (F1 для e5 / TF-IDF / tiny):** *Шешім* 0,98 / 0,98 / 0,98, *Заң* 0,98 / 0,97 / 0,97, *Бұйрық* 0,95 / 0,95 / 0,94, *Қаулы* 0,91 / 0,91 / 0,89, *Өкім* 0,79 / 0,88 / 0,80, *Жарлық* 0,78 / 0,81 / 0,75, *Нормативтік қаулы* 0,88 / 0,83 / 0,64. Слабые классы (*Өкім* 120, *Жарлық* 125, *Нормативтік қаулы* 60 примеров в test) маленькие, поэтому их F1 нестабилен.
- **Самые частые путаницы (e5):** *Шешім*, предсказанный как *Қаулы* (210 случаев), *Қаулы* как *Шешім* (148), *Қаулы* как *Бұйрық* (136), *Бұйрық* как *Қаулы* (91). Это три самых крупных класса. По одному названию часто не видно, какой орган издал акт (например, маслихат издаёт «Шешім», правительство «Қаулы»), поэтому часть таких случаев неоднозначна.
- **Уверенность модели информативна.** Доля ошибок равна 55% при вероятности top-класса не выше 0,5, 40% при 0,5-0,7, 2,4% при 0,7-0,9 и 0% выше 0,9 (171 случай). Поэтому предсказания с низкой уверенностью можно отправлять на проверку человеку, а уверенным можно доверять.
- **Модели в основном согласны друг с другом.** Все три верны на 93,0% названий test и все три ошибаются на 2,4%. Оракул, выбирающий лучшую модель для каждого примера, дал бы accuracy 97,6%, поэтому ансамбль может дать небольшой прирост.
- **Самые трудные названия короткие (1-5 слов):** 32% ошибок, но таких всего 56; в остальных группах по длине ошибок 2-8%, меньше всего у длинных названий (21+ слов: около 2%). Ошибки растут со временем (3,2% в 2024, 4,4% в 2025, 7,9% в 2026), что указывает на некоторый дрейф. На strict test доля ошибок 4,7% против 2,4% на почти-дубликатах.

## Выводы

1. Форму нормативного акта Казахстана можно предсказать по названию с macro-F1 около 0,90 и accuracy около 0,96 на временном test.
2. Сильная и дешёвая базовая модель TF-IDF + Logistic Regression здесь лучшая; предобученная e5 чуть хуже на test, но намного больше и медленнее, а маленький трансформер с нуля заметно хуже, но в 25 раз меньше e5 и работает без GPU.
3. Нужно всегда оценивать на временном разбиении и проверять дубликаты: у 43% названий test есть почти-дубликат в train, и их удаление снижает оценки на 1,5-2,5 пункта.
4. Возможные улучшения: ансамблирование, больше обучающих данных для e5 (все 131 тыс. вместо 60 тыс.), метаданные, доступные при предсказании (орган, регион).

Ограничения: один запуск на модель (нет доверительных интервалов), используются только 7 из 53 классов, лучшее C находится на краю проверенной сетки (1, 4, 16), e5 обучена на 60 тыс. примеров вместо всех 131 тыс., единственный вход это название.

## Как запустить

Ноутбук работает в Google Colab, VS Code или Jupyter. Для раздела 9 рекомендуется GPU (на CPU обучение намного медленнее).

```
pip install datasets tokenizers transformers scikit-learn
```

Другие используемые библиотеки: numpy, pandas, matplotlib, joblib, torch. Затем откройте `Nuraidar_FINAL.ipynb` и выполните все ячейки. Последняя ячейка ждёт ввода с клавиатуры, поэтому в самом конце она блокирует «Run all»; всё важное сохраняется до неё.

В ноутбуке обученные артефакты (модели, токенизатор, `experiments.db`, кэшированные признаки) сохраняются в папку результатов на Google Drive и подгружаются при следующем запуске, поэтому повторный запуск не переобучает всё заново. Тяжёлые файлы в репозитории не хранятся; ссылка на Google Drive есть в разделе «Файлы».

## Файлы

- `Nuraidar_FINAL.ipynb` - полный проект с кодом, выводами и результатами
- `README.md` - этот файл
- `results/` - небольшие файлы с результатами обучения:
  - `experiments.db` - журнал экспериментов SQLite (таблицы `experiments`, `per_class`, `predictions`)
  - `learning_curve.csv` - кривая обучения TF-IDF
  - `tfidf_best_c.json`, `tfidf_meta.json`, `tiny_meta.json` - лучшее C, время обучения и длина последовательности

**Дополнительные файлы на Google Drive** (обученные модели `tiny_transformer.pt`, `tfidf_lr.joblib` и дообученная e5 `e5_small_cls`, токенизатор `tiny_bpe.json`, `near_dup.parquet` и др.): [папка Google Drive](https://drive.google.com/drive/folders/1AWQdl28N11Vrg7Mm_RUY9kg-AE_D2cX_?usp=sharing)
