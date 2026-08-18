# LSTM Text Generator — Generative AI Interview Task

## 1. Project objective

Build an LSTM-based text generator that learns from Shakespeare's public-domain works and predicts the next word from a sequence of previous words.

This implementation follows the supplied interview task: preprocessing, tokenization, next-token sequence creation, Embedding + LSTM + Dense softmax architecture, training/validation split, early stopping/checkpointing, and seed-based text generation.

## 2. Dataset

Project Gutenberg — Shakespeare's Complete Works:
https://www.gutenberg.org/files/100/100-0.txt

The Python script downloads the `.txt` file automatically into `data/`.

## 3. Project structure

```text
lstm-text-generator/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── shakespeare.txt        # downloaded automatically, not committed
├── models/
│   └── best_lstm.keras        # generated after training
└── training_history.npz       # generated after training
```

## 4. How to run

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Training can take a while depending on your CPU/GPU.

## 5. What the code does

1. Downloads Shakespeare's text.
2. Converts text to lowercase.
3. Removes punctuation and non-letter characters.
4. Tokenizes the corpus into words.
5. Creates 30-word input sequences with the following word as the target.
6. Randomly splits examples into 90% training and 10% validation data.
7. Builds an Embedding → LSTM → Dropout → Dense(softmax) model.
8. Uses Adam optimizer and sparse categorical crossentropy.
9. Uses EarlyStopping and ModelCheckpoint.
10. Generates text from multiple seed phrases.

## 6. Example seed inputs

- `to be or not to be`
- `all the world`
- `love is`

The exact generated text depends on the trained model and random sampling, so run the program to obtain the actual outputs for your submission.

## 7. Bonus experiment

Try changing:

- `SEQ_LENGTH = 20`, `30`, `50`
- `LSTM_UNITS = 128`, `256`, `512`
- Add a second LSTM layer with `return_sequences=True`
- Change generation temperature between `0.5` and `1.0`

Record validation loss and a few generated samples for each experiment.

## 8. Submission checklist

- [ ] `main.py`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] GitHub repository URL
- [ ] Dataset URL
- [ ] 2–3 generated text samples
- [ ] Short note about bonus experiments, if completed

Do not commit the downloaded dataset or large model file unless the recruiter specifically asks for them. The `.gitignore` excludes them by default.
