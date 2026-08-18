import re
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "shakespeare.txt"
MODEL_FILE = MODEL_DIR / "best_lstm.keras"

SEQ_LENGTH = 30
EMBEDDING_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 128
EPOCHS = 20


def download_dataset():
    if DATA_FILE.exists():
        print(f"Using existing dataset: {DATA_FILE}")
        return

    print("Downloading dataset...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"Saved dataset to {DATA_FILE}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_sequences(text):
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts([text])

    encoded = tokenizer.texts_to_sequences([text])[0]

    sequences = []
    for i in range(SEQ_LENGTH, len(encoded)):
        sequences.append(encoded[i - SEQ_LENGTH:i + 1])

    sequences = np.asarray(sequences, dtype=np.int32)
    X = sequences[:, :-1]
    y = sequences[:, -1]

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size, X_train, y_train, X_val, y_val


def build_model(vocab_size):
    """Build Embedding -> LSTM -> Dropout -> Dense(softmax)."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH,)),
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM
        ),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def generate_text(model, tokenizer, seed_text, num_words=50, temperature=0.8):
    text = clean_text(seed_text)
    words = text.split()
    if not words:
        raise ValueError("Seed text must contain at least one valid word.")

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    for _ in range(num_words):
        token_ids = tokenizer.texts_to_sequences([" ".join(words)])[-1]
        token_ids = token_ids[-SEQ_LENGTH:]
        token_ids = pad_sequences([token_ids], maxlen=SEQ_LENGTH, padding="pre")

        predictions = model.predict(token_ids, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        probabilities = tf.nn.softmax(predictions).numpy()

        next_id = np.random.choice(len(probabilities), p=probabilities)
        next_word = index_to_word.get(next_id, "")

        if next_word and next_word != "<unk>":
            words.append(next_word)

    return " ".join(words)


def main():
    download_dataset()

    raw_text = DATA_FILE.read_text(encoding="utf-8", errors="ignore")
    start = raw_text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = raw_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        raw_text = raw_text[start:end]

    text = clean_text(raw_text)
    print(f"Cleaned corpus size: {len(text):,} characters")

    tokenizer, vocab_size, X_train, y_train, X_val, y_val = prepare_sequences(text)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Training examples: {len(X_train):,}")
    print(f"Validation examples: {len(X_val):,}")

    model = build_model(vocab_size)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    np.savez(
        "training_history.npz",
        loss=np.asarray(history.history["loss"]),
        val_loss=np.asarray(history.history["val_loss"]),
        accuracy=np.asarray(history.history["accuracy"]),
        val_accuracy=np.asarray(history.history["val_accuracy"]),
    )

    print("\n--- GENERATED TEXT SAMPLES ---")
    seeds = [
        "to be or not to be",
        "all the world",
        "love is",
    ]

    for seed in seeds:
        print(f"\nSeed: {seed}")
        print(generate_text(model, tokenizer, seed, num_words=50, temperature=0.8))

    print("\nTraining complete.")
    print(f"Best model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
