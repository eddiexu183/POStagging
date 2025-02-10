import numpy as np
from datasets import load_dataset
import spacy
from spacy.tokens import Doc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from tqdm import tqdm

dataset = load_dataset("batterydata/pos_tagging")
train_subset, test_data = dataset["train"].select(range(1000)), dataset["test"]

nlp = spacy.load("en_core_web_md")

def get_embeddings_and_labels_with_context(split, window_size=1):
    X, y = [], []
    for example in tqdm(split, desc="Processing", unit="sent"):
        tokens, pos_tags = example["words"], example["labels"]
        doc = Doc(nlp.vocab, words=tokens)
        for i, tag in enumerate(pos_tags):
            current_vec = doc[i].vector
            left_vec = doc[i - window_size].vector if i - window_size >= 0 else np.zeros_like(current_vec)
            right_vec = doc[i + window_size].vector if i + window_size < len(doc) else np.zeros_like(current_vec)
            X.append(np.concatenate([left_vec, current_vec, right_vec]))
            y.append(tag)
    return np.array(X), np.array(y)

X_train, y_train = get_embeddings_and_labels_with_context(train_subset)
X_test, y_test = get_embeddings_and_labels_with_context(test_data)

encoder = LabelEncoder()
encoder.fit(y_train)

def encode_labels(labels, encoder, unk_token="UNK"):
    if unk_token not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, unk_token)
    return np.array([encoder.transform([label])[0] if label in encoder.classes_ else encoder.transform([unk_token])[0] for label in labels])

y_train_enc, y_test_enc = encode_labels(y_train, encoder), encode_labels(y_test, encoder)

param_grid = {"C": [0.1, 1], "solver": ["lbfgs"]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid,
                           cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
                           scoring="accuracy", n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train_enc)

print("Best hyperparameters:", grid_search.best_params_)
print("Cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

y_pred = grid_search.best_estimator_.predict(X_test)
print("Test accuracy: {:.2f}%".format(accuracy_score(y_test_enc, y_pred) * 100))