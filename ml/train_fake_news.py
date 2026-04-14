import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 1   # FAKE
true["label"] = 0   # REAL

# Combine datasets
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)
# Keep only needed columns
df = df[["text", "label"]]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF (IMPORTANT upgrade)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1,3),
    max_features=10000

)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model (BEST for text)
model = LogisticRegression(max_iter=2000, C=2)
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")