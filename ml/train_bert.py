import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true])
df = df[["text", "label"]]

# Reduce dataset for speed
df = df.sample(1000)

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2
)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

extra_data = [
    ("The claim that banks will shut down is false and services will continue.", 0),
    ("Authorities confirmed that viral rumors about bank shutdown are not true.", 0),
    ("This message about banks closing permanently is fake and misleading.", 0),
    ("Government denied rumors of bank shutdown and confirmed normal operations.", 0),
]

extra_df = pd.DataFrame(extra_data, columns=["text", "label"])

df = pd.concat([df, extra_df])
# Training loop
model.train()

for epoch in range(1):
    print(f"Epoch {epoch+1} started...")

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

print("Training completed!")

# Save model
model.save_pretrained("../models/bert_model")
tokenizer.save_pretrained("../models/bert_model")

print("Model saved successfully!")