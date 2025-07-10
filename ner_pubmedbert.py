import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================
# Step 1: Read IOB data
# =========================
def read_iob_data(filepath):
    sentences = []
    labels = []
    sentence = []
    label_seq = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
                continue
            parts = line.strip().split("\t")
            if len(parts) == 2:
                token, tag = parts
                sentence.append(token)
                label_seq.append(tag)

    if sentence:
        sentences.append(sentence)
        labels.append(label_seq)

    return sentences, labels


# =========================
# Step 2: Encode Inputs
# =========================
def encode_examples(sentences, labels, label2id, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    label_ids = []

    for sent, label_seq in zip(sentences, labels):
        tokens = []
        label_ids_seq = []

        for word, label in zip(sent, label_seq):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)

            label_ids_seq.extend([label2id[label]] + [-100]*(len(word_tokens)-1))

        tokens = tokens[:max_len-2]
        label_ids_seq = label_ids_seq[:max_len-2]

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids_seq = [-100] + label_ids_seq + [-100]

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1]*len(input_id)

        pad_len = max_len - len(input_id)
        input_id += [tokenizer.pad_token_id]*pad_len
        attention_mask += [0]*pad_len
        label_ids_seq += [-100]*pad_len

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        label_ids.append(label_ids_seq)

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(label_ids)


# =========================
# Step 3: Create Dataloader
# =========================
def create_dataloader(input_ids, attention_masks, label_ids, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, label_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =========================
# Run Preprocessing
# =========================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
    sentences, labels = read_iob_data(r"C:\Users\Andromeda\Desktop\MESys project\AnatEM-IOB\train.tsv")

    # create label mapping
    unique_tags = sorted(set(tag for seq in labels for tag in seq))
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    print(f"Labels: {label2id}")

    input_ids, attention_masks, label_ids = encode_examples(sentences, labels, label2id, tokenizer)

    train_loader = create_dataloader(input_ids, attention_masks, label_ids)
    print(f"✅ Dataloader ready with {len(train_loader)} batches.")


# =========================
# Step 4: Define the NER Model using PubMedBERT and a Linear Classifier
# =========================
class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        # Load pre-trained PubMedBERT
        self.bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        # Linear classification layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        logits = self.classifier(sequence_output)    # (batch_size, seq_len, num_labels)
        return logits


# =========================
# Step 5: Model Training Loop (Loss, Optimizer, and Backpropagation)
# =========================
class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    trust_remote_code=True,
    use_safetensors=True
)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Assuming label2id is already defined
num_labels = len(label2id)
model = NERModel(num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Using device:", device)

# Define optimizer and loss
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# Train for 3 epochs
epochs = 3
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        batch_input_ids, batch_mask, batch_labels = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        logits = model(batch_input_ids, batch_mask)
        
        # reshape for loss: [batch*seq_len, num_labels]
        loss = loss_fn(logits.view(-1, num_labels), batch_labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


# =========================
# Step 6: Evaluation on Validation or Test Set
# =========================
# Load and preprocess test data
test_sentences, test_labels = read_iob_data(r"C:\Users\Andromeda\Desktop\MESys project\AnatEM-IOB\test.tsv")
test_input_ids, test_attention_masks, test_label_ids = encode_examples(test_sentences, test_labels, label2id, tokenizer)
test_loader = create_dataloader(test_input_ids, test_attention_masks, test_label_ids)

# Evaluation
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, mask, true_labels = [x.to(device) for x in batch]
        logits = model(input_ids, mask)
        predictions = torch.argmax(logits, dim=-1)

        for pred, true in zip(predictions, true_labels):
            for p, t in zip(pred, true):
                if t != -100:
                    all_preds.append(p.item())
                    all_true.append(t.item())

# Report
target_names = [tag for tag, idx in sorted(label2id.items(), key=lambda x: x[1])]
print("\n Evaluation Report:")
print(classification_report(all_true, all_preds, target_names=target_names))

# =========================
# Step 7 - Save model and tokenizer
# =========================
model_path = r"C:\Users\Andromeda\Desktop\MESys project\Model"
os.makedirs(model_path, exist_ok=True)

# Save model state dict (weights only)
torch.save(model.state_dict(), os.path.join(model_path, "ner_model.pt"))

# Save label2id dictionary
import json
with open(os.path.join(model_path, "label2id.json"), "w") as f:
    json.dump(label2id, f)

# Save tokenizer
tokenizer.save_pretrained(model_path)

print("✅ Model, tokenizer, and label mapping saved.")
