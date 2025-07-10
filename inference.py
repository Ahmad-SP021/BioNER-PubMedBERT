# inference.py
# --------------------------------------------------

import os, json, torch, torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel

# ---------- paths ----------
MODEL_DIR   = r"C:\Users\Andromeda\Desktop\MESys project\Model"   # where ner_model.pt + label2id.json live
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")              # downloaded once from hub
WEIGHT_PATH = os.path.join(MODEL_DIR, "ner_model.pt")            # your fine‑tuned weights

# ---------- label map ----------
with open(os.path.join(MODEL_DIR, "label2id.json"), "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

# ---------- model definition (same as training) ----------
class NERModel(nn.Module):
    def __init__(self, config_path: str, num_labels: int):
        super().__init__()
        # build PubMedBERT *skeleton* from local config (no weights)
        bert_config = AutoConfig.from_pretrained(config_path)
        self.bert = AutoModel.from_config(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        return self.classifier(x)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- build + load weights ----------
model = NERModel(CONFIG_PATH, NUM_LABELS)
state = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)  # local tokenizer

print(f"✅ Model loaded on {device}")

# ---------- helper: predict entities in one sentence ----------
@torch.inference_mode()
def predict(sentence: str, max_len: int = 128):
    # tokenise & truncate
    tokens = tokenizer.tokenize(sentence)[: max_len - 2]
    tokens_full = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokens_full)
    attention = [1] * len(input_ids)

    pad = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad
    attention += [0] * pad

    # tensors
    ids_tensor = torch.tensor([input_ids], device=device)
    att_tensor = torch.tensor([attention], device=device)

    # forward
    logits = model(ids_tensor, att_tensor)
    preds = logits.argmax(dim=-1).squeeze().tolist()

    # skip CLS & SEP when mapping back
    tags = [
        id2label[pred] for idx, pred in enumerate(preds)
        if 0 < idx <= len(tokens)
    ]
    return list(zip(tokens, tags))

# ---------- demo ----------
if __name__ == "__main__":
    sent = (
    "The thoracic cavity contains vital organs such as the heart and lungs. "
    "Surrounding the spinal cord is the vertebral column, which protects this crucial part of the central nervous system. "
    "The abdominal cavity holds the stomach, liver, pancreas, and intestines. "
    "Muscles like the diaphragm and intercostal muscles support respiration. "
    "Blood vessels, including the aorta and inferior vena cava, run alongside the spine. "
    "The cranial cavity houses the brain and is protected by the skull. "
    "The pelvic region includes the bladder and reproductive organs, supported by pelvic floor muscles."
)
    print("\n🔍 Result:")
    for tok, tag in predict(sent):
        print(f"{tok:15} → {tag}")
