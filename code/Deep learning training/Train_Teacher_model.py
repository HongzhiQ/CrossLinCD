import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import warnings
import random

warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BATCH_SIZE = 8
LEARNING_RATE = 1e-6
NUM_EPOCHS = 200
LABEL_COLUMNS = ["情绪化推理", "以偏概全", "乱贴标签", "读心术", "先知错误", "非此即彼", "应该句式", "放大", "罪责归己", "心理过滤", "否定正面思考", "罪责归他", "无标签"]
ALL_LABEL_COLUMNS = ["all labels here"]
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 150
SEED = 44


set_seed(SEED)


def calculate_evaluation(prediction, true_label, mask):

    mask_indices = np.where(mask)[0]
    true_label_filtered = []
    prediction_filtered = []
    for t, p in zip(true_label, prediction):
        if t in mask_indices:
            true_label_filtered.append(t)
            prediction_filtered.append(p)

    true_label_filtered = np.array(true_label_filtered).astype(int)
    prediction_filtered = np.array(prediction_filtered).astype(int)
    print("true_label_filtered")
    print(true_label_filtered)
    print("prediction_filtered")
    print(prediction_filtered)
    
    report = classification_report(true_label_filtered, prediction_filtered, output_dict=True, zero_division=1)
    accuracy = accuracy_score(true_label_filtered, prediction_filtered)
    return accuracy, report

def read_tsv(file_path, label_columns, all_label_columns):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['Original text'].tolist()
    labels = np.zeros((len(df), len(all_label_columns)))
    actual_label_columns = [col for col in label_columns if col in df.columns]
    for col in actual_label_columns:
        labels[:, all_label_columns.index(col)] = df[col].fillna(0)

    mask = np.isin(all_label_columns, df.columns)
    print("mask")
    print(mask)
    print(labels.argmax(axis=1))
    return texts, labels.argmax(axis=1), mask

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=150):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, padding='max_length',
                                              return_attention_mask=True, return_tensors='pt', truncation=True)

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class XLMRobertaForMultiLabelSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=16, drop=0.3):
        super(XLMRobertaForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlmroberta = XLMRobertaModel.from_pretrained(config)
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(1024, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlmroberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pred = torch.sigmoid(logits)
        return pred

    def load_labelembedd(self, label_embed):
        if label_embed is not None:
            embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
            embed.weight = torch.nn.Parameter(label_embed)
        else:
            embed = torch.nn.Embedding(self.num_labels, 1024)
        return embed

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.lstm_hid_dim).to(device),
                torch.randn(2, batch_size, self.lstm_hid_dim).to(device))

def custom_loss(outputs, labels, mask):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(outputs, labels)
    mask = torch.tensor(mask, dtype=torch.float32).to(device)
    mask = mask[labels]
    loss = loss * mask
    return loss.sum() / mask.sum()


train_texts, train_labels, train_mask = read_tsv(
    'train_data.tsv', LABEL_COLUMNS, ALL_LABEL_COLUMNS)
val_texts, val_labels, val_mask = read_tsv('val_data.tsv', LABEL_COLUMNS, ALL_LABEL_COLUMNS)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForMultiLabelSequenceClassification(MODEL_NAME, num_labels=len(ALL_LABEL_COLUMNS))
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

fgm = FGM(model)

best_f1 = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
    train_loss = []
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = custom_loss(outputs, labels, train_mask)
        train_loss.append(float(loss))
        loss.backward()


        fgm.attack()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss_adv = custom_loss(outputs, labels, train_mask)
        loss_adv.backward()
        fgm.restore()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = np.mean(train_loss)
    print("avg_loss", avg_loss)

    model.eval()
    predictions = []
    true_labels = []
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())


    print(f"True labels shape: {len(true_labels)}")
    print(f"Predictions shape: {len(predictions)}")

    val_accuracy, val_report = calculate_evaluation(np.array(predictions), np.array(true_labels), val_mask)
    val_f1 = val_report["weighted avg"]["f1-score"]
    print(
            f'Val Accuracy: {val_accuracy:.4f}, '
            f'Val Precision: {val_report["weighted avg"]["precision"]:.4f}, Val Recall: {val_report["weighted avg"]["recall"]:.4f}, Val F1-Score: {val_report["weighted avg"]["f1-score"]:.4f}'
        )

    if best_f1 < val_f1:
        model_name = f"model_name.pt"
        torch.save(model, model_name)
        best_f1 = val_f1
        print(f'The model has been saved.')
    print('best_f1:', best_f1)
