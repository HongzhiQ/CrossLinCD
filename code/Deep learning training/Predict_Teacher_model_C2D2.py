import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import random


MODEL_PATH = "model_path.pt"
BATCH_SIZE = 16
LABEL_COLUMNS = ["情绪化推理", "以偏概全", "乱贴标签", "读心术", "先知错误", "非此即彼", "应该句式", "放大", "罪责归己",
                 "心理过滤", "否定正面思考", "罪责归他","无标签"]
ALL_LABEL_COLUMNS = ["all labels here"]
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 150

torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_tsv(file_path, label_columns, all_label_columns):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['原始文本'].tolist()
    labels = np.zeros((len(df), len(all_label_columns)))
    actual_label_columns = [col for col in label_columns if col in df.columns]
    for col in actual_label_columns:
        labels[:, all_label_columns.index(col)] = df[col].fillna(0)
    mask = np.isin(all_label_columns, df.columns)
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


test_texts, test_labels, test_mask = read_tsv(
    'C2D2_test_data.tsv',
    LABEL_COLUMNS, ALL_LABEL_COLUMNS)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


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

    report = classification_report(true_label_filtered, prediction_filtered, output_dict=True, zero_division=1,
                                   labels=mask_indices, target_names=[ALL_LABEL_COLUMNS[i] for i in mask_indices])
    accuracy = accuracy_score(true_label_filtered, prediction_filtered)
    return accuracy, report


def get_label_name(pred):
    return ALL_LABEL_COLUMNS[pred]


seeds = []
best_f1 = 0
best_seed = None

for seed in seeds:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(MODEL_PATH)
    model.to(device)
    model.eval()

    predictions = []
    true_labels = []
    texts = []

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.cpu().numpy()
            preds = np.argmax(logits, axis=1)
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            texts.extend(batch['text'])


    print(f"True labels shape (Seed {seed}): {len(true_labels)}")
    print(f"Predictions shape (Seed {seed}): {len(predictions)}")


    test_accuracy, test_report = calculate_evaluation(np.array(predictions), np.array(true_labels), test_mask)
    print(
        f'Seed {seed} - Test Accuracy: {test_accuracy:.4f}, '
        f'Test Precision: {test_report["weighted avg"]["precision"]:.4f}, '
        f'Test Recall: {test_report["weighted avg"]["recall"]:.4f}, '
        f'Test F1-Score: {test_report["weighted avg"]["f1-score"]:.4f}'
    )

    if test_report["weighted avg"]["f1-score"] > best_f1:
        best_f1 = test_report["weighted avg"]["f1-score"]
        best_seed = seed


print(f'\nbest_seed: {best_seed}, best F1-Score: {best_f1:.4f}')
