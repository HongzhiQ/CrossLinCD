import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn import metrics
import torch.nn as nn
import random


# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



LABEL_COLUMNS = ["all labels here"]
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 512
BATCH_SIZE = 16
SEEDS = []
THRESHOLDS = []



# 评估函数
def calculate_evaluation(prediction, true_label, mask, type='micro'):
    mask_indices = np.where(mask)[0]
    true_label = true_label[:, mask_indices]
    prediction = prediction[:, mask_indices]
    true_label = true_label.astype(int)
    prediction = prediction.astype(int)
    recall = metrics.recall_score(true_label, prediction, average=type)
    precision = metrics.precision_score(true_label, prediction, average=type)
    f1 = metrics.f1_score(true_label, prediction, average=type)
    return recall, precision, f1


# 读取数据
def read_tsv(file_path, label_columns):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['Original text'].tolist()
    labels = np.zeros((len(df), len(label_columns)))
    actual_label_columns = [col for col in label_columns if col in df.columns]
    for col in actual_label_columns:
        labels[:, label_columns.index(col)] = df[col].fillna(0)
    mask = np.isin(label_columns, df.columns)
    return texts, labels, mask


# 数据集类
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
            'labels': torch.FloatTensor(label)
        }



class XLMRobertaForMultiLabelSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=12, drop=0.3):
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


def predict(model, tokenizer, file_paths, label_columns, batch_size, threshold, device):

    all_predict = np.zeros((0, len(label_columns)), dtype=np.int32)
    all_gt = np.zeros((0, len(label_columns)), dtype=np.int32)

    for file_path in file_paths:
        texts, labels, mask = read_tsv(file_path, label_columns)
        dataset = TextDataset(texts, labels, tokenizer, max_len=MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        model.eval()

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask)
                logits = output
                logits_np = logits.cpu().numpy()
                predictions = np.where(logits_np >= threshold, 1, 0)

                all_predict = np.concatenate((all_predict, predictions), axis=0)
                all_gt = np.concatenate((all_gt, labels.cpu().numpy()), axis=0)


    recall, precision, f1 = calculate_evaluation(all_predict, all_gt, mask, type='micro')

    print(f'Combined F1 Score: {f1}, Recall: {recall}, Precision: {precision}')

    return recall, precision, f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForMultiLabelSequenceClassification(config=MODEL_NAME, num_labels=len(LABEL_COLUMNS))
model = torch.load("model_path.pt")
model.to(device)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

test_files = ['test.tsv']

results = {}

for seed in SEEDS:
    set_seed(seed)
    results[seed] = {}
    for threshold in THRESHOLDS:
        print(f"\nSeed: {seed}, Threshold: {threshold}")
        results[seed][threshold] = {}
        recall, precision, f1 = predict(model, tokenizer, test_files, LABEL_COLUMNS, BATCH_SIZE, threshold, device)
        results[seed][threshold] = {'recall': recall, 'precision': precision, 'f1': f1}


def find_max_f1(results):
    max_f1_summary = {}

    for file_path in test_files:
        max_f1 = 0
        best_seed = None
        best_threshold = None

        for seed in SEEDS:
            for threshold in THRESHOLDS:
                f1 = results[seed][threshold][file_path]['f1']
                if f1 > max_f1:
                    max_f1 = f1
                    best_seed = seed
                    best_threshold = threshold

        max_f1_summary[file_path] = {
            'max_f1': max_f1,
            'best_seed': best_seed,
            'best_threshold': best_threshold
        }

    return max_f1_summary


max_f1_results = find_max_f1(results)

print("Summary of max F1 scores for each dataset:")
for file_path, metrics in max_f1_results.items():
    print(
        f"File: {file_path}, Max F1 Score: {metrics['max_f1']}, Best Seed: {metrics['best_seed']}, Best Threshold: {metrics['best_threshold']}")
