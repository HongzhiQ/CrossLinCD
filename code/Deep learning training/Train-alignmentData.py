import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import warnings
import random

warnings.filterwarnings("ignore")


# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 参数定义
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 300
LABEL_COLUMNS = ["all labels here"]
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 150
THRESHOLD = 0.25
SEED = 44

# 设置随机数种子
set_seed(SEED)


def calculate_evaluation(prediction, true_label, type='micro'):
    true_label = true_label.astype(int)
    prediction = prediction.astype(int)
    recall = metrics.recall_score(true_label, prediction, average=type)
    precision = metrics.precision_score(true_label, prediction, average=type)
    f1 = metrics.f1_score(true_label, prediction, average=type)
    return recall, precision, f1


def read_tsv(file_path, label_columns):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['Original text'].tolist()
    labels = np.zeros((len(df), len(label_columns)))
    for col in label_columns:
        if col in df.columns:
            labels[:, label_columns.index(col)] = df[col].fillna(0)
    return texts, labels


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
    def __init__(self, config, num_labels=15, drop=0.3):
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


def custom_loss(outputs, labels):
    loss_fct = nn.BCELoss()
    return loss_fct(outputs, labels)



train_texts, train_labels = read_tsv('train.tsv', LABEL_COLUMNS)
val_texts, val_labels = read_tsv('val.tsv', LABEL_COLUMNS)


tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForMultiLabelSequenceClassification(MODEL_NAME)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
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
        loss = custom_loss(outputs, labels)
        train_loss.append(float(loss))
        loss.backward()

        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = np.mean(train_loss)
    print("avg_loss", avg_loss)

    model.eval()
    predict = np.zeros((0, len(LABEL_COLUMNS)), dtype=np.int32)
    gt = np.zeros((0, len(LABEL_COLUMNS)), dtype=np.int32)
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output
            logits_np = logits.cpu().numpy()
            predictions = np.where(logits_np >= THRESHOLD, 1, 0)
            predict = np.concatenate((predict, predictions), axis=0)
            gt = np.concatenate((gt, labels.cpu().numpy()), axis=0)

    recall, precision, f1 = calculate_evaluation(predict, gt, type='micro')
    print('epoch:', epoch, '  F1:', f1, '  recall:', recall, '  precision:', precision)
    if best_f1 < f1:
        model_name = f"model_path.pt"
        torch.save(model, model_name)
        best_f1 = f1
        print(f'The model has been saved.')
    print('best_f1:', best_f1)
