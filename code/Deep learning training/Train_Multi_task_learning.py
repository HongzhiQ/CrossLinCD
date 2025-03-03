import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import warnings
import random
from tqdm import tqdm

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
SEED = 22  # 设置随机种子

# 设置随机数种子
set_seed(SEED)
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def read_tsv(file_path, label_columns):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['Original text'].tolist()
    labels = np.zeros((len(df), len(label_columns)))
    actual_label_columns = [col for col in label_columns if col in df.columns]
    for col in actual_label_columns:
        labels[:, label_columns.index(col)] = df[col].fillna(0)
    mask = np.isin(label_columns, df.columns)
    return texts, labels, mask

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

class XLMRobertaForMultiTaskLearning(nn.Module):
    def __init__(self, config, num_labels_dict, drop=0.5):
        super(XLMRobertaForMultiTaskLearning, self).__init__()
        self.num_labels_dict = num_labels_dict
        self.xlmroberta = XLMRobertaModel.from_pretrained(config)
        self.dropout = nn.Dropout(drop)
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(1024, num_labels) for task, num_labels in num_labels_dict.items()
        })

    def forward(self, input_ids, attention_mask=None, task=None):
        outputs = self.xlmroberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[task](pooled_output)
        pred = torch.sigmoid(logits)
        return pred

def custom_loss(outputs, labels, mask):
    mask = torch.FloatTensor(mask).to(device)
    loss_fct = nn.BCELoss(reduction='none')
    loss = loss_fct(outputs, labels)
    loss = loss * mask
    return loss.sum() / mask.sum()


train_texts_chn1, train_labels_chn1, train_mask_chn1 = read_tsv(
    'trainData1.tsv', LABEL_COLUMNS)
val_texts_chn1, val_labels_chn1, val_mask_chn1 = read_tsv(
    'testData1.tsv', LABEL_COLUMNS)

train_texts_eng1, train_labels_eng1, train_mask_eng1 = read_tsv(
    'trainData2.tsv', LABEL_COLUMNS)
val_texts_eng1, val_labels_eng1, val_mask_eng1 = read_tsv(
    'testData2.tsv', LABEL_COLUMNS)

train_texts_eng2, train_labels_eng2, train_mask_eng2 = read_tsv(
    'trainData3.tsv', LABEL_COLUMNS)
val_texts_eng2, val_labels_eng2, val_mask_eng2 = read_tsv(
    'testData3.tsv', LABEL_COLUMNS)

train_texts_eng3, train_labels_eng3, train_mask_eng3 = read_tsv(
    'trainData4.tsv', LABEL_COLUMNS)
val_texts_eng3, val_labels_eng3, val_mask_eng3 = read_tsv(
    'testData4.tsv', LABEL_COLUMNS)


tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset_chn1 = TextDataset(train_texts_chn1, train_labels_chn1, tokenizer, max_len=MAX_LEN)
val_dataset_chn1 = TextDataset(val_texts_chn1, val_labels_chn1, tokenizer, max_len=MAX_LEN)
train_loader_chn1 = DataLoader(train_dataset_chn1, batch_size=BATCH_SIZE)
val_loader_chn1 = DataLoader(val_dataset_chn1, batch_size=BATCH_SIZE)

train_dataset_eng1 = TextDataset(train_texts_eng1, train_labels_eng1, tokenizer, max_len=MAX_LEN)
val_dataset_eng1 = TextDataset(val_texts_eng1, val_labels_eng1, tokenizer, max_len=MAX_LEN)
train_loader_eng1 = DataLoader(train_dataset_eng1, batch_size=BATCH_SIZE)
val_loader_eng1 = DataLoader(val_dataset_eng1, batch_size=BATCH_SIZE)

train_dataset_eng2 = TextDataset(train_texts_eng2, train_labels_eng2, tokenizer, max_len=MAX_LEN)
val_dataset_eng2 = TextDataset(val_texts_eng2, val_labels_eng2, tokenizer, max_len=MAX_LEN)
train_loader_eng2 = DataLoader(train_dataset_eng2, batch_size=BATCH_SIZE)
val_loader_eng2 = DataLoader(val_dataset_eng2, batch_size=BATCH_SIZE)

train_dataset_eng3 = TextDataset(train_texts_eng3, train_labels_eng3, tokenizer, max_len=MAX_LEN)
val_dataset_eng3 = TextDataset(val_texts_eng3, val_labels_eng3, tokenizer, max_len=MAX_LEN)
train_loader_eng3 = DataLoader(train_dataset_eng3, batch_size=BATCH_SIZE)
val_loader_eng3 = DataLoader(val_dataset_eng3, batch_size=BATCH_SIZE)


num_labels_dict = {
    'chn1': len(LABEL_COLUMNS),
    'eng1': len(LABEL_COLUMNS),
    'eng2': len(LABEL_COLUMNS),
    'eng3': len(LABEL_COLUMNS)
}
model = XLMRobertaForMultiTaskLearning(MODEL_NAME, num_labels_dict)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

#Initialize the best F1 score for each task.
best_f1_dict = {
    'chn1': 0,
    'eng1': 0,
    'eng2': 0,
    'eng3': 0
}


model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    for train_loader, task, train_mask in [
        (train_loader_chn1, 'chn1', train_mask_chn1), 
        (train_loader_eng1, 'eng1', train_mask_eng1), 
        (train_loader_eng2, 'eng2', train_mask_eng2), 
        (train_loader_eng3, 'eng3', train_mask_eng3)]:
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} - {task}', leave=False)
        train_loss = []
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, task=task)
            loss = custom_loss(outputs, labels, train_mask)
            train_loss.append(float(loss))
            loss.backward()

            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = np.mean(train_loss)
        print(f"{task} avg_loss", avg_loss)

    # 验证
    model.eval()
    for val_loader, task, val_mask in [
        (val_loader_chn1, 'chn1', val_mask_chn1), 
        (val_loader_eng1, 'eng1', val_mask_eng1), 
        (val_loader_eng2, 'eng2', val_mask_eng2), 
        (val_loader_eng3, 'eng3', val_mask_eng3)]:
        
        predict = np.zeros((0, len(LABEL_COLUMNS)), dtype=np.int32)
        gt = np.zeros((0, len(LABEL_COLUMNS)), dtype=np.int32)
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask, task=task)
                logits = output
                logits_np = logits.cpu().numpy()
                predictions = np.where(logits_np >= THRESHOLD, 1, 0)
                predict = np.concatenate((predict, predictions), axis=0)
                gt = np.concatenate((gt, labels.cpu().numpy()), axis=0)

        recall, precision, f1 = calculate_evaluation(predict, gt, val_mask, type='micro')
        print(f'{task} epoch:', epoch, '  F1:', f1, '  recall:', recall, '  precision:', precision)
        
        # 检查并保存每个任务的最佳模型
        if best_f1_dict[task] < f1:
            model_name = f"model_name"
            torch.save(model.state_dict(), model_name)
            best_f1_dict[task] = f1
            print(f'The model has been saved.')

    print(f'best F1: {best_f1_dict}')
