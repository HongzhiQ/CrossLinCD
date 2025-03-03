import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from sklearn import metrics
import torch.nn as nn
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Parameter definitions
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 300
LABEL_COLUMNS = ["all labels here"]
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 200
THRESHOLD = 0.25
SEED = 55  # Set random seed

# Set random seed
set_seed(SEED)



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_evaluation(prediction, true_label, mask, type='micro'):
    mask_indices = np.where(mask)[0]
    true_label = true_label[:, mask_indices]
    prediction = prediction[:, mask_indices]
    true_label = true_label.astype(int)
    prediction = prediction.astype(int)
    recall = metrics.recall_score(true_label, prediction, average=type, zero_division=0)
    precision = metrics.precision_score(true_label, prediction, average=type, zero_division=0)
    f1 = metrics.f1_score(true_label, prediction, average=type, zero_division=0)
    return recall, precision, f1

def read_tsv_with_soft_labels(file_path, label_columns, soft_labels_file):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['Original text'].tolist()
    labels = np.zeros((len(df), len(label_columns)))
    actual_label_columns = [col for col in label_columns if col in df.columns]
    for col in actual_label_columns:
        labels[:, label_columns.index(col)] = df[col].fillna(0)
    mask = np.isin(label_columns, df.columns)

    # Read soft labels
    soft_df = pd.read_csv(soft_labels_file, delimiter='\t')
    soft_labels = soft_df[label_columns].values.astype(float)
    return texts, labels, soft_labels, mask

class TextDatasetWithSoftLabels(Dataset):
    def __init__(self, texts, labels, soft_labels, mask, tokenizer, max_len=150):
        self.texts = texts
        self.labels = labels
        self.soft_labels = soft_labels
        self.mask = mask
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        soft_label = self.soft_labels[item]

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label),
            'soft_labels': torch.FloatTensor(soft_label),
            'mask': torch.FloatTensor(self.mask)
        }

class XLMRobertaForMultiTaskLearning(nn.Module):
    def __init__(self, config, num_labels_dict, drop=0.3):
        super(XLMRobertaForMultiTaskLearning, self).__init__()
        self.num_labels_dict = num_labels_dict
        self.xlmroberta = XLMRobertaModel.from_pretrained(config)
        self.dropout = nn.Dropout(drop)
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(1024, num_labels) for task, num_labels in num_labels_dict.items()
        })

    def forward(self, input_ids, attention_mask=None, task=None):
        outputs = self.xlmroberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[task](pooled_output)
        pred = torch.sigmoid(logits)
        return pred

def mixed_loss_fn(student_logits, hard_labels, soft_labels, mask, alpha=0.8):
    mask = mask.to(student_logits.device)
    hard_loss = nn.BCELoss(reduction='none')(student_logits, hard_labels)
    hard_loss = hard_loss * mask
    hard_loss = hard_loss.sum() / mask.sum()

    soft_loss = nn.KLDivLoss(reduction='none')(
        torch.log(student_logits + 1e-8), soft_labels
    )
    soft_loss = soft_loss * mask
    soft_loss = soft_loss.sum() / mask.sum()

    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss

# Data loading with soft labels
# Replace the paths with your actual file paths
train_texts_chn1, train_labels_chn1, train_soft_labels_chn1, train_mask_chn1 = read_tsv_with_soft_labels(
    'train1.tsv',
    LABEL_COLUMNS,
    'train1_withSoftLabel.tsv'  # Replace with your soft labels file
)
val_texts_chn1, val_labels_chn1, val_soft_labels_chn1, val_mask_chn1 = read_tsv_with_soft_labels(
    'val2.tsv',
    LABEL_COLUMNS,
    'val2_withSoftLabel.tsv'  # Replace with your soft labels file
)

train_texts_eng1, train_labels_eng1, train_soft_labels_eng1, train_mask_eng1 = read_tsv_with_soft_labels(
    'train3.tsv',
    LABEL_COLUMNS,
    'train3_withSoftLabel.tsv'  # Replace with your soft labels file
)
val_texts_eng1, val_labels_eng1, val_soft_labels_eng1, val_mask_eng1 = read_tsv_with_soft_labels(
    'val3.tsv',
    LABEL_COLUMNS,
    'val3_withSoftLabel.tsv'  # Replace with your soft labels file
)

train_texts_eng2, train_labels_eng2, train_soft_labels_eng2, train_mask_eng2 = read_tsv_with_soft_labels(
    'train4.tsv',
    LABEL_COLUMNS,
    'train4_withSoftLabel.tsv'  # Replace with your soft labels file
)
val_texts_eng2, val_labels_eng2, val_soft_labels_eng2, val_mask_eng2 = read_tsv_with_soft_labels(
    'val4.tsv',
    LABEL_COLUMNS,
    'val4_withSoftLabel.tsv'  # Replace with your soft labels file
)

train_texts_eng3, train_labels_eng3, train_soft_labels_eng3, train_mask_eng3 = read_tsv_with_soft_labels(
    'train5.tsv',
    LABEL_COLUMNS,
    'train5_withSoftLabel.tsv'  # Replace with your soft labels file
)
val_texts_eng3, val_labels_eng3, val_soft_labels_eng3, val_mask_eng3 = read_tsv_with_soft_labels(
    'val5.tsv',
    LABEL_COLUMNS,
    'val5_withSoftLabel.tsv'  # Replace with your soft labels file
)

# Initialize tokenizer and data loaders
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset_chn1 = TextDatasetWithSoftLabels(train_texts_chn1, train_labels_chn1, train_soft_labels_chn1, train_mask_chn1, tokenizer, max_len=MAX_LEN)
val_dataset_chn1 = TextDatasetWithSoftLabels(val_texts_chn1, val_labels_chn1, val_soft_labels_chn1, val_mask_chn1, tokenizer, max_len=MAX_LEN)
train_loader_chn1 = DataLoader(train_dataset_chn1, batch_size=BATCH_SIZE, shuffle=True)
val_loader_chn1 = DataLoader(val_dataset_chn1, batch_size=BATCH_SIZE)

train_dataset_eng1 = TextDatasetWithSoftLabels(train_texts_eng1, train_labels_eng1, train_soft_labels_eng1, train_mask_eng1, tokenizer, max_len=MAX_LEN)
val_dataset_eng1 = TextDatasetWithSoftLabels(val_texts_eng1, val_labels_eng1, val_soft_labels_eng1, val_mask_eng1, tokenizer, max_len=MAX_LEN)
train_loader_eng1 = DataLoader(train_dataset_eng1, batch_size=BATCH_SIZE, shuffle=True)
val_loader_eng1 = DataLoader(val_dataset_eng1, batch_size=BATCH_SIZE)

train_dataset_eng2 = TextDatasetWithSoftLabels(train_texts_eng2, train_labels_eng2, train_soft_labels_eng2, train_mask_eng2, tokenizer, max_len=MAX_LEN)
val_dataset_eng2 = TextDatasetWithSoftLabels(val_texts_eng2, val_labels_eng2, val_soft_labels_eng2, val_mask_eng2, tokenizer, max_len=MAX_LEN)
train_loader_eng2 = DataLoader(train_dataset_eng2, batch_size=BATCH_SIZE, shuffle=True)
val_loader_eng2 = DataLoader(val_dataset_eng2, batch_size=BATCH_SIZE)

train_dataset_eng3 = TextDatasetWithSoftLabels(train_texts_eng3, train_labels_eng3, train_soft_labels_eng3, train_mask_eng3, tokenizer, max_len=MAX_LEN)
val_dataset_eng3 = TextDatasetWithSoftLabels(val_texts_eng3, val_labels_eng3, val_soft_labels_eng3, val_mask_eng3, tokenizer, max_len=MAX_LEN)
train_loader_eng3 = DataLoader(train_dataset_eng3, batch_size=BATCH_SIZE, shuffle=True)
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

# Initialize best F1 scores for each task
best_f1_dict = {
    'chn1': 0,
    'eng1': 0,
    'eng2': 0,
    'eng3': 0
}

# Training loop with soft labels
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
            soft_labels = batch['soft_labels'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, task=task)
            loss = mixed_loss_fn(outputs, labels, soft_labels, mask)
            train_loss.append(float(loss))
            loss.backward()

            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = np.mean(train_loss)
        print(f"{task} avg_loss", avg_loss)

    # Validation
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
                logits_np = output.cpu().numpy()
                predictions = (logits_np >= THRESHOLD).astype(int)
                predict = np.concatenate((predict, predictions), axis=0)
                gt = np.concatenate((gt, labels.cpu().numpy()), axis=0)

        recall, precision, f1 = calculate_evaluation(predict, gt, val_mask, type='micro')
        print(f'{task} epoch:', epoch, '  F1:', f1, '  recall:', recall, '  precision:', precision)

        # Save the best model for each task
        if best_f1_dict[task] < f1:
            model_name = f"model_path.pt"
            torch.save(model.state_dict(), model_name)
            best_f1_dict[task] = f1
            print(f'Model saved {model_name} (Task: {task})')

    print(f'Current best F1 scores: {best_f1_dict}')
