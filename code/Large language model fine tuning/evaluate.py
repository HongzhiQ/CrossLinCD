import pandas as pd
import json
import pandas as pd
from sklearn.metrics import f1_score

data = pd.read_csv('Your_data.tsv', sep='\t')


labels = ["all labels here"]

predict_df = pd.DataFrame(0, index=range(len(data)), columns=labels)


predict_df.insert(0, "Original text", data["Original text"])


with open(
        r'generated_predictions.jsonl', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        prediction = json.loads(line)
        predicted_labels = prediction["predict"].replace("\n", "").split('，')
        predicted_labels = [label.strip() for label in predicted_labels]
        for label in predicted_labels:
            if label in predict_df.columns:
                predict_df.at[i, label] = 1

predict_tsvFiles = 'generated_predictions.tsv'


predict_df.to_csv(predict_tsvFiles, sep='\t', index=False)
print("predict.tsv")




original_data = pd.read_csv('generated_predictions.tsv', sep='\t')

predicted_data = pd.read_csv(predict_tsvFiles, sep='\t')

labels = ["all labels here"]


y_true = original_data[labels].values
y_pred = predicted_data[labels].values

micro_f1 = f1_score(y_true, y_pred, average='micro')
print("Micro F1 score:", micro_f1)
