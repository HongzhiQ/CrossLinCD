# -*- coding: utf-8 -*-
import csv
import json


labels = ["all labels here"]


input_file = 'Your_data.tsv'
output_file = 'Your_data.jsonl'

with open(input_file, 'r', encoding='utf-8') as tsvfile, open(output_file, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')

    for row in reader:
        try:

            system_content = ("You are a psychologist familiar with Burns' theory of cognitive distortions. "
                              "Please assess the user's statement to determine whether the user exhibits any of the following "
                              "cognitive distortions and, if so, which specific ones are present: Emotional Reasoning, "
                              "Overgeneralization, Labeling, Mind Reading, Fortune-telling, All-or-nothing thinking, "
                              "Should statements, Magnification, Personalization, Mental filter. If multiple cognitive distortions "
                              "are present, please separate them with commas. If none of the above cognitive distortions are identified, "
                              "respond with: No cognitive distortions detected.")

            user_content = row['original text']

            distortions = [label for label in labels if row[label] == '1']
            assistant_content = ",".join(distortions) if distortions else "No cognitive distortions detected"


            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]


            json_line = {"messages": messages}
            jsonlfile.write(json.dumps(json_line, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"Error processing row: {row} - {e}")

print(f"The file is {output_file}")
