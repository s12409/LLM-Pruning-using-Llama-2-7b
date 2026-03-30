import pandas as pd
import json
import string



def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')
    return text

def transform_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    formatted_data = []
    for index, row in df.iterrows():
        input = clean_text(row['article'])
        output = clean_text(row['highlights'])
        formatted_data.append({
            # "subject": row['id'],  
            # "input": input, 
            # "output": output  
            "input": input,  
            "output": output  
        })


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    transform_dataset('./hw2_datasets/train.csv', "./hw2_datasets/train.json")
    transform_dataset('./hw2_datasets/test.csv', "./hw2_datasets/test.json")
    transform_dataset('./hw2_datasets/validation.csv', "./hw2_datasets/validation.json")

