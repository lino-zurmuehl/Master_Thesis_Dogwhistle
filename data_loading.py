from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from cleanlab import Datalab
import numpy as np
import random

def run_data_loading():
    # Check for MPS device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available on this device")
    else:
        device = torch.device("cpu")
        print("MPS is not available, using CPU")

    # Load the dataset
    dataset = load_dataset("SALT-NLP/silent_signals")
    dataset = dataset["train"]
    df = pd.DataFrame(dataset)

    # Function to drop duplicates and save dropped instances
    def drop_duplicates_save(df):
        duplicates = df[df.duplicated(subset=["content"], keep=False)]
        df = df.drop_duplicates(subset=["content"])
        df = df.dropna(subset=["content"])
        if "lable" in df.columns:
            df = df.dropna(subset=["lable"])
        return df, duplicates

    # Filter informal and formal sentences
    df_informal = df[df["type"] == "Informal"]
    df_formal = df[df["type"] == "Formal"]
    df_informal, duplicates_if_1 = drop_duplicates_save(df_informal)
    df_formal, duplicates_f_1 = drop_duplicates_save(df_formal)

    # Load null datasets
    null_dataset_formal = pd.read_csv("0_data/formal_neg_predictions.csv")
    null_dataset_informal = pd.read_csv("0_data/informal_neg_predictions.csv")

    # Drop duplicates from null datasets
    null_dataset_informal, duplicates_if_0 = drop_duplicates_save(null_dataset_informal)
    null_dataset_formal, duplicates_f_0 = drop_duplicates_save(null_dataset_formal)
    null_dataset = pd.concat([null_dataset_formal, null_dataset_informal])

    # Join datasets and handle labels
    df_informal = pd.concat([df_informal, null_dataset_informal])
    df_informal['lable'] = df_informal['lable'].fillna(1)
    df_formal = pd.concat([df_formal, null_dataset_formal])
    df_formal['lable'] = df_formal['lable'].fillna(1)

    # Prepare data for cleanlab
    raw_texts = df_informal['content'].tolist()
    labels = df_informal['lable'].tolist()
    dog_whistles = df_informal['dog_whistle'].tolist()

    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-cased')
    model = BertModel.from_pretrained('google-bert/bert-base-cased').to(device)

    def get_embeddings_in_batches(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    # Get embeddings and prepare cross-validation
    embeddings_np_informal = get_embeddings_in_batches(raw_texts, batch_size=32)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Train classifier using cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    pred_probs = cross_val_predict(
        clf, embeddings_np_informal, labels, method="predict_proba", cv=cv
    )

    # Use Cleanlab to find issues
    data_dict = {"texts": raw_texts, "labels": labels}
    lab = Datalab(data_dict, label_name="labels")
    lab.find_issues(pred_probs=pred_probs, features=embeddings_np_informal)
    lab.report()

    # Get label issues
    label_issues = lab.get_issues("label")
    identified_label_issues = label_issues[label_issues["is_label_issue"] == True]

    # Create dataframe with suggested labels
    data_with_suggested_labels = pd.DataFrame(
        {"dog_whistles": dog_whistles, 
        "text": raw_texts, 
        "given_label": labels, 
        "suggested_label": label_issues["predicted_label"], 
        "problem": identified_label_issues["is_label_issue"], 
        "label_score": identified_label_issues["label_score"]}
    )
    potential_wrong_label = data_with_suggested_labels.dropna(subset=["problem"]).sort_values(by="label_score", ascending=False)

    # Save to CSV
    potential_wrong_label.to_csv("vetting_instances/potential_wrong_label_new.csv")
    print("Data loaded successfully. CL pipe for potential wrong labels run and saved to CSV.")

if __name__ == "__main__":
    run_data_loading()