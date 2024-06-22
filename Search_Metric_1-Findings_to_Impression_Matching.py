#%%
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from typing import List

# Define a function to extract data from a dataframe column into a list
def extract_list_from_column_and_remove_line_break(df: pd.DataFrame, column_name: str) -> List[str]:
    """
    Extracts a list of strings from the specified column of a DataFrame, converts each string to lowercase,
    and removes any line breaks in each item.

    Args:
    df_main (pd.DataFrame): The DataFrame from which to extract the column.
    column_name (str): The name of the column from which to extract the list.

    Returns:
    List[str]: A list of cleaned strings from the specified column.
    """
    return df[column_name].dropna().apply(lambda x: x.lower().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('   ', ' ').replace('    ', ' ').replace('  ', ' ')).tolist()


# Set up the device for computation, preferring GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("Warning: No GPU found.")

#%%
# Load the datasets for Top 5 rank calculations
dataset_folder_path = 'path_to_your/folder/'
radiology_reports_dataset = 'radiology_reports'
df = pd.read_csv(dataset_folder_path + radiology_reports_dataset + '.csv', index_col=0)

# Extract the radiology report findings and impressions sections and store them as lists
findings_list = extract_list_from_column_and_remove_line_break(df, "radiology_reports_findings_only_ct_chest_with_contrast_only")
impressions_list = extract_list_from_column_and_remove_line_break(df, "radiology_reports_impression_only_ct_chest_with_contrast_only")
report_ids = extract_list_from_column_and_remove_line_break(df, "report_id_ct_chest_with_contrast_only_total")

#%%
# Load the model
your_model_save_path = 'your_output_folder_path/RadBERT-RoBERTa-4m/your_model_final'

# Alternatively, you can use a different open-source/open-weights embedding model with the Hugging Face model path
huggingface_model_path = 'Alibaba-NLP/gte-large-en-v1.5'

# Initialize the model
embedder = SentenceTransformer(your_model_save_path, trust_remote_code=True) # Or SentenceTransformer(huggingface_model_path, trust_remote_code=True) to use a model on Hugging Face 
embedder.max_seq_length = 514  # Set maximum token count for embeddings. May need to adjust for longer radiology reports
embedder = embedder.to(device) # Move the model to the device (GPU or CPU)

# Pre-encode all impressions at once
corpus_embeddings = embedder.encode(impressions_list, convert_to_tensor=True, device=device)

# Batch processing of queries
batch_size = 128  # Adjust batch size based on your GPU capacity
all_top_results = []

# Process queries in batches
for batch_start in range(0, len(findings_list), batch_size):
    batch_queries = findings_list[batch_start:batch_start + batch_size]
    query_embeddings = embedder.encode(batch_queries, convert_to_tensor=True, device=device)
    cos_scores = util.pytorch_cos_sim(query_embeddings, corpus_embeddings) # Compute cosine similarity scores

    top_k = 5 # Adjust to specify the number of ranks to retrieve
    for scores in cos_scores:
        top_results = torch.topk(scores, k=top_k)
        top_indices = top_results.indices.tolist()
        # Collect top results along with their respective report IDs
        all_top_results.append([(impressions_list[i], report_ids[i]) for i in top_indices])

# Flatten the results and create DataFrame
columns = [f'rank_{i + 1}_impression' for i in range(top_k)] + [f'rank_{i + 1}_id' for i in range(top_k)]
flat_list = [item for sublist in all_top_results for item in sublist]
impressions, ids = zip(*flat_list)
top_results_df = pd.DataFrame({
    **{f'rank_{i + 1}_impression': impressions[i::top_k] for i in range(top_k)},
    **{f'rank_{i + 1}_id': ids[i::top_k] for i in range(top_k)}
})

# Concatenate the original DataFrame with the top results DataFrame
df_final = pd.concat([df.reset_index(drop=True), top_results_df], axis=1)

# Save the final DataFrame to a CSV file
df_final.to_csv(dataset_folder_path + 'Search_Metric_1_Ranks_your_model_final' + '.csv')