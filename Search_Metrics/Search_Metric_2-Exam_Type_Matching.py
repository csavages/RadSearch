#%%
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np

# Determine if CUDA (GPU) is available for PyTorch, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

# Load the dataset of radiology reports
folder_path = 'path_to_your/folder/'
radiology_reports_dataset = 'radiology_reports'
df = pd.read_csv(f"{folder_path}/{radiology_reports_dataset}.csv")

# Define the exam type, findings sections of reports of that exam type to be used as a query, the entire corpus of searchable impressions (i.e., all exam types), and the impressions of only the 
# exams of the specified exam type
exam_type = 'CT_chest'
query_column_name = f'radiology_report_findings_only_{exam_type}'
corpus_column_name = 'radiology_report_impression_only'
specified_exam_type_only_impressions_list_name = f"{corpus_column_name}_{exam_type}"
specified_exam_type_only_impressions_list = df[specified_exam_type_only_impressions_list_name].dropna().tolist()

#%%
def run_semantic_search(df, embedder_path, query_column, corpus_column, exam_type):
    embedder = SentenceTransformer(embedder_path, trust_remote_code=True)
    embedder.max_seq_length = 514 # Set maximum token count for embeddings. May need to adjust for longer radiology reports
    queries = df[query_column].dropna().tolist()
    query_embedding = embedder.encode(queries, convert_to_tensor=True, device=device)

    corpus = df[corpus_column].dropna().tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, device=device)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(specified_exam_type_only_impressions_list))

    results = []
    for query_id, query_hits in enumerate(hits):
        result_dict = {
            f"report_id_{exam_type}": df.iloc[query_id][f"report_id_{exam_type}"],
            f"query_{exam_type}": queries[query_id]
        }
        for rank, hit in enumerate(query_hits):
            result_dict[f"rank_{rank + 1}"] = corpus[hit['corpus_id']]
        results.append(result_dict)

    exam_type_semantic_search_results = pd.DataFrame(results)

    output_path = f'{folder_path}/exam_type_search_results_{embedder_path.split("/")[-1]}'
    exam_type_semantic_search_results.to_csv(output_path + '_' + exam_type + '.csv', index=False)


# List of embedding models to perform semantic search with
embedder_path_list = ['your_output_folder_path/RadBERT-RoBERTa-4m/your_model_final', 'sentence-transformers/all-mpnet-base-v2', 'sentence-transformers/msmarco-distilbert-base-v4',
                      'Alibaba-NLP/gte-large-en-v1.5']

"""
Iterate embedding models through the semantic search function to generate the top ranks corresponding to 
the number of exams of that type in the dataset (e.g., Top 6417 ranks in a dataset of 13958 reports where 
6417 are CT chest reports).
NOTE: This will likely take a long time to complete
"""
for embedder in embedder_path_list:
    run_semantic_search(df, embedder, query_column_name, corpus_column_name, exam_type)

#%%
# Function to calculate average precision
def calculate_average_precision(ranked_list, specified_exam_type_only_impressions):
    relevant_flags = np.isin(ranked_list, specified_exam_type_only_impressions)
    k_array = np.arange(1, len(ranked_list) + 1)
    precision_at_k = np.cumsum(relevant_flags) / k_array
    sum_precision = np.sum(precision_at_k * relevant_flags)
    return sum_precision / len(specified_exam_type_only_impressions) if len(specified_exam_type_only_impressions) > 0 else 0


# Create a list of the resulting spreadsheets from the semantic search function
exam_type_semantic_search_rank_results = [
    'exam_type_search_results_' + 'your_model_final' + '_' + exam_type,
    'exam_type_search_results_' + 'all-mpnet-base-v2' + '_' + exam_type,
    'exam_type_search_results_' + 'msmarco-distilbert-base-v4' + '_' + exam_type,
    'exam_type_search_results_' + 'gte-large-en-v1.5' + '_' + exam_type
]

# Columns to consider for AP calculation
selected_columns = [f'rank_{i}' for i in range(1, len(specified_exam_type_only_impressions_list) + 1)]

# Initialize an empty list to hold average precision (AP) values
AP_list = []

# Calculate average precision and add to DataFrame
for exam_type_semantic_search_rank_result in exam_type_semantic_search_rank_results:
    df_to_rank = pd.read_csv(f"{folder_path}/{exam_type_semantic_search_rank_result}.csv")
    for index, row in df_to_rank.iterrows():
        AP_value = calculate_average_precision(row[selected_columns].tolist(), specified_exam_type_only_impressions_list)
        AP_list.append(AP_value)

# Calculate the mean average precision (mAP)
mAP = np.mean(AP_list)

# Print mAP result
print(f"Mean Average Precision: {mAP}")
