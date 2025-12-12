####################################################
################ 1) ADD QUESTION ID ################
####################################################
import pandas as pd
import uuid
import numpy as np

# add all experiments to be cleaned
experiments_list = ['t5_combo_reranker_50_1.csv']

for filename in experiments_list:
    # load data
    data_unclean = pd.read_csv(f"experiments/{filename}", index_col = False)
    data = data_unclean[data_unclean['rouge_score'].notna()]

    # add UUIDs to identify questions & corresponding means/stds
    index = int((len(data['question']) - 1) / 3)
    data['question_ID'] = None
    for i in range(0, index):
        data.loc[i, 'question_ID']= str(uuid.uuid5(uuid.NAMESPACE_DNS,  str(data.iloc[i, 1])))
        data.loc[i + 1 + index, 'question_ID'] = str(uuid.uuid5(uuid.NAMESPACE_DNS,  str(data.iloc[i, 1])))
        data.loc[i + 1 + 2 * index, 'question_ID'] = str(uuid.uuid5(uuid.NAMESPACE_DNS,  str(data.iloc[i, 1])))

    # save to csv
    columns = data.columns.to_list()
    columns[0] = ""
    data.columns = columns
    data.to_csv(f"experiments/{filename}", index = 0)

# Find rows with NaN values for any of the metrics
print(len(data))
nan_questions = []
columns = [
    'context_precision_score', 'context_recall_score', 'non_LLM_context_recall_score',
    'rouge_score', 'answer_relevancy_score', 'answer_faithfulness_score', 'rag_latency', 'question_ID'
]
metric_columns = [
    'context_precision_score', 'context_recall_score', 'non_LLM_context_recall_score',
    'rouge_score', 'answer_relevancy_score', 'answer_faithfulness_score', 'rag_latency'
]
all_col = set(data.columns.to_list())
drop = all_col - set(columns)
data = data.drop(columns = drop)
row1 = pd.DataFrame({
    index: data.index,
    'rag_latency': np.nan, 
    'context_precision_score': np.nan, 
    'context_recall_score': np.nan, 
    'non_LLM_context_recall_score': np.nan, 
    'rouge_score': np.nan, 
    'answer_relevancy_score': np.nan, 
    'answer_faithfulness_score': np.nan,
    'question_ID': 'rand_string_1'
})

row2 = pd.DataFrame({
    index: data.index,
    'rag_latency': np.nan, 
    'context_precision_score': np.nan, 
    'context_recall_score': np.nan, 
    'non_LLM_context_recall_score': np.nan, 
    'rouge_score': np.nan, 
    'answer_relevancy_score': np.nan, 
    'answer_faithfulness_score': np.nan,
    'question_ID': 'rand_string_2'
})
data = pd.concat([data,row1,row2])

for metric in metric_columns:
    # Find rows where either pre or post has NaN for this metric
    nan_rows = data[data[metric].isna()]
    
    if not nan_rows.empty:
        for _, row in nan_rows.iterrows():
            q_id = row['question_ID']
            if q_id not in nan_questions:
                nan_questions.append(q_id)

# Get unique question IDs with NaNs
nan_questions = list(set(nan_questions))
print(f"All question IDs with NaN values: {nan_questions}")

print(len(data))

data_filtered = data[~data['question_ID'].isin(nan_questions)]
print(len(data_filtered))