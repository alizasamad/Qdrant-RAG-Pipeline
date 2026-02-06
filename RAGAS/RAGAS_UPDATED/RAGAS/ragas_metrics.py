import asyncio
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from ragas import Dataset, experiment, SingleTurnSample
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
from ragas.metrics import RougeScore, NonLLMContextRecall
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.embeddings.base import embedding_factory, LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper
import time
import logging
import numpy as np

####################################################
############### 0) ADJUSTABLE PARAMS ###############
####################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
log = logging.getLogger(__name__)

load_dotenv()
OWUI_TOKEN = os.getenv("OWUI_TOKEN")
MODEL = os.getenv("MODEL")
COLLECTION_ID = "1b91887e-8afd-47b8-b5c9-2441224baf25" #os.getenv("TEST_COLLECTION")
# 1b91887e-8afd-47b8-b5c9-2441224baf25 -- google_NQ
# 638d4dcf-8df1-402b-a908-3f26d38db917 -- t5
# f64f49dc-d53e-408c-a813-fca6bff764e5 -- swa

dataset = 'google_NQ_simplified.csv' # target dataset
#'test-set-naive-synth-82-t5.csv'
#'test-set-naive-synth-100-swa.csv'
#'google_NQ_simplified.csv'

# for naming results files
n = 50  # sample size
reps = 1  # num times to run experiment on same sample
model = 'combo_reranker' # RAG model being tested
policy = 'google_NQ'  # doc tag

# for ragas experiment
embed = 'sentence-transformers'  # options: 'cohere' | '', 'sentence-transformers' (default)
owui_prompt = "Answer questions concisely based on the documents provided. Keep responses succinct."  # prompt sent to OWUI

####################################################
############# 1) CALL OPEN WEBUI RAG API ###########
####################################################

def chat_with_collection(query):
    system_prompt = owui_prompt
    url = "http://localhost:3000/api/chat/completions"
    headers = {"Authorization": f"Bearer {OWUI_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": query},
                     {"role": "system", "content": system_prompt} ],
        "files": [{"type": "collection", "id": COLLECTION_ID}],
        "max_tokens": 4000
    }

    response = requests.post(url, headers=headers, json=payload, timeout=600)
    return response.json()

########################################################
############# 2) SETTING UP EVALUATORS ###########
########################################################

# set bedrock LLM evaluator
model_id = MODEL
region_name = "us-east-1"
bedrock_llm = ChatBedrock(model_id=model_id, region_name=region_name, model_kwargs = {"max_tokens": 4096})
evaluator_llm = LangchainLLMWrapper(bedrock_llm)

# cohere embeddings
from langchain_aws import BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    region_name=region_name,
)

# huggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if embed == 'cohere':
    embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)
else:
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Set LLM and embeddings globally on metrics
context_precision.llm = evaluator_llm
context_recall.llm = evaluator_llm
answer_relevancy.llm = evaluator_llm
answer_relevancy.embeddings = embeddings
faithfulness.llm = evaluator_llm

######################################################
############# 3) RAGAS EXPERIMENT FUNCTION ###########
######################################################

@experiment()
async def owui_rag_experiment(row):
    log.info(f"Processing next row")
    rag_start = time.time()
    chat = chat_with_collection(row["question"])
    log.info(f"Response Generated: {chat}")
    rag_end=time.time()
    rag_latency = rag_end - rag_start

    retrieved = chat['sources'][0]['document']
    retrieved_string = ",".join(retrieved)
    response = chat["choices"][0]["message"]["content"]
    input_tokens = chat['usage']['prompt_tokens']
    output_tokens = chat['usage']['completion_tokens']
    total_tokens = chat['usage']['total_tokens']
    log.info(f"{input_tokens}, {output_tokens}")
    log.info(f"Response: {response}")

    if policy == 'google_NQ':
        reference = row["short_answers"]
        user_input = row["question"]
        reference_contexts = [row["reference_texts"]]
    else:
        reference = row["answer"]
        user_input = row["question"]
        reference_contexts = [row["reference_text"]]

    # Initialize scores as None
    precision = None
    recall = None
    relevancy = None
    faith = None

    # Define metric sample
    sample = SingleTurnSample(
        user_input=user_input,
        retrieved_contexts=[retrieved_string],
        response=response,
        reference = reference,
        reference_contexts = reference_contexts
    )

    try:
        # retrieval
        log.info("Awaiting retrieval metrics...") 
        precision = await context_precision.single_turn_ascore(sample)
        recall = await context_recall.single_turn_ascore(sample)
        nonLLMrecall = await NonLLMContextRecall().single_turn_ascore(sample)
        log.info('''Finished retrieval metrics.
                 Awaiting generation metrics...''')

        # generation
        scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")
        result = await scorer.single_turn_ascore(sample)
        relevancy = await answer_relevancy.single_turn_ascore(sample)  # LLM-based metrics take very long to run
        faith = await faithfulness.single_turn_ascore(sample)
        log.info("Finshed generation metrics")
        
        log.info("New row being added!")
        return {
            **row,
            "response": response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            "rag_latency": rag_latency,
            "context_precision_score": precision,
            "context_recall_score": recall,
            "non_LLM_context_recall_score": nonLLMrecall,
            "rouge_score": result,
            "answer_relevancy_score": relevancy,
            "answer_faithfulness_score": faith
        }
    except Exception as e:
        log.error(f"Error processing row: {e}")
        # Return row with NaN values for metrics
        return {
            **row,
            "response": "ERROR",
            'input_tokens': np.nan,
            'output_tokens': np.nan,
            'total_tokens': np.nan,
            "rag_latency": np.nan,
            "context_precision_score": np.nan,
            "context_recall_score": np.nan,
            "non_LLM_context_recall_score": np.nan,
            "rouge_score": np.nan,
            "answer_relevancy_score": np.nan,
            "answer_faithfulness_score": np.nan
        }

###############################################
############# 4) RUN THE EXPERIMENT ###########
###############################################

#Google NQ
#82 t5
#100 swa

# load data
df = pd.read_csv(f"datasets/{dataset}")

import random
random.seed(123)
indices = random.sample(range(len(df)), n)
df_sample = df.iloc[indices]

questions = df_sample['question']
print(questions)

dataset = Dataset.from_pandas(df_sample, name="swa_dataset", backend="local/csv", root_dir=".")

file_name = f"{policy}_{model}_{n}_{reps}"  # output filename

# Initialize array
all_metrics = []
columns = ["rag_latency","context_precision_score","context_recall_score", 
                    "non_LLM_context_recall_score", "rouge_score", "answer_relevancy_score", 
                    "answer_faithfulness_score"]

# Run experiment `reps` times
for i in range(0, reps):
    print("Running iteration: " + str(i+1))
    result = asyncio.run(owui_rag_experiment.arun(dataset, name=file_name))
    data = pd.read_csv(f"experiments/{file_name}.csv")
    
    # Store this metrics each run
    all_metrics.append(data[columns].to_numpy())

all_metrics_array = np.array(all_metrics)

# Calculate mean 
mean_metrics = np.mean(all_metrics_array, axis=0)
mean_df = pd.DataFrame(mean_metrics, columns=columns, index = ["MEAN_" + str(i) for i in range(0, len(all_metrics_array[0]))])

# Calculate std
std_metrics = np.std(all_metrics_array, axis=0)
std_df = pd.DataFrame(std_metrics, columns=columns, index = ["STD_" + str(i) for i in range(0, len(all_metrics_array[0]))])

# Add summary stats to csv file
combined_df = pd.concat([
    data,
    pd.DataFrame([["---"] * len(columns)], columns=columns, index = ["---"]),
    mean_df,
    std_df
])

print(mean_df)
print(std_df)

combined_df['dataset_label'] = [policy for _ in range(len(combined_df))]
combined_df['model'] = [model for _ in range(len(combined_df))]
combined_df.to_csv(f"experiments/{file_name}.csv", index=True)
print("csv successfully updated!")