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
import time

load_dotenv()

OWUI_TOKEN = os.getenv("OWUI_TOKEN")
MODEL = os.getenv("MODEL")
COLLECTION_ID = os.getenv("TEST_COLLECTION")

####################################################
############# 1) CALL OPEN WEBUI RAG API ###########
####################################################
def chat_with_collection(query):
    system_prompt = "Answer questions concisely based on the documents provided. Respond with no more than 5 words."
    url = "http://localhost:3000/api/chat/completions"
    headers = {"Authorization": f"Bearer {OWUI_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": query},
                     {"role": "system", "content": system_prompt} ],
        "files": [{"type": "collection", "id": COLLECTION_ID}],
    }
    response = requests.post(url, headers=headers, json=payload).json()
    return response

########################################################
############# 2) LOCAL EVALUATOR LLM (LLAMA) ###########
########################################################
# evaluator_llm = llm_factory (
#     "llama3.2:latest", 
#     "litellm", 
#     client="http://localhost:11434")

# deprecated version, llm_factory does not work well with local models
ollama_llm = ChatOllama(  
    model="qwen2.5:14b",
    base_url="http://localhost:11434",
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(ollama_llm)

# set embeddings for answer_relevancy to use
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
    rag_start = time.time()
    chat = chat_with_collection(row["question"])
    rag_end=time.time()
    rag_latency = rag_end - rag_start

    retrieved = chat['sources'][0]['document']
    retrieved_string = ",".join(retrieved)
    response = chat["choices"][0]["message"]["content"]
    reference = row["short_answers"]

    # Initialize scores as None
    precision = None
    recall = None
    relevancy = None
    faith = None

    # Extract Reference texts
    txt_contents = []
    for filename in os.listdir("nq_txt_files"):
        if filename.endswith(".txt"):
            file_path = os.path.join("nq_txt_files", filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    txt_contents.append(content)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # Define metric sample
    sample = SingleTurnSample(
        user_input=row["question"],
        retrieved_contexts=[retrieved_string],
        response=response,
        reference = reference,
        reference_contexts= txt_contents
    )

    try:
        # retrieval 
        precision = await context_precision.single_turn_ascore(sample)
        recall = await context_recall.single_turn_ascore(sample)
        nonLLMrecall = await NonLLMContextRecall().single_turn_ascore(sample)

        # generation
        scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")
        result = await scorer.single_turn_ascore(sample)
        relevancy = await answer_relevancy.single_turn_ascore(sample)  # LLM-based metrics take very long to run
        faith = await faithfulness.single_turn_ascore(sample)
        
        return {
            **row,
            "response": response,
            "retrieved_contexts": [retrieved_string],
            "rag_latency": rag_latency,
            "context_precision_score": precision,
            "context_recall_score": recall,
            "non_LLM_context_recall_score": nonLLMrecall,
            "rouge_score": result,
            "answer_relevancy_score": relevancy,
            "answer_faithfulness_score": faith
        }
    except Exception as e:
        print(f"Error processing row: {e}")


###############################################
############# 4) RUN THE EXPERIMENT ###########
###############################################
df = pd.read_csv("datasets/test_set.csv")
dataset = Dataset.from_pandas(df, name="google_NQ_simplified", backend="local/csv", root_dir=".")
import asyncio
result = asyncio.run(owui_rag_experiment.arun(dataset, name="owui_rag_with_system_prompt_test"))

# view results
print("success!")
