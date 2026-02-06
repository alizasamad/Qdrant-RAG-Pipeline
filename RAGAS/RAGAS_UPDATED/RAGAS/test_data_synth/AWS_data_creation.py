# Import necessary libraries for downloading files
from urllib.request import urlretrieve 
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
import json
import os
import sys
import boto3
import prompt_templates as pt
import pandas as pd
import time
from langchain_core.documents.base import Document
from langchain_core.documents.base import Document
from tqdm import tqdm
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
log = logging.getLogger(__name__)

BEDROCK = boto3.client('bedrock-runtime', region_name='us-east-1') 

def download_pdfs(folder_name, files):
    # Check if folder already exists, if yes do nothing
    # If no, create the folder
    if os.path.exists(folder_name):
        pass  
    else:
        os.mkdir(folder_name)

    # Iterate through list of URLs 
    for url in files:
        # Get file name from URL to use as local file name
        file_path = os.path.join(folder_name, url.rpartition("/")[2]) 
        urlretrieve(url, file_path)

    print(f"Downloaded files to {folder_name}")


def chunk_doc(folder_name, filename):
    # Load PDF document from folder
    filepath = f"./{folder_name}/{filename}"
    loader = PyPDFLoader(filepath)
    #loader = PyPDFDirectoryLoader(f"./{folder_name}/")
    documents = loader.load()
    
    # Use recursive character splitter, works better for this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(

        # Split documents into small chunks
        chunk_size = 1500,  

        # Overlap chunks to reduce cutting sentences in half
        chunk_overlap  = 100,
        separators=["\n\n", "\n", ".", " ", ""],

    )


    # Split loaded documents into chunks
    docs = text_splitter.split_documents(documents)

    # Print metadata of the loaded documents
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(f'Average length among {len(documents)} pages loaded is {avg_char_count_pre} characters.')
    print(f'After the split you have {len(docs)}')
    print(f'Average length among {len(docs)} chunks is {avg_char_count_post} characters.')

    return docs


def llm(prompt):
    payload = {
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
        "anthropic_version": "bedrock-2023-05-31"
    }
    
    try:
        # Call Bedrock
        response = BEDROCK.invoke_model(
            modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # or your preferred model
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        return response_body
        
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        return None

def generate_question(doc):
    # Pass in values to the input variables
    initial_question_prompt = pt.INITIAL_QUESTION_PROMPT_TEMPLATE.format(context=doc)
    #print(initial_question_prompt)
    question = llm(initial_question_prompt)

    return question['content'][0]['text']

def generate_answer(question: str, doc):
    answer_prompt = pt.ANSWER_PROMPT_TEMPLATE.format(question = question, context=doc)
    answer = llm(answer_prompt)
    
    return answer['content'][0]['text']

def generate_source(question: str, doc):
    source_prompt = pt.SOURCE_PROMPT_TEMPLATE.format(question = question, context=doc)
    source_sentence = llm(source_prompt)
    
    return source_sentence['content'][0]['text']

def compress_question(question): 
    # Pass in values to the input variables
    question_compress_prompt = pt.QUESTION_COMPRESS_PROMPT_TEMPLATE.format(question=question)
    question_compressed = llm(question_compress_prompt)
        
    return question_compressed['content'][0]['text']

def generate_qa_dataset_doc(doc: Document, dataset, doc_number):
    """A function to create a test dataset of questions for a given Document(Langchain Document type)"""
    
    # generate the initial question for the RAG testdataset
    question = generate_question(doc)
    dataset.at[doc_number, "question"] = question
    
    # generate compressed question to variate the dataset
    compressed_question = compress_question(question)
    dataset.at[doc_number, "question_compressed"] = compressed_question
   
    
    answer = generate_answer(question, doc)
    dataset.at[doc_number, "reference_answer"] = answer
        
    source_sentence = generate_source(question, doc)
    dataset.at[doc_number, "source_sentence"] = source_sentence
    
    source_raw = doc
    dataset.at[doc_number, "source_raw"] = source_raw.page_content
    
    source_document = doc.metadata["source"]
    dataset.at[doc_number, "source_document"] = source_document
    
    return dataset

def generate_dataset(documents: Document, dataset):

    print(f"start generating dataset from {len(documents)} docuements")
    print("---")
    generation_time_start = time.time()
    
    for doc in tqdm(range(len(documents))):
        q_generation_time_start = time.time()
        dataset = generate_qa_dataset_doc(doc = documents[doc], dataset = dataset, doc_number = doc)
        q_generation_time_end = time.time()
        total_elapsed_time_generation = q_generation_time_end - q_generation_time_start


        print(f"Finished creating evaluation data for chunk {doc+1}")
        print(f"Generation time for doc: {total_elapsed_time_generation}")
        print("---")
        
    generation_time_end = time.time()
    total_elapsed_time= generation_time_end - generation_time_start
    print(f"Generation time for all docs: {total_elapsed_time}")
        
    return dataset
    
def generate_groundedness_check(question, source_raw): 
    # Pass in values to the input variables
    groundedness_prompt = pt.GROUNDEDNESS_CHECK_PROMPT_TEMPLATE.format(question=question, context=source_raw)
    
    groundedness_rating = llm(groundedness_prompt)
        
    return groundedness_rating['content'][0]['text']

def generate_relevance_check(question): 
    # Pass in values to the input variables
    relevance_prompt = pt.RELEVANCE_CHECK_SWA.format(question=question)
    
    relevance_rating = llm(relevance_prompt)
        
    return relevance_rating['content'][0]['text']

def extract_rating(text):
    pattern = r'<rating>(.*?)</rating>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        rating = match.group(1)
        return rating
    else:
        return text
    
def extract_reasoning(text):
    pattern = r'<evaluation>(.*?)</evaluation>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        rating = match.group(1)
        return rating
    else:
        return text

def evaluate_dataset(dataset, method:str):
    i = 100
    for index, row in dataset.iterrows():
        i = i-1
        log.info(f"{i} more iterations to go!")
        log.info(f"Evaluating index: {index}!")

        if method == 'naive':
            q = 'question'
            source = 'reference_text'
        elif method == 'aws':
            q = 'question'
            source = 'source_raw'
        elif method == 'ragas':
            q = 'user_input'
            source = 'reference_contexts'
        else:
            return log.error("Invalid method.")

        question = row[q]  # naive: 'question' | aws: 'question' | ragas: 'user_input'
        source_raw = row[source]  # naive: 'reference_text' | aws: 'source_raw' | ragas: 'reference_contexts'

        # Generate groundedness check
        log.info("Attempting groudedness check...")
        groundedness_check = generate_groundedness_check(question, source_raw)
        groundedness_score = extract_rating(groundedness_check)
        groundedness_score_reasoning = extract_reasoning(groundedness_check)
        log.info("Completed Groudedness check.")

        dataset.at[index, 'groundedness_score'] = groundedness_score
        dataset.at[index, 'groundedness_score_reasoning'] = groundedness_score_reasoning

        #Generate relevance 
        log.info("Attempting relevance check...")
        relevance_check = generate_relevance_check(question)
        relevancy_score = extract_rating(relevance_check)
        relevancy_score_reasoning = extract_reasoning(relevance_check)
        log.info("Completed relevance check.")

        dataset.at[index, 'relevancy_score'] = relevancy_score
        dataset.at[index, 'relevancy_score_reasoning'] = relevancy_score_reasoning

    log.info("Returning dataset!")
    return dataset

    
