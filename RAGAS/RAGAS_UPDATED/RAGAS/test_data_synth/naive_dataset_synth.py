# adjustable parameters
n = 5  # number of chunks group as a single context block per LLM query
m = 5  # num questions to generate per context block
folder =  'policy_docs'
doc_tag = 'swa'  # policy file

##########################################################
################## 1) QUERY BEDROCK API ##################
##########################################################

import boto3
import json

def generate_qa_from_document(document_text):
    # Initialize Bedrock client
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')  # adjust region
    
    # Construct prompt with document text
    prompt = f"""Create a minimum of {m} questions & answers based on the provided policy.

Policy Document:
{document_text}

Format each Q&A pair as:
Q: [question]
A: [answer]

Question difficulty should range from simple lookups to edge cases (complex scenarios, exceptions, cross-references) involving multiple policy sections."""

    # Prepare payload (example for Claude)
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
        response = bedrock.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
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

######################################################
################## 3) EXPORT TO CSV ##################
######################################################
import csv 
import pickle

with open(f"{folder}/{doc_tag}.pk", 'rb') as fi:
    document = pickle.load(fi)

texts = []
for doc in document:
    texts.append(str(doc))

print(len(texts))

# generate qa pairs from 3 randomized sections within title_5_sections 
total_qa_pairs = [['question', 'answer', 'reference_text']]

import random
random.seed(123)
random.shuffle(texts)  # shuffle list

i = 1
while len(texts) >=n:
    print("iteration " + str(i))
    qa_pairs = generate_qa_from_document(texts[:n])
    qa_content = qa_pairs['content'][0]['text']
    qa_split = qa_content.split("\n\nQ")  # split qa pairs from other qa pairs

    for qa in qa_split:
        print(f"q/a pair: {qa_split.index(qa)+1}")
        qa_split_2 = qa.split("\nA")  # split q & a from each other
        if len(qa_split_2) == 2:
            qa_split_2.append(texts[:n])
            total_qa_pairs.append(qa_split_2)
    
    del texts[:n]  # delete used chunks to prevent repetition
    print(len(total_qa_pairs))
    print(len(texts))
    i=i+1

# write to csv
with open(f'datasets/test-set-naive-synth-{len(texts)/ n * m}-{doc_tag}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(total_qa_pairs)



"""
Goal of this script: take title 5 policy documentation & send to LLM to create RAG dataset package

Package: Query, Expected Answer, Reference Context --> export as CSV

Implementation:.
    1. Query Bedrock API. Once chunk is created, send to LLM to generate response. Prompt: create a maximum of 2 questions & answers based on the provided policy. 
    2. Export to CSV: write row to csv with question, answer, and chunk sent to LLM.
"""