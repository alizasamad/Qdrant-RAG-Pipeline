# adjustable parameters
folder =  'policy_docs'
doc_tag = 'swa'
n = 100  # number of questions to generate

#############################################
############ 1) LOAD & PREP DOCS ############
#############################################
import pickle

with open(f"{folder}/{doc_tag}.pk", 'rb') as fi:
    document = pickle.load(fi)

texts = []
for doc in document:
    texts.append(str(doc))

print(len(texts))

## Create KG
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()
for text in texts:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": text,
                "document_metadata": doc_tag,
            },
        )
    )

print(kg)

#  Set up LLM & Embedding Model
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import ChatBedrockConverse
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
import os

# load env variables
load_dotenv()
OWUI_TOKEN = os.getenv("OWUI_TOKEN")
MODEL = os.getenv("MODEL")

# set bedrock LLM
model_id = MODEL
region_name = "us-east-1"
    
bedrock_llm = ChatBedrockConverse(model_id=model_id, region_name=region_name, max_tokens= 4096)
llm = LangchainLLMWrapper(bedrock_llm)

# set embedding model to use
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding = LangchainEmbeddingsWrapper(hf_embeddings)

## Set up extractors & relation builders
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors import EmbeddingExtractor


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)

# keyphrase relations
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)
keyphrase_relations = OverlapScoreBuilder(
    property_name="keyphrases",
    new_property_name="overlap_score",
    threshold=0.2,
    distance_threshold=0.8,
)

# summary necessary to generate personas list
summary_extractor = SummaryExtractor(
    llm = llm,
    property_name = "summary"  
)
summary_embeddings = EmbeddingExtractor(
    embedding_model = embedding,
    property_name = "summary_embedding",
    embed_property_name = "summary"
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
    keyphrase_relations,
    summary_extractor,
    summary_embeddings
]

apply_transforms(kg, transforms=transforms)
print(kg)
print("Current node properties:", kg.nodes[0].properties.keys())
if kg.relationships:
    print("Relationship types:", [r.type for r in kg.relationships])
    print("last relationship properties:", kg.relationships[-1].properties)

############################################
############ 2) SET UP PERSONAS ############
############################################
from ragas.testset.persona import generate_personas_from_kg
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

personas = generate_personas_from_kg(kg=kg, llm=llm, num_personas=5)
print(personas)

query_distibution = [
    (
        MultiHopSpecificQuerySynthesizer(
            llm=llm, 
            property_name="keyphrases", 
            relation_type = "keyphrases_overlap", 
            relation_overlap_property = "overlapped_items"
        ),
        1.0,
    ),
]

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=llm,
    embedding_model=embedding,
    knowledge_graph=kg,
    persona_list=personas,
)

testset = generator.generate(testset_size=n, query_distribution=query_distibution)
testset.to_csv(f"datasets/test-set-ragas-synth-{n}-{doc_tag}.csv")
