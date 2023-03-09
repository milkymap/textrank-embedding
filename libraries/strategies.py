import re 

import spacy 
import numpy as np 
import networkx as nx
import unicodedata

import operator as op 

from math import ceil 
from spacy.language import Language
from sentence_transformers import SentenceTransformer

from typing import List, Tuple  

def load_language_model(model_name:str) -> Language:
    model = spacy.load(name=model_name)
    return model 

def load_sentence_transformer(model_name:str, cache_folder:str='cache', device:str='cpu') -> SentenceTransformer:
    sentence_model = SentenceTransformer(model_name, cache_folder=cache_folder, device=device)
    return sentence_model

def to_sentences(text:str, spacy_model:Language, valid_length:str) -> List[str]:
    document = spacy_model(text)
    sentences_iter = map(
        lambda sent: unicodedata.normalize("NFKD", sent.text), 
        document.sents
    )
    sentences = [ re.sub(r"  +", '', sent).replace('\n', '') for sent in sentences_iter ]
    sentences = [ sent for sent in sentences if len(sent.split(' ')) > valid_length]
    return sentences

def compute_fingerprint(document:List[str], vectorizer:SentenceTransformer, device:str='cpu') -> np.ndarray:
    embedding = vectorizer.encode(
        sentences=document,
        device=device,
        convert_to_numpy=True
    )
    return embedding

def compute_pairwise_matrix(embeddings:np.ndarray) -> np.ndarray:
    dot_scores = embeddings @ embeddings.T 
    fst_norms = np.linalg.norm(embeddings, axis=1)
    norms = fst_norms[:, None] * fst_norms[None, :] 
    similarity_matrix =  dot_scores / (norms + 1e-8)
    return similarity_matrix 

def semantic_textrank(adjacency_matrix:np.ndarray) -> List[Tuple[int, float]]:
    nb_nodes = len(adjacency_matrix)
    initial_scores = np.sum(adjacency_matrix, axis=1) / nb_nodes

    graph:nx.Graph = nx.from_numpy_array(adjacency_matrix)
    for node in graph.nodes():
        graph.nodes[node]['weight'] = initial_scores[node]
    
    node_scores = nx.pagerank(graph, alpha=0.9)
    return list(node_scores.keys()), list(node_scores.values())

