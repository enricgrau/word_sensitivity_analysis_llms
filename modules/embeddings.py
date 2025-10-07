import os
import gc

import datadocket as dd
# import spectrapepper as spep
from tqdm import tqdm

from modules.embedding_utils import embedding

def questions_embeddings(
    variations: list) -> bool:
    """
    """
    embeddings_file = f"results/questions/questions_embeddings.json"

    if os.path.exists(embeddings_file):
        print(f"Questions embeddings already exist in {embeddings_file}. Skipping...")
        return True

    questions_file = "data/questions.json"
    questions = dd.load.Json(questions_file)

    for question in tqdm(questions, desc="Embedding Original Questions and Variations"):
        id = question["id"]
        question_embedding = {}
        question_embedding["id"] = id
        for variation in variations:
            question_embedding[variation] = embedding(question[variation])
        
        dd.save.Json(embeddings_file, question_embedding, mode="a")

        if id % 10 == 0:
            gc.collect()

    return True
