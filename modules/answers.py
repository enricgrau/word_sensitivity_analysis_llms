import gc
import os

from tqdm import tqdm
import datadocket as dd

from modules.llm_utils import llm
from modules.embedding_utils import embedding

def get_answers(
    iterations: int, 
    llm_model: str, 
    system_prompt: str, 
    variations: list) -> bool:
    """
    _get_answers_ is a utility function that generates LLM responses and their embeddings for a set of questions and their lexical variations, saving the results to disk.

    Args:
        iterations (int): Number of times to query the LLM for each question variation (to account for LLM stochasticity).
        llm_model (str): The identifier or name of the LLM model to use for generating answers.
        system_prompt (str): The system prompt or instruction to provide to the LLM for all responses.
        variations (list): List of string keys corresponding to the different question variations to process (e.g., ["question", "synonym_change", ...]).

    Returns:
        bool: True if the process completes successfully and files are saved.
    """
    # files to use
    questions_file = "data/questions.json"
    answers_file = f"results/{llm_model.replace(':', '_')}/answers.json"

    # load questions from data/questions.json except for * which contains the definitions
    questions = dd.load.Json(questions_file)

    # load answers from data/answers.json if it exists
    if os.path.exists(answers_file):
        answers = dd.load.Json(answers_file)
        ids_ready = [a["id"] for a in answers]
        answers = []
    else:
        ids_ready = []

    # iterate over questions
    for question in tqdm(questions, desc=f"Generating answers with {llm_model}"):
        # start empty dicts
        answers_text = {}

        # get id
        id = question["id"]

        # skip if id already exists
        if id in ids_ready:
            continue

        # add id to dicts
        answers_text["id"] = id

        for variation in variations:
            # start empty lists
            answers_text[variation] = []

            for _ in range(iterations):
                # get response
                response = llm(prompt=question[variation][0], model=llm_model, system_prompt=system_prompt)
                answers_text[variation].append(response)
            
        # Save all data at once at the end
        dd.save.Json(answers_file, answers_text, mode="a")
        
        if id % 10 == 0:
            gc.collect()

    return True


def get_embeddings(
    data_file: str,
    save_file: str,
    variations: list) -> bool:
    """
    _get_answers_ is a utility function that generates LLM responses and their embeddings for a set of questions and their lexical variations, saving the results to disk.

    Args:
        iterations (int): Number of times to query the LLM for each question variation (to account for LLM stochasticity).
        llm_model (str): The identifier or name of the LLM model to use for generating answers.
        system_prompt (str): The system prompt or instruction to provide to the LLM for all responses.
        variations (list): List of string keys corresponding to the different question variations to process (e.g., ["question", "synonym_change", ...]).

    Returns:
        bool: True if the process completes successfully and files are saved.
    """
    # load questions from data/questions.json except for * which contains the definitions
    questions = dd.load.Json(data_file)

    # load answers from data/answers.json if it exists
    if os.path.exists(save_file):
        answers = dd.load.Json(save_file)
        ids_ready = [a["id"] for a in answers]
    else:
        ids_ready = []

    # iterate over questions
    for question in tqdm(questions, desc=f"Generating embeddings..."):
 
        # start empty dicts
        embeddings = {}

        # get id
        id = question["id"]

        # skip if id already exists
        if id in ids_ready:
            continue

        # add id to dicts
        embeddings["id"] = id

        for variation in variations:
            # start empty list to store the embeddings
            embeddings[variation] = []
            # iterate over the iterations
            for iteration in question[variation]:
                response_embedding = embedding(prompt=iteration)
                embeddings[variation].append(response_embedding)

        # Save all data at once at the end
        dd.save.Json(save_file, embeddings, mode="a")
        
        if id % 10 == 0:
            gc.collect()

    return True
