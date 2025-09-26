import datadocket as dd
import spectrapepper as spep
from tqdm import tqdm

def _average_embedding_(
    answers_file: str,
    average_embeddings_file: str,
    variations: list) -> bool:
    """
    Computes the average embedding for each question and its variations, and saves the results to a JSON file.

    This function processes a dataset of answers, where each answer contains multiple variations (e.g., original question, synonym change, etc.), 
    and each variation contains a list of embedding vectors (one per answer/iteration). For each question and each variation, it computes the 
    average embedding vector (using `spectrapepper.avg`), and stores the result in a new dictionary. The output is a list of dictionaries, 
    each containing the question ID and the average embedding for each variation. The results are saved to the specified JSON file.

    Args:
        answers_file (str): 
            Path to the JSON file containing the answers and their embeddings. The file should be a list of dictionaries, 
            each with an "id" and keys for each variation, where each variation is a list of embedding vectors.
        average_embeddings_file (str): 
            Path to the output JSON file where the average embeddings will be saved. The file will be a list of dictionaries, 
            each with an "id" and the average embedding for each variation.
        variations (list): 
            List of variation names (str) to process for each question (e.g., ["question", "synonym_change", ...]).

    Returns:
        bool: 
            Returns True if the process completes successfully and the output file is written.
    """

    answers = dd.load.Json(answers_file)

    average_embeddings = []
    for question in tqdm(answers, desc="Questions"):
        average_embedding_answer = {}
        average_embedding_answer["id"] = question["id"]
        
        for variation in variations:
            
            answers_embeddings = question[variation]
            answers_embeddings = [emb for emb in answers_embeddings if emb != []]
            average_embedding = spep.avg(answers_embeddings).tolist()
            average_embedding_answer[variation] = average_embedding
        
        average_embeddings.append(average_embedding_answer)

    dd.save.Json(average_embeddings_file, average_embeddings, mode="w")

    return True
