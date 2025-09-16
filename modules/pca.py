import datadocket as dd
from sklearn.decomposition import PCA
import numpy as np
import pickle

def _pca_(
    average_embeddings_file: str,
    pca_model_file: str,
    variations: list) -> bool:
    """
    Performs Principal Component Analysis (PCA) on average embedding vectors and saves the trained PCA model.

    This function loads a JSON file containing average embeddings for each question and its variations. 
    For each question, it extracts the embedding vectors for all specified variations, aggregates them, 
    and fits a PCA model (with 2 components) to reduce the dimensionality of the embeddings. 
    The trained PCA model is then saved to a file using pickle.

    Args:
        average_embeddings_file (str): 
            Path to the JSON file containing average embeddings. The file should be a list of dictionaries, 
            each with an "id" and keys for each variation, where each value is an embedding vector (list of floats).
        pca_model_file (str): 
            Path to the output file where the trained PCA model will be saved (using pickle).
        variations (list): 
            List of variation names (str) to extract embeddings from each question (e.g., 
            ["question", "synonym_change", "antonym_change", "paraphrase_change", "letter_change"]).

    Returns:
        bool: 
            Returns True if the PCA model is successfully trained and saved.
    """

    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    # collect all embeddings for PCA fitting
    all_embeddings = []
    for question_data in average_embeddings:
        embeddings = [question_data[variation] for variation in variations]
        embeddings = np.array(embeddings)
        all_embeddings.extend(embeddings)

    # fit PCA on all data
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)

    # print the explained variance ratio
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    # save the model with pickle
    with open(pca_model_file, "wb") as f:
        pickle.dump(pca, f)
    
    return True
