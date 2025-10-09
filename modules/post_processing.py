import os

import datadocket as dd
import numpy as np
import Levenshtein

def calculate_angle_degrees(vec1, vec2):
        """
        Calculate the angle between two vectors in degrees (0 to 360)

        Args:
            vec1: numpy array (question vector)
            vec2: numpy array (variation vector)

        Returns:
            angle_deg: float
        """
        
        # normalize vectors first
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        
        # Calculate angle in radians
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

def embedding_distances_stats(
    variations: list,
    data_file: str,
    save_file: str) -> bool:
    """
    Compute aggregated statistics (mean, median, std, min, max) for each metric and variation
    from a list-of-dicts input format:
    [
      {
        "id": 1,
        "question": {
            "cosine_distance": {
                "mean": float,
                "median": float,
                "std": float,
                "min": float,
                "max": float
            },
            "euclidean_distance": {...},
            "angle_degrees": {...}
        },
        "synonym_change": {
            ...
        },
        ...
      },
      ...
    ]
    """

    # Load the questions embeddings stats (list of dicts)
    questions_embeddings_distances = dd.load.Json(data_file)

    # Prepare a dictionary to hold the aggregated statistics
    aggregated_stats = {}

    # For each variation, collect all values for each metric
    metrics = ["cosine_distance", "euclidean_distance", "angle_degrees"]
    for variation in variations:
        # initialize the metric values
        metric_values = {metric: [] for metric in metrics}
        # iterate over the questions embeddings distances
        for q in questions_embeddings_distances:
            stats = q[variation]
            for i in range(len(stats)): # if "questions", only one other casaes have ITERATIONS
                for metric in metrics:
                    metric_values[metric].append(stats[i][metric])
            
        # Compute statistics for each metric
        aggregated_stats[variation] = {}
        for metric in metrics:
            values = np.array(metric_values[metric])
            if len(values) > 0:
                aggregated_stats[variation][metric] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                aggregated_stats[variation][metric] = {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None
                }

    # Save the aggregated statistics
    dd.save.Json(save_file, aggregated_stats, mode="w")
    return True


def lexical_distances_stats(
    data_file: str,
    save_file: str,
    variations: list) -> bool:
    """
    Calculate average statistics for answer sentence stats according to the structure
    created by answer_sentence_stats function.
    """

    # Load the answer stats file (should be *_lexical_stats.json)
    answer_stats = dd.load.Json(data_file)

    # Prepare to accumulate metrics for each variation
    metrics, count_metrics = {}, {}

    for entry in answer_stats:
        for variation in variations:
            if variation not in entry:
                continue
                
            if variation not in metrics:
                metrics[variation] = {
                    "question": {
                        "levenshtein_distance": [],
                        "indel_distance": [],
                        "hamming_distance": [],
                        "jaro_distance": [],
                    },
                    "variation": {
                        "levenshtein_distance": [],
                        "indel_distance": [],
                        "hamming_distance": [],
                        "jaro_distance": [],
                    }
                }
                count_metrics[variation] = {
                    "word_count": [],
                    "character_count": []
                }
            
            # Get the stats for this variation in this entry
            var_stats = entry[variation]
            
            # Process each entry in the variation list
            for var_entry in var_stats:
                # Collect counts
                if "counts" in var_entry:
                    for count_key in count_metrics[variation]:
                        if count_key in var_entry["counts"]:
                            count_metrics[variation][count_key].append(var_entry["counts"][count_key])
                
                # Collect question distances
                if "question" in var_entry:
                    for dist_key in metrics[variation]["question"]:
                        if dist_key in var_entry["question"]:
                            metrics[variation]["question"][dist_key].append(var_entry["question"][dist_key])
                
                # Collect variation distances
                if "variation" in var_entry:
                    for dist_key in metrics[variation]["variation"]:
                        if dist_key in var_entry["variation"]:
                            metrics[variation]["variation"][dist_key].append(var_entry["variation"][dist_key])

    # Compute averages
    average_metrics = {}
    for variation in variations:
        if variation not in metrics:
            continue
            
        average_metrics[variation] = {
            "question": {},
            "variation": {},
            "counts": {}
        }
        
        # Average question distances
        for dist_key, values in metrics[variation]["question"].items():
            average_metrics[variation]["question"][dist_key] = float(np.mean(values)) if values else 0.0
        
        # Average variation distances
        for dist_key, values in metrics[variation]["variation"].items():
            average_metrics[variation]["variation"][dist_key] = float(np.mean(values)) if values else 0.0
        
        # Average counts
        for count_key, values in count_metrics[variation].items():
            average_metrics[variation]["counts"][count_key] = float(np.mean(values)) if values else 0.0

    # Save the average metrics to a file
    dd.save.Json(save_file, average_metrics, mode="w")
    
    return True


def lexical_distances(
    data_file: str,
    save_file: str,
    variations: list) -> bool:
    """
    The output is a json file with the following structure:
    {
        "id": int,
        "question": [
            {
                "counts": {
                    "word_count": int,
                    "character_count": int,
                },
                "question": {
                    "levenshtein_distance": float,
                    "indel_distance": float,
                    "hamming_distance": int,
                    "jaro_distance": float
                },
                "variation": {
                    ...
                },   
            },
            {
                ...
            }
        ],
        "synonym_change": [
            {
                "counts": ...,
                "question": ...,
                "variation": ...,
            },
            ...
        ],
        ...
        }
    }
    """
    # load questions
    questions = dd.load.Json("data/questions.json")

    # load answers
    answers = dd.load.Json(data_file)

    answers_stats = []
    for answer in answers:
        answer_stats = {}
        answer_stats["id"] = answer["id"]
        # Find the corresponding question with the same id
        question_match = next((q for q in questions if q["id"] == answer["id"]), None)
        og_question = question_match["question"][0]
        
        for variation in variations:
            answer_stats[variation] = []
            for i in range(len(answer[variation])):
                counts = {
                    "word_count": len(answer[variation][i].split()),
                    "character_count": len(answer[variation][i])
                }
                answer_stats_og_question = {
                        "levenshtein_distance": Levenshtein.distance(og_question, answer[variation][i]),
                        "indel_distance": Levenshtein.ratio(og_question, answer[variation]),
                        "hamming_distance": Levenshtein.hamming(og_question, answer[variation][i]),
                        "jaro_distance": Levenshtein.jaro(og_question, answer[variation][i])
                    }
                
                question_variation = question_match[variation][0]
                answer_stats_question_variation = {
                        "levenshtein_distance": Levenshtein.distance(question_variation, answer[variation][i]),
                        "indel_distance": Levenshtein.ratio(question_variation, answer[variation][i]),
                        "hamming_distance": Levenshtein.hamming(question_variation, answer[variation][i]),
                        "jaro_distance": Levenshtein.jaro(question_variation, answer[variation][i])
                    }

                temp_variation_i = {
                    "counts": counts,
                    "question": answer_stats_og_question,
                    "variation": answer_stats_question_variation
                }
                answer_stats[variation].append(temp_variation_i)
            
        answers_stats.append(answer_stats)

    # save answers stats
    dd.save.Json(save_file, answers_stats, mode="w")

    return True


def embeddings_distances(
    data_file: str,
    save_file: str,
    variations: list) -> bool:
    """
    """
    embeddings = dd.load.Json(data_file)
    embeddings_distances = []
    for embedding in embeddings:
        embedding_distances = {}
        embedding_distances["id"] = embedding["id"]

        if len(embedding["question"]) > 1:
            question_embedding = np.mean(np.array(embedding["question"]), axis=0)
        else:
            question_embedding = np.array(embedding["question"][0])

        for variation in variations:
            embedding_distances[variation] = []

            for i in range(len(embedding[variation])):
                current_embedding = np.array(embedding[variation][i])
                euclidean_distance = np.linalg.norm(current_embedding - question_embedding)
                cosine_distance = 1 - np.dot(current_embedding, question_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(question_embedding))
                angle_degrees = calculate_angle_degrees(current_embedding, question_embedding)

                embedding_distances[variation].append({
                    "euclidean_distance": euclidean_distance,
                    "cosine_distance": cosine_distance,
                    "angle_degrees": angle_degrees
                })
        embeddings_distances.append(embedding_distances)
    dd.save.Json(save_file, embeddings_distances, mode="w")
    return True

