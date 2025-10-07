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
        "question": [
          { ...distances for variation 1... },
          { ...distances for variation 2... },
          ...
        ]
      },
      ...
    ]
    """
    if os.path.exists(save_file):
        print(f"Embedding distances stats already exist in {save_file}. Skipping...")
        return True

    # Load the questions embeddings stats (list of dicts)
    questions_embeddings_stats = dd.load.Json(data_file)

    # Prepare a dictionary to hold the aggregated statistics
    aggregated_stats = {}

    # For each variation, collect all values for each metric
    metrics = ["cosine_distance", "euclidean_distance", "angle_degrees"]
    for v_idx, variation in enumerate(variations):
        metric_values = {metric: [] for metric in metrics}
        for q in questions_embeddings_stats:
            # Each q["question"] is a list of dicts, one per variation, in the same order as variations
            if "question" in q and len(q["question"]) > v_idx:
                stats = q["question"][v_idx]
                for metric in metrics:
                    if metric in stats:
                        metric_values[metric].append(stats[metric])
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

def questions_embeddings_distances(
    variations: list) -> bool:
    """
    """
    # save file name
    save_file_name = "results/questions/questions_embeddings_distances.json"

    # check if file exists
    if os.path.exists(save_file_name):
        print(f"Questions embeddings distances already exist in {save_file_name}. Skipping...")
        return True

    questions_embeddings = dd.load.Json(f"results/questions/questions_embeddings.json")
    results = []
    for question in questions_embeddings:
        id = question["id"]
        base_embedding = np.array(question["question"])  # "question" is the base embedding

        variation_distances = []
        for variation in variations:
            embedding_vec = np.array(question[variation])
            # Cosine distance
            cosine_distance = 1 - np.dot(base_embedding, embedding_vec) / (np.linalg.norm(base_embedding) * np.linalg.norm(embedding_vec))
            # Euclidean distance
            euclidean_distance = np.linalg.norm(base_embedding - embedding_vec)
            # Angle (degrees)
            angle = calculate_angle_degrees(base_embedding, embedding_vec)
            variation_distances.append({
                "euclidean_distance": float(euclidean_distance),
                "cosine_distance": float(cosine_distance),
                "angle_degrees": float(angle),
            })
        results.append({
            "id": id,
            "question": variation_distances
        })
    dd.save.Json(save_file_name, results, mode="w")
    return True

def average_question_sentence_stats(
    variations: list) -> bool:
    """
    """

    # save file name
    save_file_name = "results/questions/questions_lexical_distances_stats.json"

    # check if file exists
    if os.path.exists(save_file_name):
        print(f"Questions lexical distances stats already exist in {save_file_name}. Skipping...")
        return True

    # load question sentence stats
    question_sentence_stats = dd.load.Json("results/questions/questions_lexical_distances.json")

    # average question sentence stats
    average_question_sentence_stats = {}

    # For each variation, collect all metrics across questions
    metrics = [
        "levenshtein_distance",
        "indel_distance",
        "hamming_distance",
        "jaro_distance",
        "jaro_winkler_distance"
    ]
    count_metrics = [
        "word_count",
        "character_count"
    ]

    for variation in variations:
        # Gather all values for each metric for this variation
        metric_values = {m: [] for m in metrics}
        count_values = {c: [] for c in count_metrics}
        for q in question_sentence_stats:
            # Defensive: skip if variation not present
            if variation not in q:
                continue
            d = q[variation].get("distances", q[variation])  # support both structures
            c = q[variation].get("counts", q[variation])
            for m in metrics:
                if m in d:
                    metric_values[m].append(d[m])
            for c_metric in count_metrics:
                if c_metric in c:
                    count_values[c_metric].append(c[c_metric])
        # Compute averages
        average_question_sentence_stats[variation] = {
            "question": {m: float(np.mean(metric_values[m])) if metric_values[m] else None for m in metrics},
            "counts": {c: float(np.mean(count_values[c])) if count_values[c] else None for c in count_metrics}
        }
    
    dd.save.Json(save_file_name, average_question_sentence_stats, mode="w")
    
    return True

def question_sentence_distances(
    variations: list) -> bool:
    """
    The output is a json file with the following structure:
    {
        "id": int,
        "question": {
            "word_count": int,
            "character_count": int,
            "levenshtein_distance": float,
            "indel_distance": float,
            "hamming_distance": int,
            "jaro_distance": float
        },
        "synonym_change": {
            ...
        },
        ...
    }
    """

    # save file name
    save_file_name = "results/questions/questions_lexical_distances.json"

    # check if file exists
    if os.path.exists(save_file_name):
        print(f"Questions lexical distances already exist in {save_file_name}. Skipping...")
        return True

    # load questions
    questions = dd.load.Json("data/questions.json")

    # word stats per question
    questions_stats = []
    for question in questions:
        question_stats = {}
        question_stats["id"] = question["id"]
        og_question = question["question"]
        for variation in variations:
            distances = {
                "levenshtein_distance": Levenshtein.distance(og_question, question[variation]),
                "indel_distance": Levenshtein.ratio(og_question, question[variation]),
                "hamming_distance": Levenshtein.hamming(og_question, question[variation]),
                "jaro_distance": Levenshtein.jaro(og_question, question[variation]),
            }
            counts = {
                "word_count": len(question[variation].split()),
                "character_count": len(question[variation])
            }
            question_stats[variation] = {
                "distances": distances,
                "counts": counts
            }
        questions_stats.append(question_stats)

    # save questions stats
    dd.save.Json(save_file_name, questions_stats, mode="w")

    return True


def answer_sentence_stats(
    llm_model: str,
    variations: list) -> bool:
    """
    Calculate average statistics for answer sentence stats according to the structure
    created by answer_sentence_stats function.
    """

    # Load the answer stats file (should be *_lexical_stats.json)
    answer_stats = dd.load.Json(f"results/{llm_model.replace(':', '_')}/answers_lexical_distances.json")

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
                        "jaro_winkler_distance": []
                    },
                    "variation": {
                        "levenshtein_distance": [],
                        "indel_distance": [],
                        "hamming_distance": [],
                        "jaro_distance": [],
                        "jaro_winkler_distance": []
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
    avg_file = f"results/{llm_model.replace(':', '_')}/average_answers_lexical_distances.json"
    dd.save.Json(avg_file, average_metrics, mode="w")
    
    return True

def answer_sentence_distances(
    llm_model: str,
    variations: list) -> bool:
    """
    The output is a json file with the following structure:
    {
        "id": int,
        "question": 
            [
                [
                    "question": {
                        "word_count": int,
                        "character_count": int,
                        "levenshtein_distance": float,
                        "indel_distance": float,
                        "hamming_distance": int,
                        "jaro_distance": float,
                        "jaro_winkler_distance": float
                    },
                    "variation": {
                        ...
                    }
                ],
                [
                    "question": {
                        ...
                    },
                    "variation": {
                        ...
                    }
                ],
                ...
            ],
        "synonym_change": 
            [
                [
                    "question": {
                        ...
                    },
                    "variation": {
                        ...
                    }
                ],
                ...
            ],
            ...
        }
    }
    """
    # load questions
    questions = dd.load.Json("data/questions.json")

    # load answers
    answers = dd.load.Json(f"results/{llm_model.replace(':', '_')}/answers.json")
    answers_stats = []
    for answer in answers:
        answer_stats = {}
        answer_stats["id"] = answer["id"]
        # Find the corresponding question with the same id
        question_match = next((q for q in questions if q["id"] == answer["id"]), None)
        og_question = question_match["question"]
        
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
                        "jaro_distance": Levenshtein.jaro(og_question, answer[variation][i]),
                        "jaro_winkler_distance": Levenshtein.jaro_winkler(og_question, answer[variation][i])
                    }
                
                question_variation = question_match[variation]
                answer_stats_question_variation = {
                        "levenshtein_distance": Levenshtein.distance(question_variation, answer[variation][i]),
                        "indel_distance": Levenshtein.ratio(question_variation, answer[variation][i]),
                        "hamming_distance": Levenshtein.hamming(question_variation, answer[variation][i]),
                        "jaro_distance": Levenshtein.jaro(question_variation, answer[variation][i]),
                        "jaro_winkler_distance": Levenshtein.jaro_winkler(question_variation, answer[variation][i])
                    }

                temp_variation_i = {
                    "counts": counts,
                    "question": answer_stats_og_question,
                    "variation": answer_stats_question_variation
                }
                answer_stats[variation].append(temp_variation_i)
            
        answers_stats.append(answer_stats)

    # save answers stats
    dd.save.Json(f"results/{llm_model.replace(':', '_')}/answers_lexical_distances.json", answers_stats, mode="w")

    return True


def answers_embeddings_distances(
    llm_model: str,
    variations: list) -> bool:
    """
    """
    save_file_name = f"results/{llm_model.replace(':', '_')}/answers_embeddings_distances.json"
    if os.path.exists(save_file_name):
        print(f"Answers embeddings distances already exist in {save_file_name}. Skipping...")
        return True

    questions_embeddings = dd.load.Json(f"results/questions/questions_embeddings.json")
    answer_embeddings = dd.load.Json(f"results/{llm_model.replace(':', '_')}/answers_embeddings.json")
    answer_embeddings_distances = []
    for answer_embedding in answer_embeddings:
        answer_embedding_distances = {}
        answer_embedding_distances["id"] = answer_embedding["id"]
        question_answer_embedding = np.mean(np.array(answer_embedding["question"]), axis=0)

        # question_dict = next(q for q in questions_embeddings if q["id"] == answer_embedding["id"])
        # question_embedding = question_dict["question"]

        for variation in variations:
            answer_embedding_distances[variation] = []
            for i in range(len(answer_embedding[variation])):
                current_embedding = np.array(answer_embedding[variation][i])
                euclidean_distance = np.linalg.norm(current_embedding - question_answer_embedding)
                cosine_distance = 1 - np.dot(current_embedding, question_answer_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(question_answer_embedding))
                angle_degrees = calculate_angle_degrees(current_embedding, question_answer_embedding)

                # euclidean_distance = np.linalg.norm(current_embedding - question_embedding)
                # cosine_distance = 1 - np.dot(current_embedding, question_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(question_embedding))
                # angle_degrees = calculate_angle_degrees(current_embedding, question_embedding)

                answer_embedding_distances[variation].append({
                    "euclidean_distance": euclidean_distance,
                    "cosine_distance": cosine_distance,
                    "angle_degrees": angle_degrees
                })
        answer_embeddings_distances.append(answer_embedding_distances)
    dd.save.Json(save_file_name, answer_embeddings_distances, mode="w")
    return True

