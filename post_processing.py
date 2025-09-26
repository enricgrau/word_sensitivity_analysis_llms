import datadocket as dd
import numpy as np
import pickle
import Levenshtein

def relative_directions(
    average_embeddings_file: str,
    pca_model_file: str,
    variations: list) -> bool:
    """
    """
    
    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    # load PCA model
    with open(pca_model_file, "rb") as f:
        pca = pickle.load(f)

    def calculate_relative_angle_degrees(question_vec, variation_vec):
        """
        Calculate the angle of variation_vec relative to question_vec as the reference (0,0)
        Returns angle in degrees (0 to 360)
        """
        # Convert to numpy arrays
        question_vec = np.array(question_vec)
        variation_vec = np.array(variation_vec)
        
        # Calculate the difference vector (variation - question)
        diff_vec = variation_vec - question_vec
        
        # Calculate angle with x-axis (using first component as x-axis reference)
        # For high-dimensional vectors, we'll use the first two principal components
        # or project onto a 2D plane for angle calculation
        
        # Method 1: Use first two dimensions for angle calculation
        if len(diff_vec) >= 2:
            x_component = diff_vec[0]
            y_component = diff_vec[1]
        else:
            # Fallback: use first dimension and magnitude
            x_component = diff_vec[0]
            y_component = np.linalg.norm(diff_vec[1:]) if len(diff_vec) > 1 else 0
        
        # Calculate angle in degrees
        angle_rad = np.arctan2(y_component, x_component)
        angle_deg = np.degrees(angle_rad)
        
        # Convert to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg

    # start directions list
    directions = []

    # iterate over average embeddings
    for average_data in average_embeddings:
        question_directions = {}
        question_directions["id"] = average_data["id"]
        
        # Get the question vector as reference
        question_vec = np.array(average_data["question"])

        for variation in variations:
            if variation == "question":
                # Question is the reference point, so angle is 0
                question_directions[variation] = 0.0
            else:
                # Calculate angle relative to the question vector
                variation_vec = np.array(average_data[variation])
                angle_degrees = calculate_relative_angle_degrees(question_vec, variation_vec)
                question_directions[variation] = angle_degrees

        directions.append(question_directions)

    # save directions
    dd.save.Json("data/directions.json", directions, mode="w")

    # get average directions for each variation type
    average_directions = {}
    for variation in variations:
        list_directions = []
        for direction in directions:
            list_directions.append(direction[variation])
        
        # Calculate circular statistics for angles
        # Convert to radians for circular mean calculation
        angles_rad = np.radians(list_directions)
        
        # Calculate circular mean
        cos_mean = np.mean(np.cos(angles_rad))
        sin_mean = np.mean(np.sin(angles_rad))
        circular_mean_rad = np.arctan2(sin_mean, cos_mean)
        circular_mean_deg = np.degrees(circular_mean_rad)
        if circular_mean_deg < 0:
            circular_mean_deg += 360
        
        # Calculate circular standard deviation
        R = np.sqrt(cos_mean**2 + sin_mean**2)  # Resultant length
        circular_std_deg = np.degrees(np.sqrt(-2 * np.log(R)))
        
        average_directions[variation] = {
            "average": circular_mean_deg,
            "standard_deviation": circular_std_deg,
            "median": np.median(list_directions),
            "min": np.min(list_directions),
            "max": np.max(list_directions)
        }

    # save average directions
    dd.save.Json("data/relative_directions_stats.json", average_directions, mode="w")


def average_question_sentence_stats(
    question_file: str,
    variations: list) -> bool:
    """
    """

    # load question sentence stats
    question_sentence_stats = dd.load.Json(question_file.replace(".json", "_lexical_stats.json"))

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
            "distances": {m: float(np.mean(metric_values[m])) if metric_values[m] else None for m in metrics},
            "counts": {c: float(np.mean(count_values[c])) if count_values[c] else None for c in count_metrics}
        }
    
    dd.save.Json(question_file.replace(".json", "_average_lexical_stats.json"), average_question_sentence_stats, mode="w")
    
    return True

def question_sentence_stats(
    questions_file: str,
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
            "jaro_distance": float,
            "jaro_winkler_distance": float
        },
        "synonym_change": {
            ...
        },
        ...
    }
    """

    # load questions
    questions = dd.load.Json(questions_file)

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
                "jaro_winkler_distance": Levenshtein.jaro_winkler(og_question, question[variation])
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
    dd.save.Json(questions_file.replace(".json", "_lexical_stats.json"), questions_stats, mode="w")

    return True


def average_answer_sentence_stats(
    answer_file: str,
    variations: list) -> bool:
    """
    Calculate average statistics for answer sentence stats according to the structure
    created by answer_sentence_stats function.
    """

    # Load the answer stats file (should be *_lexical_stats.json)
    stats_file = answer_file.replace(".json", "_lexical_stats.json")
    answer_stats = dd.load.Json(stats_file)

    # Prepare to accumulate metrics for each variation
    metrics = {}
    count_metrics = {}

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
    avg_file = answer_file.replace(".json", "_average_lexical_stats.json")
    dd.save.Json(avg_file, average_metrics, mode="w")
    
    return True

def answer_sentence_stats(
    questions_file: str,
    answers_file: str,
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
    questions = dd.load.Json(questions_file)

    # load answers
    answers = dd.load.Json(answers_file)
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
    dd.save.Json(answers_file.replace(".json", "_lexical_stats.json"), answers_stats, mode="w")

    return True


def average_embedding_distances(
    distances_file: str,
    distances_stats_file: str,
    average_embeddings_file: str,
    variations: list) -> bool:
    """
    """
    
    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    # start distances list
    distances = []

    # iterate over average embeddings
    for average_data in average_embeddings:
        question_distances = {}
        question_distances["id"] = average_data["id"]

        for variation in variations:
            # Calculate distance between the two vectors
            # First, get the difference vector, then calculate its norm
            vector1 = np.array(average_data[variation])
            vector2 = np.array(average_data["question"])
            euclidean_distance = np.linalg.norm(vector1 - vector2)
            cosine_distance = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

            question_distances[variation] = {
                "euclidean_distance": euclidean_distance,
                "cosine_distance": cosine_distance
            }

        distances.append(question_distances)

    # save distances
    dd.save.Json(distances_file, distances, mode="w")

    # get average distances for each variation type
    average_distances = {}
    for variation in variations:
        list_euclidean_distances, list_cosine_distances = [], []
        for distance in distances:
            list_euclidean_distances.append(distance[variation]["euclidean_distance"])
            list_cosine_distances.append(distance[variation]["cosine_distance"])
        
        average_distances[variation] = {
            "euclidean_distance": {
                "average": np.mean(list_euclidean_distances),
                "standard_deviation": np.std(list_euclidean_distances),
                "median": np.median(list_euclidean_distances),
                "min": np.min(list_euclidean_distances),
                "max": np.max(list_euclidean_distances)
            },
            "cosine_distance": {
                "average": np.mean(list_cosine_distances),
                "standard_deviation": np.std(list_cosine_distances),
                "median": np.median(list_cosine_distances),
                "min": np.min(list_cosine_distances),
                "max": np.max(list_cosine_distances)
            }
        }

    # save average distances
    dd.save.Json(distances_stats_file, average_distances, mode="w")

    return True


def average_embedding_directions(
    directions_file: str,
    directions_stats_file: str,
    average_embeddings_file: str,
    variations: list) -> bool:
    """
    """

    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    def calculate_angle_degrees(vec1, vec2):
        """
        Calculate the angle between two vectors in degrees (0 to 360)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine of angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        
        # Calculate angle in radians
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    # start directions list
    directions = []

    # iterate over average embeddings
    for average_data in average_embeddings:
        question_directions = {}
        question_directions["id"] = average_data["id"]

        for variation in variations:
            # Calculate angle between the two vectors
            vector1 = np.array(average_data["question"])
            vector2 = np.array(average_data[variation])
            angle_degrees = calculate_angle_degrees(vector1, vector2)
            question_directions[variation] = angle_degrees

        directions.append(question_directions)

    # save directions
    dd.save.Json(directions_file, directions, mode="w")

    # get average directions for each variation type
    average_directions = {}
    for variation in variations:
        list_directions = []
        for direction in directions:
            list_directions.append(direction[variation])
        
        average_directions[variation] = {
            "average": np.mean(list_directions),
            "standard_deviation": np.std(list_directions),
            "median": np.median(list_directions),
            "min": np.min(list_directions),
            "max": np.max(list_directions)
        }

    # save average directions
    dd.save.Json(directions_stats_file, average_directions, mode="w")

    return True
