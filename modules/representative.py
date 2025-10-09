import datadocket as dd
import numpy as np

def representative_answers(
    variations: list,
    model_name: str) -> bool:
    """
    """
    questions = dd.load.Json("data/questions.json")
    answers = dd.load.Json(f"results/{model_name}/answers.json")
    distances = dd.load.Json(f"results/{model_name}/answers_embeddings_distances.json")
    stats = dd.load.Json(f"results/{model_name}/answers_embeddings_distances_stats.json")
    closest_value, furthest_value = np.inf, 0
    closest_text, furthest_text = "", ""
    representative_answers = []

    for variation in variations:
        for q, a, d in zip(questions, answers, distances):
            temp_answer = {}
            id = q["id"]    
        
            mean = stats[variation]["cosine_distance"]["mean"]

            for i, iteration in enumerate(d[variation]):
                if abs(iteration["cosine_distance"] - mean) < closest_value:
                    closest_value = abs(iteration["cosine_distance"] - mean)
                    closest_text = a[variation][i]
                    closest_question_variation = q[variation][0]
                    closest_question = q["question"][0]
                    closest_index = id
                    closest_variation = variation
                if abs(iteration["cosine_distance"] - mean) > furthest_value:
                    furthest_value = abs(iteration["cosine_distance"] - mean)
                    furthest_text = a[variation][i]
                    furthest_question_variation = q[variation][0]
                    furthest_question = q["question"][0]
                    furthest_index = id
                    furthest_variation = variation

            temp_answer[variation] = {
                "closest_value": closest_value,
                "closest_index": closest_index,
                "closest_question_variation": closest_question_variation,
                "closest_question": closest_question,
                "closest_answer": closest_text,
                "closest_variation": closest_variation,
                "furthest_value": furthest_value,
                "furthest_index": furthest_index,
                "furthest_question_variation": furthest_question_variation,
                "furthest_question": furthest_question,
                "furthest_answer": furthest_text,
                "furthest_variation": furthest_variation
            }
        representative_answers.append(temp_answer)

    dd.save.Json(f"results/{model_name}/representative_answers.json", representative_answers, mode="w")

    return True
