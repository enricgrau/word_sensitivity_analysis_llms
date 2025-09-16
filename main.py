import datadocket as dd
from modules.get_answers import _get_answers_
from modules.average_embedding import _average_embedding_
from modules.pca import _pca_
from plots import absolute_directions_plot, box_plot, pca_plot_all, relative_directions_plot
from post_processing import relative_directions, question_sentence_stats, average_question_sentence_stats, answer_sentence_stats, average_embedding_distances, average_embedding_directions, average_answer_sentence_stats


# params
ITERATIONS = 10 # number fo times to run the LLM on a single question
LLM_MODELS = ["gemma3:270m"] # models to run the LLM on
SYSTEM_PROMPT = "Answer the questions in one single sentence"
VARIATIONS = ["question", "synonym_change", "antonym_change", "paraphrase_change", "letter_change"]
QUESTIONS_FILE = "data/questions.json"
EMBEDDINGS_FILE = "data/embeddings.json"
ANSWERS_FILE = "data/answers.json"
AVERAGE_EMBEDDINGS_FILE = "data/average_embeddings.json"

for model in LLM_MODELS:
    # make directory
    model_name_fix = model.replace(':', '_')
    # dd.utils.MakeDir(f"results/{model_name_fix}")

    # _get_answers_(
    #     questions_file=QUESTIONS_FILE, 
    #     embeddings_file=EMBEDDINGS_FILE, 
    #     answers_file=ANSWERS_FILE, 
    #     iterations=ITERATIONS, 
    #     llm_model=model, 
    #     system_prompt=SYSTEM_PROMPT, 
    #     variations=VARIATIONS)

    # _average_embedding_(
    #     answers_file=ANSWERS_FILE,
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     variations=VARIATIONS)

    # _pca_(
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     pca_model_file=f"pca_model/pca_model_{model_name_fix}.pkl",
    #     variations=VARIATIONS)

    # question_sentence_stats(
    #     questions_file=QUESTIONS_FILE,
    #     variations=VARIATIONS)

    # average_question_sentence_stats(
    #     question_file=QUESTIONS_FILE,
    #     variations=VARIATIONS)

    # answer_sentence_stats(
    #     answers_file=ANSWERS_FILE,
    #     questions_file=QUESTIONS_FILE,
    #     variations=VARIATIONS)

    # average_answer_sentence_stats(
    #     answer_file=ANSWERS_FILE,
    #     variations=VARIATIONS)

    # average_embedding_distances(
    #     distances_file=f"data/distances_{model_name_fix}.json",
    #     distances_stats_file=f"data/distances_stats_{model_name_fix}.json",
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     variations=VARIATIONS)

    # average_embedding_directions(
    #     directions_file=f"data/directions_{model_name_fix}.json",
    #     directions_stats_file=f"data/directions_stats_{model_name_fix}.json",
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     variations=VARIATIONS)

    # relative_directions(
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     pca_model_file=f"pca_model/pca_model_{model_name_fix}.pkl",
    #     variations=VARIATIONS)

    ## PLOTS ##

    # box_plot(
    #     stat_file=f"data/distances_stats_{model_name_fix}.json",
    #     variations=VARIATIONS,
    #     show=False)
    
    # box_plot(
    #     stat_file=f"data/directions_stats_{model_name_fix}.json",
    #     variations=VARIATIONS,
    #     show=False)

    # pca_plot_all(
    #     average_embeddings_file=AVERAGE_EMBEDDINGS_FILE,
    #     pca_model_file=f"pca_model/pca_model_{model_name_fix}.pkl",
    #     variations=VARIATIONS,
    #     show=True)

    absolute_directions_plot(
        variations=VARIATIONS,
        show=True)

    # relative_directions_plot(
    #     variations=VARIATIONS,
    #     show=True)
    