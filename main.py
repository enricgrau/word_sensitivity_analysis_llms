import datadocket as dd

from modules.answers import get_answers
from modules.embeddings import questions_embeddings
from plots import stats_boxplot_2, stats_boxplot, questions_lexical_stats_plot, directions_plot, word_count_plot
from plots import all_models_word_count_plot, all_models_lexical_stats_plot
from post_processing import questions_embeddings_distances, question_sentence_distances, embedding_distances_stats
from post_processing import average_question_sentence_stats, answer_sentence_stats, answer_sentence_distances, answers_embeddings_distances


# params
ITERATIONS = 10 # number fo times to run the LLM on a single question
LLM_MODELS = ["gemma3:270m", "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"] # models to run the LLM on
SYSTEM_PROMPT = "Answer this question in one sentence."
VARIATIONS = ["question", "synonym_change", "antonym_change", "paraphrase_change", "letter_change"]
SHOW_PLOTS = False

# questions stats
question_sentence_distances(
    variations=VARIATIONS)

average_question_sentence_stats(
    variations=VARIATIONS)

questions_embeddings(
    variations=VARIATIONS)

questions_embeddings_distances(
    variations=VARIATIONS)

embedding_distances_stats(
    variations=VARIATIONS,
    data_file="results/questions/questions_embeddings_distances.json",
    save_file="results/questions/questions_embeddings_distances_stats.json")

# plots
dd.utils.MakeDir("results/questions/plots")

questions_lexical_stats_plot(
    variations=VARIATIONS,
    comparison="question",
    show=SHOW_PLOTS)

word_count_plot(
    variations=VARIATIONS,
    show=SHOW_PLOTS)

stats_boxplot(
    variations=VARIATIONS,
    variable_type="cosine_distance",
    hide_outliers=True,
    data_file="results/questions/questions_embeddings_distances_stats.json",
    save_file="results/questions/plots/questions_embeddings_distances_boxplot.png",
    show=SHOW_PLOTS)

directions_plot(
    variations=VARIATIONS[1:],
    model_name="Questions",
    data_file="results/questions/questions_embeddings_distances_stats.json",
    save_file="results/questions/plots/questions_embeddings_distances_directions_relative.png",
    show=SHOW_PLOTS,
    relative=True)

for model in LLM_MODELS:
    # make directory
    model_name_fix = model.replace(':', '_')
    dd.utils.MakeDir(f"results/{model_name_fix}")

    # processing
    get_answers(
        iterations=ITERATIONS, 
        llm_model=model, 
        system_prompt=SYSTEM_PROMPT, 
        variations=VARIATIONS)

    # metrics
    answer_sentence_distances(
        llm_model=model,
        variations=VARIATIONS)

    # sentence metrics' stats
    answer_sentence_stats(
        llm_model=model,
        variations=VARIATIONS)
    
    # embeddings metrics for all answers
    answers_embeddings_distances(
        llm_model=model,
        variations=VARIATIONS)
    
    embedding_distances_stats(
        variations=VARIATIONS,
        data_file=f"results/{model_name_fix}/answers_embeddings_distances.json",
        save_file=f"results/{model_name_fix}/answers_embeddings_distances_stats.json")

    # plots
    dd.utils.MakeDir(f"results/{model_name_fix}/plots")

    stats_boxplot_2(
        variations=VARIATIONS,
        variable_type="cosine_distance",
        hide_outliers=True,
        data_file=f"results/{model_name_fix}/answers_embeddings_distances.json",
        save_file=f"results/{model_name_fix}/plots/answers_embeddings_distances_boxplot.png",
        show=SHOW_PLOTS)

    directions_plot(
        variations=VARIATIONS[1:],
        model_name=model,
        data_file=f"results/{model_name_fix}/answers_embeddings_distances_stats.json",
        save_file=f"results/{model_name_fix}/plots/answers_embeddings_distances_directions.png",
        relative=True,
        show=SHOW_PLOTS)
    
    questions_lexical_stats_plot(
        data_file=f"results/{model_name_fix}/average_answers_lexical_distances.json",
        save_file=f"results/{model_name_fix}/plots/answers_lexical_stats_question_plot.png",
        variations=VARIATIONS,
        comparison="question",
        show=SHOW_PLOTS)

    questions_lexical_stats_plot(
        data_file=f"results/{model_name_fix}/average_answers_lexical_distances.json",
        save_file=f"results/{model_name_fix}/plots/answers_lexical_stats_variation_plot.png",
        variations=VARIATIONS,
        comparison="variation",
        show=SHOW_PLOTS)


# plot all models in the same plot

# all_models_word_count_plot(
#     variations=VARIATIONS,
#     models=LLM_MODELS[:-1],
#     variable_type="character_count",
#     show=SHOW_PLOTS)

# all_models_word_count_plot(
#     variations=VARIATIONS,
#     models=LLM_MODELS[:-1],
#     variable_type="word_count",
#     show=SHOW_PLOTS)

# all_models_lexical_stats_plot(
#     variations=VARIATIONS,
#     models=LLM_MODELS,
#     show=SHOW_PLOTS)
