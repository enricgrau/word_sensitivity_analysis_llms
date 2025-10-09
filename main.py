import datadocket as dd

from modules.answers import get_answers, get_embeddings
from modules.plots import stats_boxplot, questions_lexical_stats_plot, directions_plot, word_count_plot
from modules.plots import all_models_word_count_plot, all_models_lexical_stats_plot
from modules.post_processing import lexical_distances, lexical_distances_stats, embeddings_distances, embedding_distances_stats
from modules.representative import representative_answers

# params
ITERATIONS = 10 # number fo times to run the LLM on a single question
LLM_MODELS = ["gemma3:27b"]
LLM_MODELS = ["gemma3:270m", "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"] # models to run the LLM on
SYSTEM_PROMPT = "Answer this question in one sentence."
VARIATIONS = ["question", "synonym_change", "paraphrase_change", "letter_change", "antonym_change"] 
SHOW_PLOTS = False


lexical_distances(
    data_file="data/questions.json",
    save_file="results/questions/lexical_distances.json",
    variations=VARIATIONS)

lexical_distances_stats(
    data_file="results/questions/lexical_distances.json",
    save_file="results/questions/lexical_distances_stats.json",
    variations=VARIATIONS)

get_embeddings(
    data_file="data/questions.json",
    save_file="results/questions/questions_embeddings.json",
    variations=VARIATIONS)

embeddings_distances(
    data_file="results/questions/questions_embeddings.json",
    save_file="results/questions/questions_embeddings_distances.json",
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
    data_file="results/questions/lexical_distances_stats.json",
    save_file="results/questions/plots/lexical_stats_question_plot.png",
    show=SHOW_PLOTS)

word_count_plot(
    variations=VARIATIONS,
    data_file="results/questions/lexical_distances_stats.json",
    save_file="results/questions/plots/lexical_stats_word_count_plot.png",
    show=SHOW_PLOTS)

stats_boxplot(
    model_name="Questions",
    variations=VARIATIONS,
    variable_type="cosine_distance",
    data_file="results/questions/questions_embeddings_distances.json",
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
    print(f"\n############ Processing {model} ############")
    # make directory
    model_name_fix = model.replace(':', '_')
    dd.utils.MakeDir(f"results/{model_name_fix}")

    # processing
    get_answers(
        iterations=ITERATIONS, 
        llm_model=model, 
        system_prompt=SYSTEM_PROMPT, 
        variations=VARIATIONS)
    
    get_embeddings(
        data_file=f"results/{model_name_fix}/answers.json",
        save_file=f"results/{model_name_fix}/answers_embeddings.json",
        variations=VARIATIONS)

    lexical_distances(
        data_file=f"results/{model_name_fix}/answers.json",
        save_file=f"results/{model_name_fix}/lexical_distances.json",
        variations=VARIATIONS)

    lexical_distances_stats(
        data_file=f"results/{model_name_fix}/lexical_distances.json",
        save_file=f"results/{model_name_fix}/lexical_distances_stats.json",
        variations=VARIATIONS)
    
    embeddings_distances(
        data_file=f"results/{model_name_fix}/answers_embeddings.json",
        save_file=f"results/{model_name_fix}/answers_embeddings_distances.json",
        variations=VARIATIONS)
    
    embedding_distances_stats(
        variations=VARIATIONS,
        data_file=f"results/{model_name_fix}/answers_embeddings_distances.json",
        save_file=f"results/{model_name_fix}/answers_embeddings_distances_stats.json")

    # plots
    dd.utils.MakeDir(f"results/{model_name_fix}/plots")

    stats_boxplot(
        model_name=model,
        variations=VARIATIONS,
        variable_type="cosine_distance",
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
        data_file=f"results/{model_name_fix}/lexical_distances_stats.json",
        save_file=f"results/{model_name_fix}/plots/lexical_stats_question_plot.png",
        variations=VARIATIONS,
        comparison="question",
        show=SHOW_PLOTS)

    questions_lexical_stats_plot(
        data_file=f"results/{model_name_fix}/lexical_distances_stats.json",
        save_file=f"results/{model_name_fix}/plots/lexical_stats_variation_plot.png",
        variations=VARIATIONS,
        comparison="variation",
        show=SHOW_PLOTS)
    
    # reprsentative questions
    representative_answers(
        variations=VARIATIONS,
        model_name=model_name_fix)

# plot all models in the same plot
all_models_word_count_plot(
    variations=VARIATIONS,
    models=LLM_MODELS,
    variable_type="character_count",
    show=SHOW_PLOTS)

all_models_word_count_plot(
    variations=VARIATIONS,
    models=LLM_MODELS,
    variable_type="word_count",
    show=SHOW_PLOTS)

all_models_lexical_stats_plot(
    variations=VARIATIONS,
    models=LLM_MODELS,
    show=SHOW_PLOTS)
