# How to run
## Clone repo
```
clone https://github.com/enricgrau/word_sensitivity_analysis_llms.git
```
## Go to the created directpry
```
cd word_sensitivity_analysis_llms
```
## Install ollama
### For Linux
```
curl -fsSL https://ollama.com/install.sh | sh
```
### For Windows
```
https://ollama.com/download/windows
```
## Create virtual environment
```
py -m venv venv_se
```
# Activate environment
```
.\venv_se\Scripts\activate
```
## Install requirements.txt
```
pip install -r requirements.txt
```
## Run ollama
```
ollama serve
```
# Run main.py
```
py main.py
```
# Data
All this work revolves around the file `data/questions.json`, which contain a list of 100 different questions and their variations. the structure of the file is as follows:
```
[
    {
        "id": 0, # id of the question to track throughout the process
        "question": "original question from wich variations are created",
        "synonym_change": "change one word in the question with a synonym",
        "antonym_change": "change one word in the question with an antonym",
        "paraphrase_change": "paraphrase the question",
        "letter_change": "change one letter in the question, a typo"
    },
    {
        "id": 1,
        ...
    },
    ...
]
```
# Output
## From `questions.json`
### Data files

- `results/questions/questions_embeddings.json`: embeddgins of the questions and their variations.
- `results/questions/questions_embeddings_distances.json`:
- `results/questions/questions_embeddings_distances_stats.json`:
- `results/questions/questions_lexical_distances.json`: lexical stats of the questions and variations.
- `results/questions/questions_lexical_stats.json`: average lexical stats of the questions and variations.
### Plots
- `results/questions/plots/box_plot_questionis_lexical_stats_plots.png`: plot 

##  From each model
Each model defined in the variable `LLM_MODELS` yields the following files:
- `results/{llm_model}/answers.json`: 
- `results/{llm_model}/answers_embeddings.json`: 
- `results/{llm_model}/average_answers_embeddings.json`:
- `results/{llm_model}/answers_lexical_distances.json`:
- `results/{llm_model}/average_answers_lexical_distances.json`:
- `results/{llm_model}/answers_embeddings_distances.json`:
- `results/{llm_model}/answers_embeddings_distances_stats.json`:


### Plot files
- `results/plots/.png`:
- `results/plots/.png`:
- `results/plots/.png`:
- `results/plots/.png`:
- `results/plots/.png`:
- `results/plots/.png`:

