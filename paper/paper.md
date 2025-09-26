# Words don't matter: effects in large language model unstructured responses by minor lexical changes in prompts.

## Abstract

Prompt engineering has become an essential skill for AI engineers and data scientists, as well-crafted prompts enable better results and optimal costs. While research has extensively studied the effects of different prompt aspects—focusing on structures, formatting, and strategies—very little work has explored the impact of minor lexical changes, such as single character or word modifications. Although it is well-documented that such changes affect model outputs in diverse ways, most studies compare outputs by measuring accuracy, structure, or word usage. However, little research has examined how small changes affect the meaning of unstructured outputs while accounting for the stochastic nature of large language models (LLMs). This work performs experiments to explore these effects systematically. The results suggest that while minor lexical changes do influence how LLMs respond, these effects are minimal and show no practical consequence in the semantic content of the generated responses.

## Introduction

Large Language Models (LLMs) have revolutionized natural language processing and are increasingly deployed across diverse applications, from content generation to decision support systems. As these models become more prevalent in critical applications, understanding their robustness and sensitivity to input variations has become paramount. A growing body of research has revealed that LLMs exhibit surprising sensitivity to seemingly minor changes or in their input prompts, often producing substantially different outputs in response to imperceptible variations [sclar_2023, chatterjee_2024, zhan_2024].

The phenomenon of prompt sensitivity has been extensively documented across various dimensions. Salinas et al. [salinas_2024] demonstrated that even minimal perturbations, such as adding a space at the end of a prompt, can cause LLMs to change their classification decisions. Sclar et al. [sclar_2023] found that LLMs can exhibit performance differences of up to 76 accuracy points when evaluated with different prompt formats, highlighting the critical importance of prompt design choices. Similarly, Zhan et al. [zhan_2024] revealed that LLMs are over-sensitive to lexical variations in task instructions, even when these variations are imperceptible to humans. Chatterjee et al. [chatterjee_2024] developed POSIX, a novel PrOmpt Sensitivity IndeX that quantifies the relative change in loglikelihood when replacing prompts with intent-preserving alternatives, finding that paraphrasing results in the highest sensitivity in open-ended generation tasks. Hu et al. [hu_2024] demonstrated that even short prefixes can steer RAG-based LLMs toward factually incorrect answers, while Sahoo et al. [sahoo_2024] provided a comprehensive survey of prompt engineering techniques, emphasizing the need for systematic understanding of prompt design methodologies.

While existing research has primarily focused on structured tasks and classification problems, there remains a significant gap in understanding how minor lexical changes affect the semantic content and meaning of unstructured LLM responses. Most studies have concentrated on measuring accuracy, performance metrics, or structural changes in outputs, but few have systematically examined the semantic implications of prompt variations in open-ended generation tasks.

This work addresses this gap by conducting a comprehensive analysis of how minor lexical modifications to single-sentence questions affect the semantic content of LLM responses. Specifically, this work investigates four types of lexical variations: (1) synonym substitution, where a word is replaced with a semantically equivalent term; (2) antonym substitution, where a word is replaced with its opposite; (3) paraphrasing, where the entire question is rephrased while preserving meaning; and (4) single-letter typos, where one character is altered to create a misspelling.

To quantify the impact of these variations, a dual-metric approach is employed, capturing both semantic and lexical differences. First, embedding-based semantic differences are measured using an open-source sentence embedding model to assess how the meaning of responses changes across prompt variations. Second, lexical differences are analyzed to understand the surface-level changes in word choice, sentence structure, and linguistic patterns.

This experimental design systematically generates variations of single-sentence questions and collects multiple responses from LLMs to account for their stochastic nature. By analyzing both embedding distances and lexical metrics, a comprehensive view of how minor prompt changes propagate through the model's generation process and manifest in the final outputs is provided.

The findings from this study have important implications for prompt engineering practices, model evaluation methodologies, and the deployment of LLMs in real-world applications where robustness to input variations is crucial. Understanding the relationship between prompt sensitivity and semantic stability is essential for developing more reliable and trustworthy language models.


## Methodology

### Large Language Models

For this experiment, the family of Gemma3 models were selected. This is due to the different sizes they offer: 270 million, 1 billion, 4 billion, 12 billion and 27 billion. This allows to test how model size affects the results. For all models, the system instructions was simply "Answer the questions in one single sentence" to indicate the model to respond in an unstructured way that is also comparable for each iteration.

### Embedding Model

The embedding model used is Nomic Embed [Nussbaum_2024A] [Nussbaum_2024B]. This is an open-source model widely used, showing good and consistent results. The mebedding model was not changed since it was a tool to measure output differences only. However, an extension of this study could include various embeddign models to ain insihts on this.

### Questions and variations

A set of 100 questions was created, each question with 4 different variations: (1) synonym substitution, where a word is replaced with a semantically equivalent term; (2) antonym substitution, where a word is replaced with its opposite; (3) paraphrasing, where the entire question is rephrased while preserving meaning; and (4) single-letter typos, where one character is altered to create a misspelling. This were developed using Gemini Pro and manually revised for quality asurance, making ure that each variation fits each type.

| Variation Type         | Example Variation                |
|------------------------|----------------------------------|
| Original               | "wht did the chicken cross the road? |
| Synonym Substitution   | "why did the chicken cross the street?" |
| Antonym Substitution   | "why didn't the chicken cross the road?" |
| Paraphrasing           | "what was the reason for the chicken crossing the road?" |
| Single-letter Typo     | "why did the chicken cross the roae?" |

### Metrics

Semantic and lexical proeprties where measured to contrast the different dimensions of the answers.

## Results

### Embedding differences

### Semantic differences

### Uncertenty

## Conclusions

