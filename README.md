# CMPS 6730 Philosophy Natural Language Processing Project

## Overview

This repository contains to Bobby Becker's project for CMPS 4730/6730 at Tulane University, which applies advanced NLP techniques to analyze and interpret major philosophical texts. The project utilizes Latent Dirichlet Allocation (LDA) and Word2Vec models to explore thematic connections within and between 43 works of philosophy, inlcuding philosophers such as Plato, Aristotle, Marx, Nietzsche, Kant, and others.

## Project Artifacts

### 1. `final_project.ipynb`

This Jupyter notebook showcases the data analysis of the project. Here's what it contains:

#### Latent Dirichlet Allocation (LDA)
- **Text Preparation**: Each of the 43 philosophical works is preprocessed to remove stopwords and other non-informative text elements. This clean text is then tokenized.
- **LDA Processing**: The LDA model is applied to the tokenized text to extract key themes, each represented by four words. This thematic extraction helps in understanding the central topics discussed in each work.

#### Word2Vec
- **Vector Training**: Post-LDA, a Word2Vec model is trained on the corpus to generate word vectors for the identified thematic words.
- **Vector Averaging**: For each text, the vectors of its four theme words are averaged to create a single vector that represents the overall thematic essence of the text.

#### Principal Component Analysis (PCA)
- **Dimensionality Reduction**: The high-dimensional vectors are reduced to 2D and 3D using PCA, which creates a visual representation of the vector similarities between the philosophical works. Philosphical works are shown compared to each other, compared to the word vectors representing each philosopher, and to word vectors representing the philosphical themes.

### 2. `flask.py` and `plato_matcher_online.py`

The web component uses Flask for an interactive interface that allows the user to make queries into Plato's texts. This can be viewed as a potential application of the research portion of this project.

#### Workflow
1. **Text Segmentation**: First, works of Plato are loaded in, tokenized, & segmented into 500 parts.
2. **Latent Dirichlet Allocation (LDA)** Then, using LDA, we analyze each portion of the text and generate 6 words to represent that passage.
3. **Vector Representation**: We then load in a Word2Vec model, trained on all 43 philosophical works used in the Jupyter notebook. The 6 words generated by each passage are averaged together to create a unique vector to represent each passage. 
4. **User Interaction**: Users submit text through the web interface, which goes through the same process as the passages of Plato: 6 words are generated by LDA to represent the user input, and those 6 words are vectorized and averaged to create a vector representing the user's input.
5. **Similarity Calculation**: We then calculate cosine similarity between the user's vector and each passage's vector to find the best match.
6. **Text Refinement and Citation**: Once the most relevant passage is identified, a GPT-3.5 model is used to identify and rewrite the most important portion of the passage and provide a citation to the user.

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- Gensim
- NLTK
- sklearn
- OpenAI API key

### Installation

bash:
git clone https://github.com/yourusername/cmps-6730-nlp-project.git
cd cmps-6730-nlp-project
pip install -r requirements.txt

### OpenAI Key
Put in your OpenAI key at the top of the 'plato_online_matcher.py' file.

### Running the Application
python flask.py and navigate to http://127.0.0.1:5000/ in your web browser

### Example Usage:
Question:
<img width="1249" alt="Question_Friendship" src="https://github.com/tulane-cmps6730/project-philosophy/assets/86581611/dda477e9-7bb4-40fe-9d4d-743ca5f0b75e">

Answer:
<img width="1216" alt="Answer_Friendship" src="https://github.com/tulane-cmps6730/project-philosophy/assets/86581611/5f0af556-7a8d-4e20-826d-3045484efec4">


Question:
<img width="1233" alt="Question_Politics" src="https://github.com/tulane-cmps6730/project-philosophy/assets/86581611/535ed02b-1305-4307-97f6-f54b00254f66">

Answer:
<img width="1216" alt="Answer_Politics" src="https://github.com/tulane-cmps6730/project-philosophy/assets/86581611/f15d67f8-622f-411f-914b-d872e4686203">


