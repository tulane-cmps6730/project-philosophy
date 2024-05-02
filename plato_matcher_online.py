import gensim
from gensim.models import Word2Vec
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.ldamodel import LdaModel
import nltk
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

def split_string(s, num_parts):
    n = len(s)
    part_length = n // num_parts
    remainder = n % num_parts

    parts = []
    start = 0

    for i in range(num_parts):
        # If there's a remainder, add 1 to the part length for the first 'remainder' parts
        end = start + part_length + (1 if i < remainder else 0)
        parts.append(s[start:end])
        start = end

    return parts

Plato_Texts = ""

with open('textfiles/Plato_TheRepublic.txt', 'r', encoding='utf-8') as file:
    book11 = file.read()
    Plato_Texts += book11

with open('textfiles/Plato_TheSymposium.txt', 'r', encoding='utf-8') as file:
    book12 = file.read()
    Plato_Texts += book12

with open('textfiles/Plato_Apology.txt', 'r', encoding='utf-8') as file:
    book13 = file.read()
    Plato_Texts += book13

with open('textfiles/Plato_Sophist.txt', 'r', encoding='utf-8') as file:
    book14 = file.read()
    Plato_Texts += book14

with open('textfiles/Plato_Parmenides.txt', 'r', encoding='utf-8') as file:
    book27 = file.read()
    Plato_Texts += book27

with open('textfiles/Plato_Crito.txt', 'r', encoding='utf-8') as file:
    book28 = file.read()
    Plato_Texts += book28

with open('textfiles/Plato_Meno.txt', 'r', encoding='utf-8') as file:
    book29 = file.read()
    Plato_Texts += book29

with open('textfiles/Plato_Theaetetus.txt', 'r', encoding='utf-8') as file:
    book30 = file.read()
    Plato_Texts += book30


nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
custom_stop_words = ['cannot', 'must', 'one', 'also', 'thus', 'would', 'may', 'therefore', 'without', 'even', 'according', 'accordingly', 'greek', 'small', 'alpha', 'letter', 
                     'gamma', 'omicron', 'iota', 'epsilon', 'yet', 'still', 'something', 'perhaps', 'every', 'like', 'nu', 'sigma', 'us', 'tau', 'upon,', 'said', 'yes', 'e', 'replied', 'another','1', 'thou', 'unto', 'thy', 'thee', 'upon', 'obj', 'footnote',
                     'could', 'say', 'reply', '_', '3', '2', 'doth', 'men', 'man', 'tis', 'g', 'either', 'always', 'german']
stop_words = set(stopwords.words('english')).union(custom_stop_words)


def preprocess(text):
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in stop_words]
    return stopped_tokens

Plato_Text_Processed = preprocess(Plato_Texts)

texts = split_string(Plato_Text_Processed, 500)

def create_lda_model(texts, num_topics=1, passes=15):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, dictionary, corpus

def extract_words_from_topic(topic):
    return re.findall(r'"([^"]*)"', topic[1])

lda_models = []
topics_dict = {}
for i, text_group in enumerate(texts):
    lda_model, dictionary, corpus = create_lda_model([text_group]) 
    lda_models.append((lda_model, dictionary, corpus))
    #print(i)
    topics = lda_model.print_topics(num_words=6)
    top_words = []
    for topic in topics:
        #print(topic)
        top_words.extend(extract_words_from_topic(topic))
    topics_dict[i] = top_words
    #print("\n")

VecModel = Word2Vec.load("43texts_word2vec.model")
# Function to calculate the average vector for a list of words
def get_average_vector(words, model):
    vector_sum = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            vector_sum += model.wv[word]
            count += 1
    return vector_sum / count if count > 0 else None

# Function to calculate cosine similarity
def calculate_similarity(vector1, vector2):
    if vector1 is not None and vector2 is not None:
        return cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return 0

from gensim import corpora, models

def preprocess_and_vectorize(input_text, lda_model, dictionary, w2v_model):
    # Preprocess the text
    processed_text = preprocess(input_text)

    # Convert text to a bag-of-words format
    bow = dictionary.doc2bow(processed_text)

    # Get the LDA topic distribution for the input text
    topic_distribution = lda_model.get_document_topics(bow)

    # Find the dominant topic or get top N topics
    dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0]
    topic_words = lda_model.show_topic(dominant_topic[0], topn=3)

    # Extract words from the topic
    words = [word for word, prob in topic_words]

    # Compute the average vector for these words
    return get_average_vector(words, w2v_model)

# Example usage
#user_input = input("Enter your text: ")
#user_vector = preprocess_and_vectorize(user_input, lda_model, dictionary, VecModel)

# Comparing user input vector to topic vectors
#similarities = {}
#for topic_id, words in topics_dict.items():
 #   topic_vector = get_average_vector(words, VecModel)
  #  similarity_score = calculate_similarity(user_vector, topic_vector)
   # similarities[topic_id] = similarity_score

#list_of_ids = []
# Display the similarities
# Display the top 5 similarities
#sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
#for topic_id, score in sorted_similarities[:3]:  # Limit to top 5
    #print(f"Topic {topic_id} similarity: {score:.4f}")
 #   list_of_ids.append(topic_id)

#print(list_of_ids)
portions_of_text = split_string(Plato_Texts, 500)
#print(portions_of_text[list_of_ids[0]])


def summarize_text(text, user_input):
    openai.api_key = "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";

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # or another suitable model
        prompt=f"Look at this passage from Plato: \n\n{text} \n\n Now, rewrite, exactly as written, one paragraph (2-4 sentences) from the text most similar to the user input of '{user_input}'. Write it now, exactly as written: ",
        max_tokens=2000,  # Adjust based on how detailed you want the summary
        temperature=0.5,  # Adjust for creativity level; lower values make responses more deterministic
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

def cite_text(text):
    openai.api_key = "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";

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # or another suitable model
        prompt=f"Look at this passage from Plato: \n\n{text} \n\n Name the dialogue it is from:",
        max_tokens=100,  # Adjust based on how detailed you want the summary
        temperature=0.5,  # Adjust for creativity level; lower values make responses more deterministic
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()


# Example usage:
api_key = "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";
#important_text = portions_of_text[list_of_ids[0]]  # from your earlier code where you find the most relevant text section



def process_input(user_input):
    user_vector = preprocess_and_vectorize(user_input, lda_model, dictionary, VecModel)
    
    # Compute similarities with the text segments
    similarities = {}
    for topic_id, words in topics_dict.items():
        topic_vector = get_average_vector(words, VecModel)
        similarity_score = calculate_similarity(user_vector, topic_vector)
        similarities[topic_id] = similarity_score

    # Find the top matching segment
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_match_id = sorted_similarities[0][0] if sorted_similarities else None
    important_text = portions_of_text[top_match_id] if top_match_id is not None else ""

    # Use GPT-3.5 to summarize and cite the text
    summary = summarize_text(important_text, user_input)
    text_before = portions_of_text[top_match_id - 1]
    text_after = portions_of_text[top_match_id + 1]
    important_text_and_friends = text_before + important_text + text_after
    citation = cite_text(important_text_and_friends)

    return summary, citation