import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored

nlp = spacy.load('en_core_web_sm') # Needed for lemmatization

# Function to lemmatize the text (normalize words to their base forms)
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Loading questions and answers in separate lists
questions = []
answers = []
qa_files = ['Data/PyTorch Interview Questions - adaface.csv','Data/PyTorch Interview Questions - GitHub.csv', 'Data/PyTorch Interview Questions - ChatGPT.csv', 'Data/PyTorch Interview Questions - javatpoint.csv', 'Data/ML Interview Questions - Kaggle.csv']
#'Data Science - Stack Exchange PyTorch.csv',
# Reading CSV files and extracting questions and answ
#ers
for file in qa_files:
    try:
        # Attempt to read CSV files with utf-8 encoding
        temp_dataframe = pd.read_csv(file, encoding='utf-8', delimiter=',', on_bad_lines='skip')  # Try reading with utf-8
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 encoding due to some encoding errors
        temp_dataframe = pd.read_csv(file, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')  # Fallback to ISO-8859-1 encoding
    questions.extend(temp_dataframe['QuestionBody'].str.lower().apply(lemmatize_text).tolist())
    answers.extend(temp_dataframe['AnswerBody'].tolist())

# Load pre-trained Sentence-BERT model (BERT-based embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2') # 'all-MiniLM-L6-v2' OR 'bert-base-nli-mean-tokens'
question_embeddings = model.encode(questions)   # Generate embeddings for all questions

# Function to handle conversation
def conversation(input_message):
    global question_embeddings, answers, model
    lemmatized_message = lemmatize_text(input_message[0].lower()) #Lemmatize the input message
    query_vec = model.encode([lemmatized_message])[0] # Generate embedding for user question
    
    cos_sim = cosine_similarity([query_vec], question_embeddings) #Get similarity between question and other Q's

    # If the highest cosine similarity is below a threshold, we respond with following messages else we return answer
    if max(cos_sim[0]) < 0.1: 
        return "Can you speak English?"
    if max(cos_sim[0]) < 0.4: 
        return "Sorry, I did not quite understand that"
    if max(cos_sim[0]) < 0.55: 
        return "Do you mind rephrasing that and asking again?"
    else:
        return answers[np.argmax(cos_sim[0])]

# Main function for the chat system
def main():
    user_name = input("Please enter your username: ")
    ai_name = "Q&A support"
    while True:
        input_message = input(colored("{}: ".format(user_name), "cyan"))
        if input_message.lower() == 'bye':
            print("{}: bye!".format(ai_name))
            break
        else:
            print(colored("{}:".format(ai_name), "light_green") + "{}\n".format(conversation([input_message])))
main()