import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from termcolor import colored

# Loading questions and answers in separate lists
questions = []
answers = []
qa_files = ['Data/PyTorch Interview Questions - adaface.csv','Data/PyTorch Interview Questions - GitHub.csv', 'Data/PyTorch Interview Questions - ChatGPT.csv', 'Data/PyTorch Interview Questions - javatpoint.csv', 'Data/ML Interview Questions - Kaggle.csv']

# Reading CSV files and extracting questions and answers
for file in qa_files:
    try:
        temp_dataframe = pd.read_csv(file, encoding='utf-8', delimiter=',', on_bad_lines='skip')  # Try reading with utf-8
    except UnicodeDecodeError:
        temp_dataframe = pd.read_csv(file, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')  # Fallback to ISO-8859-1 encoding
    questions.extend(temp_dataframe['QuestionBody'].str.lower().tolist())
    answers.extend(temp_dataframe['AnswerBody'].str.lower().tolist())

# Vectorizing the questions
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(questions)

# Applying term frequency inverse document frequency (TF-IDF)
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_vec)

# Function to handle conversation
def conversation(im):
    global tfidf, answers, X_tfidf
    #Vecorizing and transforming the user question in order to be compared to other questions in data
    Y_vec = vectorizer.transform(im)
    Y_tfidf = tfidf.transform(Y_vec)
    cos_sim = cosine_similarity(Y_tfidf, X_tfidf) #Get similarity between question and other q&a Q's
    
    # If the highest cosine similarity is below a threshold, we respond with not understanding else we return an answer
    if max(cos_sim[0]) < 0.6:
        return "sorry, I did not quite understand that"
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