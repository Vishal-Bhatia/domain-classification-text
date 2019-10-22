# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# path_train : location of test file
# Code starts here
##Loading the CSV data onto a Pandas dataframe
df = pd.read_csv(path_train)

##Defining a function to check every row for a category match, and applying the same
col_list = df.columns[1:].tolist()
def label_race(row):
    for i in range(0, len(col_list)):
        if row[i + 1] == "T":
            return col_list[i]
df["category"] = df.apply(lambda row: label_race(row), axis = 1)

##Dropping the unnecessary columns
df.drop(col_list, inplace = True, axis = 1)




# --------------
from sklearn.feature_extraction.text import TfidfVectorizer

# Code starts here
# Sampling only 1000 samples of each category
##Sampling the data as instructed
df = df.groupby("category").apply(lambda x: x.sample(n = 1000, random_state = 0))

##Saving the text data as a variable, and lower-casing it
all_text = df["message"].str.lower().tolist()

##Intantiating a TfidfVectorizer object with stopwords, and applying the same to yield the X dataset
tfidf = TfidfVectorizer(stop_words = "english")
tfidf.fit(all_text)
vector_tfidf = tfidf.fit_transform(all_text)
X = vector_tfidf.toarray()

##Intantiating a LabelEncoder object\, and applying the same to yield the y dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df["category"])




# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here
##Applying train-test split as instructed
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

##Intantiating a LogisticRegression model, fitting the same, and saving its accuracy in a variable
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_val)
log_accuracy = accuracy_score(y_val, y_pred)

##Intantiating a MultinomialNB model, fitting the same, and saving its accuracy in a variable
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_val)
nb_accuracy = accuracy_score(y_val, y_pred)

##Intantiating a LinearSVC model, fitting the same, and saving its accuracy in a variable
lsvm = LinearSVC(random_state = 0)
lsvm.fit(X_train, y_train)
y_pred = lsvm.predict(X_val)
lsvm_accuracy = accuracy_score(y_val, y_pred)




# --------------
# path_test : Location of test data

# Code starts here

#Loading the dataframe
##Loading the test CSV data onto a Pandas dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
##Creating the "category" column as in train data, and dropping the unnecessary columns
df_test["category"] = df_test.apply(lambda row: label_race (row), axis = 1)
df_test.drop(col_list, inplace = True, axis = 1)

##Creating the X_test and y_test dataframes as with train data
all_text = df_test["message"].str.lower().tolist()
vector_tfidf = tfidf.transform(all_text)
X_test = vector_tfidf.toarray()
y_test = le.transform(df_test["category"])

##Checking the accuracy of the earlier trained models on the test data
y_pred = log_reg.predict(X_test)
log_accuracy_2 = accuracy_score(y_test, y_pred)
y_pred = nb.predict(X_test)
nb_accuracy_2 = accuracy_score(y_test, y_pred)
y_pred = lsvm.predict(X_test)
lsvm_accuracy_2 = accuracy_score(y_test, y_pred)




# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]

# Code starts here
##Creating the id2word dictionary
dictionary = corpora.Dictionary(doc_clean)

##Creating a word corpus
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

##Intantiating an LSI model, and printing the topics
lsimodel = LsiModel(corpus = doc_term_matrix, num_topics = 5, id2word = dictionary)
pprint(lsimodel.print_topics())




# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return topic_list, coherence_values


# Code starts here
##Calling the function as instructed
topic_list, coherence_value_list = compute_coherence_values(dictionary = dictionary, corpus = doc_term_matrix, texts = doc_clean, start = 1, limit = 41, step = 5)

##Outputting the optimal topic count
opt_topic_index = np.where(np.array(coherence_value_list) == np.max(coherence_value_list))[0][0]
opt_topic = topic_list[opt_topic_index]

##Intantiating the LDA model, and printing the top 5 topics
lda_model = LdaModel(corpus = doc_term_matrix, num_topics = opt_topic, id2word = dictionary, iterations = 10, passes = 30, random_state = 2)
pprint(lda_model.print_topics(5))




