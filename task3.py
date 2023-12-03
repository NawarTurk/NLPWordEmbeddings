
# COMP472A2 TASK3
# -- written by Chris Anglin -- 40216346

import os
import gensim
import nltk.tokenize as token

print('\n\n -> RUNNING TASK 3..')

print('\n -- data preprocessing : STARTING')
print(' -- preprocessing books into 2D array..')

# import and preprocess my chosen books:

path = os.path.expanduser('books/')

# this list will be used for holding the processed book data
preprocessed = []

for entry in os.listdir(path):
    if not entry.endswith('.txt'):
        continue
    with open('books/' + entry, 'r') as file:
        book = file.read()
        # tokenize books into sentences
        sentences = token.sent_tokenize(book)
        for sentence in sentences:
            # tokenize sentences into words
            preprocessed.append(token.word_tokenize(sentence))

print(' -- data preprocessing : DONE !')

print('\n -- model training : STARTING')
print(' -- creating the four new word2vec models..')

# chosen params for models (experimenting pretty much)
windowsize1 = 10
windowsize2 = 30
embeddingsize5 = 25
embeddingsize6 = 300

# create 4 word2vec models using the processed data from books and the above paramerters
w1e5 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize1, vector_size=embeddingsize5)
w1e6 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize1, vector_size=embeddingsize6)
w2e5 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize2, vector_size=embeddingsize5)
w2e6 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize2, vector_size=embeddingsize6)

print(' -- saving the trained models..')

if not os.path.exists('./task3models/'):
    os.mkdir('./task3models/')

# save models   (.model files in ./task3models)
w1e5.save('./task3models/task3-w' + str(windowsize1) +'-e' + str(embeddingsize5) + '.model')
w1e6.save('./task3models/task3-w' + str(windowsize1) +'-e' + str(embeddingsize6) + '.model')
w2e5.save('./task3models/task3-w' + str(windowsize2) +'-e' + str(embeddingsize5) + '.model')
w2e6.save('./task3models/task3-w' + str(windowsize2) +'-e' + str(embeddingsize6) + '.model')

print(' -- model training : DONE !')

print('\n -- bonus model : STARTING..')
print(' -- creating one more model using just a thesaurus..')

theseaurus_path = os.path.expanduser('thesaurus/')

preprocessed_thesaurus = []

for entry in os.listdir(theseaurus_path):
    if not entry.endswith('.txt'):
        continue
    with open('thesaurus/' + entry, 'r') as file:
        thesaurus = file.read()
        # tokenize books into sentences
        sentences_thesaurus = token.sent_tokenize(thesaurus)
        for sentence in sentences_thesaurus:
            # tokenize sentences into words
            preprocessed_thesaurus.append(token.word_tokenize(sentence))

w2e6_thesaurus = gensim.models.Word2Vec(sentences=preprocessed_thesaurus, window=windowsize2, vector_size=embeddingsize6)

w2e6_thesaurus.save('./task3models/ThesaurusBot30-300.model')

print(' -- bonus model : DONE ! :)')

print('\n <- TASK 3 : DONE ! :)')