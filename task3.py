
# COMP472A2 TASK3
# -- written by Chris Anglin -- 40216346

import os
import gensim
import nltk.tokenize as token

print('\n\n -> RUNNING TASK 3..')

print('\n -- data preprocessing : STARTING')
print(' -- preprocessing books into 2D array..')

# import and preprocess my 7 chosen books
path = os.path.expanduser('books/')

preprocessed = []

for entry in os.listdir(path):
    if not entry.endswith('.txt'):
        continue
    with open('books/' + entry, 'r') as file:
        book = file.read()
        sentences = token.sent_tokenize(book)
        for sentence in sentences:
            preprocessed.append(token.word_tokenize(sentence))

print(' -- data preprocessing : DONE !')

print('\n -- model training : STARTING')
print(' -- creating the four new word2vec models..')

# chosen params for models (experimenting pretty much)
windowsize1 = 13
windowsize2 = 37
embeddingsize5 = 99
embeddingsize6 = 373

# create 4 word2vec models using the processed data from books and the above paramerters
w1e5 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize1, vector_size=embeddingsize5)
w1e6 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize1, vector_size=embeddingsize6)
w2e5 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize2, vector_size=embeddingsize5)
w2e6 = gensim.models.Word2Vec(sentences=preprocessed, window=windowsize2, vector_size=embeddingsize6)

print(' -- saving the trained models..')

if not os.path.exists('./task3models/'):
    os.mkdir('./task3models/')

# save models
w1e5.save('./task3models/task3-w' + str(windowsize1) +'-e' + str(embeddingsize5) + '.model')
w1e6.save('./task3models/task3-w' + str(windowsize1) +'-e' + str(embeddingsize6) + '.model')
w2e5.save('./task3models/task3-w' + str(windowsize2) +'-e' + str(embeddingsize5) + '.model')
w2e6.save('./task3models/task3-w' + str(windowsize2) +'-e' + str(embeddingsize6) + '.model')

print(' -- model training : DONE !')

print('\n <- TASK 3 : DONE ! :)')