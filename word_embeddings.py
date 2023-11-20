# pip install gensim
# Run this in the terminal

# pip install matplotlib
# Run this in the terminal to install matplotlib for drawing the barcharts (used for analysing the data)

# pip show gensim
# Use it to make sure that you have downloaded Gensim library properly

import gensim.downloader as api
import json
import random
import os
import matplotlib.pyplot as plt


folder_name = 'reports'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open('data_set/synonym.json', 'r') as file:
  data_set = json.load(file)

reports_file_path = os.path.join(folder_name, 'analysis.csv')
with open(reports_file_path, 'w') as file:
  file.write('Model Name,Size of Vocabulary,#Correct Labels,Model Accuracy\n')


model_names = [
    "word2vec-google-news-300",
    "glove-wiki-gigaword-300",
    "fasttext-wiki-news-subwords-300",
    "glove-twitter-50",
    "glove-twitter-25"
]

# models = {}

# for model_name in model_names:
#     try:
#         # Attempt to load the model from the gensim cache
#         models[model_name] = api.load(model_name, return_path=False)
#         print(f'{model_name} is already downloaded.')
#     except ValueError:
#         # Model not found in cache, download it
#         models[model_name] = api.load(model_name)
#         print(f'{model_name} has been successfully downloaded')

models = {
    "word2vec-google-news-300": None, 
    "glove-wiki-gigaword-300": None,  # trained on Wikipedia + Gigaword.
    "fasttext-wiki-news-subwords-300": None,  # trained on Common Crawl.
    "glove-twitter-50": None,  # trained on a dataset composed of tweets. 
    "glove-twitter-25": None  # trained on a dataset composed of tweets. 
}

for model_name in models.keys():
  models[model_name] = api.load(model_name)
  print(f'{model_name} has been successfylly downloaded')


model_stats = {}
for model_name, model in models.items():
  file_path = os.path.join(folder_name, f'{model_name}-details.csv')
  with open(file_path, 'w') as file:
    file.write("#,question_word,correct_answer_word,system_guess_word,label\n")
    i = 1
    correct_guesses_counter = 0
    wrong_guesses_counter = 0

    for entry in data_set:
      question_word = entry["question"]
      choices = entry["choices"]
      correct_answer_word = entry["answer"]
      label = "guess"
      similarities = {}

      if question_word in model:
        for choice in choices:
          if choice in model:
            similarities[choice] = model.similarity(question_word, choice)
        if  len(similarities) > 0:
          system_guess_word = max(similarities, key=similarities.get)
          if system_guess_word == correct_answer_word:
            label = 'correct'
            correct_guesses_counter += 1
          else:
            label = 'wrong'
            wrong_guesses_counter += 1

      if label == 'guess':
        system_guess_word = random.choice(choices)

      file.write(f"{i},{question_word},{correct_answer_word},{system_guess_word},{label}\n")
      i+=1
  print(f'{model_name}-details.csv has been sucessfully generated')


  vocab_size = len(model.key_to_index)
  model_accuracy = 0
  if (correct_guesses_counter + wrong_guesses_counter) > 0:
    model_accuracy = correct_guesses_counter / (correct_guesses_counter + wrong_guesses_counter) 

  model_stats[model_name] = {"accuracy": model_accuracy, "total_correct_words": correct_guesses_counter, "total_model_guesses": correct_guesses_counter + wrong_guesses_counter} # for analyzing the data later

  with open(reports_file_path, 'a') as file:
    file.write(f'{model_name},{vocab_size},{correct_guesses_counter},{model_accuracy*100:.1f}%\n')

print(f'The analysis.csv has been sucessfully generated')


model_names = list(model_stats.keys())
accuracy = [model_stats[model]['accuracy'] for model in model_names]
total_correct_words = [model_stats[model]['total_correct_words'] for model in model_names]
total_model_guesses = [model_stats[model]['total_model_guesses'] for model in model_names]

plt.figure(figsize=(14, 6))

# Accuracy Bar Chart
plt.subplot(1, 3, 1)
plt.bar(model_names, accuracy, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)

# Total Model Guesses Bar Chart
plt.subplot(1, 3, 2)
plt.bar(model_names, total_correct_words, color='lightgreen')
plt.title('Total Model Correct Words Comparison')
plt.xlabel('Model')
plt.ylabel('Total Correct Words')
plt.xticks(rotation=90)

# Total Model Guesses Bar Chart
plt.subplot(1, 3,3)
plt.bar(model_names, total_model_guesses, color='salmon')
plt.title('Total Model Guesses Comparison')
plt.xlabel('Model')
plt.ylabel('Total Model Guesses')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('model_comparison_charts.png')
plt.show()