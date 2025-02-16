#Step 1: Install Gensim

pip install gensim 


#Step 2: Load Pre-trained Word2Vec Model
#We’ll use the pre-trained Word2Vec model from Google News.

import gensim.downloader as api model = api.load('glove-wiki-gigaword-50')

#Step 3: Exploring Word Embeddings

#Let’s find the vector for a word and perform some operations.
# Get vector for a word 
word_vector = model['king’] 


# Find similar words 
similar_words = model.most_similar('king', topn=5) 
print("Similar words to 'king':", similar_words) 


# Vector arithmetic 
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1) 
print("Result of 'king' - 'man' + 'woman':", result)
