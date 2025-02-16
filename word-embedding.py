#Step 1: Install Gensim

pip install gensim 


#Step 2: Load Pre-trained glove-wiki-gigaword-50 Model(비교적 크기가 작은 모델 다운로드)

import gensim.downloader as api model = api.load('glove-wiki-gigaword-50')

#Step 3: Exploring Word Embeddings

# Get vector for a word (워드 King에 대한 벡터값을 조회해 봅니다.) 
word_vector = model['king’] 


# Find similar words (King이라는 단어와 비슷한 단어를 5개 찾아봅니다.)
similar_words = model.most_similar('king', topn=5) 
print("Similar words to 'king':", similar_words) 


# Vector arithmetic (워드에 대한 더하기 뺴기를 해 봅니다.)
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1) 
print("Result of 'king' - 'man' + 'woman':", result)
