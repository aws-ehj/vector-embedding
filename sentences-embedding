#Creating Custom Embeddings with Sentence Transformers
#Now, let’s create custom embeddings using the sentence-transformers library.

#Step 1: Install Sentence Transformers
pip install sentence-transformers 

#Step 2: Generate Sentence Embeddings
from sentence_transformers import SentenceTransformer 
# Load pre-trained model 
model = SentenceTransformer('all-MiniLM-L6-v2’) 

# Encode 50 sentences 
sentences = ["The ripe mango's sweet aroma filled the kitchen.", "Skyscrapers dominated the city's skyline.", "She laughed heartily at her friend's joke.", "Strawberries are often considered a summer fruit.", "The old courthouse stood as a testament to history.", "His eyes sparkled with excitement.", "Bananas are an excellent source of potassium.", "The architect designed an eco-friendly office building.", "Children played happily in the park.", "Apples come in a variety of colors and flavors.", "The museum's modern design attracted many visitors.", "She pursued her dreams with unwavering determination.", "Pineapples have a distinctive spiky exterior.", "The apartment complex housed hundreds of families.", "He struggled to overcome his fear of public speaking.", "Grapes can be eaten fresh or used to make wine.", "The cathedral's stained glass windows were breathtaking.", "Teenagers often experience mood swings during puberty.", "Kiwis are small fruits with fuzzy brown skin.", "The factory's smokestack loomed over the town.", "She volunteered at the local animal shelter every weekend.", "Oranges are known for their high vitamin C content.", "The library's quiet atmosphere was perfect for studying.", "He practiced meditation to reduce stress.", "Lemons add a tart flavor to many dishes.", "The shopping mall was bustling with activity.", "Parents worried about their children's screen time.", "Watermelons are a refreshing treat on hot summer days.", "The elderly couple held hands as they walked.", "Blueberries are often called a superfood.", "The teacher patiently explained the concept to her students.", "Pomegranates have numerous edible seeds inside.", "The hotel's luxurious amenities impressed its guests.", "He struggled with insomnia for years.", "Raspberries have a delicate texture and sweet-tart taste.", "Athletes trained rigorously for the upcoming competition.", "Coconuts provide both meat and water.", "The bride walked gracefully down the aisle.", "Peaches have soft, fuzzy skin and juicy flesh.", "The artist's creativity knew no bounds.", "Cherries are often used in desserts and cocktails.", "The surgeon performed a life-saving operation.", "Limes are frequently used in Mexican cuisine.", "She felt a sense of accomplishment after finishing the marathon.", "Blackberries grow wild in many parts of the world.", "The politician delivered a rousing speech to his supporters.", "Figs have a unique texture and sweet flavor.", "The firefighter bravely entered the burning building.", "Passion fruit has a tart, tropical taste.", "He proposed to his girlfriend on a moonlit beach."]

embeddings = model.encode(sentences) 

print("Sentence Embeddings:")
for sentence, embedding in zip(sentences, embeddings): 
  print(f"Sentence: {sentence}\nEmbedding: {embedding[:5]}...\n")

#To visualize high-dimensional embeddings, we can use t-SNE.


#Step 1: Install t-SNE
pip install scikit-learn matplotlib

#Step 2: Visualize Embeddings
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 

# Reduce dimensions 
tsne = TSNE(n_components=3, random_state=0, perplexity=30)
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot 
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1]) 
for i, sentence in enumerate(sentences): 
	plt.annotate(sentence, (reduced_embeddings[i, 0], reduced_embeddings[i, 1])) 

plt.show() 

