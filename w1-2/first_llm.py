# Check if GPU is available
import warnings
warnings.filterwarnings('ignore')
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("-" * 50)


from transformers import pipeline
# Load a pre-trained model
classifier = pipeline("sentiment-analysis")

texts = [
    "I love this movie!", 
    "This is terrible.",
    "It's okay, nothing special."
]

for text in texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")

print("-" * 50)


from transformers import AutoTokenizer
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Test different texts
texts = [
    "Hello world!",
    "This is a longer sentence with more words.",
    "Supercalifragilisticexpialidocious" # Long/rare word
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    tokens_ids = tokenizer.encode(text)

    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {tokens_ids}")
    print(f"Decoded: {tokenizer.decode(tokens_ids)}")
    print("-" * 50)

import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer

# Load model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

def visualize_attention(text, layer=0, head=0):
    """Visualize attention patterns"""
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions
    
    # Get attention for specific layer and head
    attention = attentions[layer][0, head].detach().numpy()

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='Blues')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(f'Attention Pattern - Layer {layer}, Head {head}')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Test with a sentence
#visualize_attention("The cat sat on the mat")
print("-" * 50)

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_word_embedding(word):
    """Get the embedding for a single word"""
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[0, 0].numpy()

# Test semantic similarity
words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
embeddings = {word: get_word_embedding(word) for word in words}

# Calculate similarities
for word1 in words:
    for word2 in words:
        if word1 != word2:
            sim  = cosine_similarity([embeddings[word1]], [embeddings[word2]])[0][0]
            print(f"{word1} - {word2}: {sim:.3f}")



# Compare different tokenizers
tokenizers = [
'bert-base-uncased',
'gpt2',
'roberta-base'
]
text = "The quick brown fox jumps over the lazy dog!"
for model_name in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    print(f"{model_name}: {tokens}")
    print("@" * 50)

