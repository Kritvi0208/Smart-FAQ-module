from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load the FAQ data from the JSON file
with open('faqs.json') as f:
    faq_data = json.load(f)

# Prepare the data in a flat list for easy processing
faqs_list = []
for category, faqs in faq_data.items():
    for faq in faqs:
        faqs_list.append({
            'category': category,
            'question': faq['question'],
            'answer': faq['answer']
        })

# Extract the questions for TF-IDF vectorization
questions = [faq['question'] for faq in faqs_list]

# Create TF-IDF Vectorizer and fit it with FAQ questions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Function to find the most relevant FAQ
def get_relevant_faq(user_query):
    query_vec = vectorizer.transform([user_query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_faq_index = cosine_similarities.argmax()
    return faqs_list[top_faq_index]

@app.route('/')
def home():
    return "Welcome to the Smart FAQ Module!"

@app.route('/faq', methods=['POST'])
def search_faq():
    user_query = request.json.get('query')
    relevant_faq = get_relevant_faq(user_query)
    
    return jsonify({
        'category': relevant_faq['category'],
        'question': relevant_faq['question'],
        'answer': relevant_faq['answer']
    })

if __name__ == "__main__":
    app.run(debug=True)
