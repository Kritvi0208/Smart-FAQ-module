from flask import Flask, request, jsonify, render_template_string
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

# Home route (GET) to display the form
@app.route('/')
def home():
    return render_template_string('''
    <form action="/faq" method="post">
        <label for="query">Enter your query:</label>
        <input type="text" id="query" name="query">
        <input type="submit" value="Submit">
    </form>
    ''')

# FAQ route (POST) to handle queries
@app.route('/faq', methods=['POST'])
def search_faq():
    user_query = request.form.get('query')  # Get the query from form data
    relevant_faq = get_relevant_faq(user_query)

    # Format the output using HTML
    formatted_response = f"""
    <h3>Category: {relevant_faq['category']}</h3>
    <p><strong>Question:</strong> {relevant_faq['question']}</p>
    <p><strong>Answer:</strong> {relevant_faq['answer']}</p>
    """

    return formatted_response  # Return the HTML response

if __name__ == "__main__":
    app.run(debug=True)
