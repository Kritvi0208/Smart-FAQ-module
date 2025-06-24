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

# Home route (GET) to display the form with CSS
@app.route('/')
def home():
    return render_template_string('''
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; }
            h1 { color: #333; text-align: center; }
            form { background-color: white; padding: 20px; border-radius: 8px; max-width: 500px; margin: 50px auto; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
            input[type="text"] { width: 100%; padding: 10px; margin-top: 10px; border: 1px solid #ccc; border-radius: 4px; }
            input[type="submit"] { margin-top: 20px; padding: 10px 15px; border: none; background-color: #007BFF; color: white; cursor: pointer; border-radius: 4px; }
            input[type="submit"]:hover { background-color: #0056b3; }
            .error { color: red; text-align: center; }
            .faq-result { text-align: center; font-size: 1.2em; }
        </style>
    </head>
    <body>
        <h1>Smart FAQ Module</h1>
        <form action="/faq" method="post">
            <label for="query">Enter your query:</label>
            <input type="text" id="query" name="query" placeholder="Ask a question">
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    ''')

# FAQ route (POST) to handle queries with error handling
@app.route('/faq', methods=['POST'])
def search_faq():
    user_query = request.form.get('query')  # Get the query from form data

    # Error handling for empty queries
    if not user_query.strip():
        return render_template_string('''
        <html>
        <head><style>.error { color: red; text-align: center; }</style></head>
        <body>
            <p class="error">Error: Query cannot be empty. Please go back and enter a valid question.</p>
            <a href="/">Go Back</a>
        </body>
        </html>
        ''')

    # Get the most relevant FAQ
    relevant_faq = get_relevant_faq(user_query)

    # If no FAQ matches (e.g., cosine similarity returns a low score)
    if cosine_similarity(vectorizer.transform([user_query]), tfidf_matrix).max() == 0:
        return render_template_string('''
        <html>
        <head><style>.error { color: red; text-align: center; }</style></head>
        <body>
            <p class="error">Error: No matching FAQ found. Please try again with a different query.</p>
            <a href="/">Go Back</a>
        </body>
        </html>
        ''')

    # Format the output using HTML
    formatted_response = f"""
    <html>
    <body>
        <h3 class="faq-result">Category: {relevant_faq['category']}</h3>
        <p class="faq-result"><strong>Question:</strong> {relevant_faq['question']}</p>
        <p class="faq-result"><strong>Answer:</strong> {relevant_faq['answer']}</p>
        <a href='/'>Ask another question</a>
    </body>
    </html>
    """

    return formatted_response  # Return the HTML response

if __name__ == "__main__":
    app.run(debug=True)
