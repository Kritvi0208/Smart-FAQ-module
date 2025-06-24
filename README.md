# 🔎 Smart FAQ System using TF-IDF and NLP

A lightweight, intelligent FAQ system built using **Flask** and **scikit-learn** that dynamically responds to user queries by identifying the most relevant frequently asked question using **TF-IDF vectorization** and **cosine similarity**.

---

## 📌 Project Overview

This project simulates how natural language processing (NLP) techniques can be applied to improve customer support automation. Users enter queries through a web interface, and the system finds the best matching answer from a structured set of FAQs.

> Designed for hackathons, technical evaluations, and educational use cases where FAQ automation or NLP-based response systems are needed.

---

## 💡 Key Highlights

- ✅ Flask-based REST interface with clean HTML/CSS frontend  
- ✅ Intelligent query resolution using TF-IDF + Cosine Similarity  
- ✅ JSON-based modular FAQ data handling  
- ✅ Error handling for edge cases (empty input, unmatched queries)  
- ✅ Modular and extendable architecture

---

## 🛠 Tech Stack

| Layer         | Tech Used                         |
|--------------|-----------------------------------|
| Backend       | Python, Flask                     |
| NLP & ML      | scikit-learn (TF-IDF, cosine sim) |
| Data Format   | JSON                              |
| Frontend      | HTML5, CSS                        |
| Deployment    | Localhost (can be extended)       |

---

## 📂 File Structure

```

Smart-FAQ-module/
├── app.py                 # Main Flask app
├── faqs.json              # FAQ data in JSON format
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

---

## 🚀 How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/Kritvi0208/Smart-FAQ-module.git
cd Smart-FAQ-module
````

2. **Set up virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Flask app**

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

---

## 🧠 How It Works

* User inputs a natural language query
* The system transforms the input using **TF-IDF**
* Cosine similarity is calculated against all predefined questions
* The most relevant question is returned along with its answer and category

---

## 🧪 Sample Query & Response

**User Query:** *“Can I return a product after a month?”*
**Matched FAQ:**

* **Category:** Returns
* **Question:** What is your return policy?
* **Answer:** You can return any item within 30 days of purchase.

---

## 🎯 Ideal For

* NLP-based search prototypes
* Chatbot backend modules
* Customer support FAQ systems
* Educational demonstrations of TF-IDF

---

## 🛠 Future Enhancements

* [ ] Add minimum similarity threshold to reject irrelevant queries
* [ ] Integrate with a chatbot interface (e.g., Telegram, Slack)
* [ ] Add support for multilingual queries
* [ ] Host on a cloud platform with persistent database


---
Built by **Ritvika**

GitHub: [@Kritvi0208](https://github.com/Kritvi0208)
Project: [Smart FAQ Module](https://github.com/Kritvi0208/Smart-FAQ-module)

---
Licensed under the MIT License.
