from flask import Flask, render_template, request
from transformers import pipeline
from urllib.parse import urlparse
import fitz  # PyMuPDF
from newspaper import Article
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Hardcoded reputation scores
news_reputation = {
    "bbc.com": "High",
    "cnn.com": "Medium",
    "foxnews.com": "Medium",
    "nytimes.com": "High",
    "theonion.com": "Low",
    "infowars.com": "Low",
    "reuters.com": "High"
}

# Load models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Persona labels
persona_labels = {
    "neutral": "🤖 Neutral AI",
    "professor": "🎓 Media Professor",
    "detective": "🕵️ Fact Detective"
}

# Predict bias
def predict_bias(text):
    result = classifier(text, candidate_labels=["Left", "Center", "Right"])
    return result['labels'][0], round(result['scores'][0] * 100, 2)

# Sensationalism score
def sensationalism_score(text):
    # Truncate to first 512 tokens (approx. ~2000 characters)
    truncated = text[:2000] if len(text) > 2000 else text
    result = sentiment(truncated)[0]
    score = result['score'] * 100
    if result['label'] == "NEGATIVE":
        score *= 1.25
    return round(min(score, 100), 2)


# Generate summary with persona
def generate_summary(text, persona):
    result = summarizer(text, max_length=100, min_length=25, do_sample=False)
    summary = result[0]['summary_text']
    if persona == "professor":
        summary = f"As a professor, I'd explain it this way: {summary}"
    elif persona == "detective":
        summary = f"Upon inspection, this is the essence: {summary}"
    return summary

# Source credibility checker
def check_source_reputation(text):
    try:
        for word in text.split():
            if word.startswith("http"):
                domain = urlparse(word).netloc.replace("www.", "")
                return news_reputation.get(domain, "Unknown")
    except:
        pass
    return "Unknown"

# Extract article from URL using newspaper3k
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"❌ Failed to extract article: {e}"

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"❌ Failed to extract PDF: {e}"

# Chat route
@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        persona = request.form.get("persona", "neutral")
        text = ""

        # Case 1: User pasted a URL
        if "url" in request.form and request.form["url"]:
            text = extract_text_from_url(request.form["url"])

        # Case 2: User uploaded a PDF
        elif "pdf" in request.files and request.files["pdf"]:
            pdf_file = request.files["pdf"]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(file_path)
            text = extract_text_from_pdf(file_path)
            os.remove(file_path)

        # Case 3: User typed or pasted raw text
        elif "text" in request.form and request.form["text"]:
            text = request.form["text"]

        if not text.strip():
            return render_template("chat.html", user_input="❌ No content to analyze.")

        bias, confidence = predict_bias(text)
        sensational = sensationalism_score(text)
        summary = generate_summary(text, persona)
        reputation = check_source_reputation(text)

        return render_template("chat.html",
                               user_input=text,
                               bias=bias,
                               confidence=f"{confidence}%",
                               sensational=sensational,
                               summary=summary,
                               reputation=reputation,
                               persona_label=persona_labels.get(persona, "🤖 Neutral AI"))
    return render_template("chat.html")


if __name__ == "__main__":
    app.run(debug=True)

