from flask import Flask, render_template, request
from transformers import pipeline
from urllib.parse import urlparse
import fitz  # PyMuPDF
from newspaper import Article
import os
import re
from transformers import pipeline
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Reputation scores
news_reputation = {
    "thehindu.com": "High",           
    "thehindubusinessline.com": "Medium",  
    "indiatoday.in": "Medium",       
    "opindia.com": "Medium",         
    "news18.com": "Low",            
    "ltn.com.tw": "Medium",      
    "udn.com": "Medium",         
    "mingpao.com.hk": "High",        
    "hkfp.com": "Medium",             
    "scmp.com": "High",             
    "inmediahk.net": "High",          
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
    "professor": "📘 Explains News Clearly",
    "detective": "🔍 Checks Facts Deeply"
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
def generate_summary(text, persona, reputation=None):
    result = summarizer(text, max_length=100, min_length=25, do_sample=False)
    summary = result[0]['summary_text']

    if persona == "professor":
        summary = simplify_text(summary)
        summary = f"As a professor, let me break it down simply:\n\n{summary}"

    elif persona == "detective":
        summary = highlight_keywords(summary)
        summary = f"After close inspection, here’s the essence:\n\n{summary}"

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

# Extracts article from URL using newspaper3k
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"❌ Failed to extract article: {e}"
    
def simplify_text(text):
    replacements = {
    "e.g.": "for example",
    "i.e.": "that is",
    "significant": "important",
    "utilize": "use",
    "utilization": "use",
    "approximately": "about",
    "individuals": "people",
    "commence": "start",
    "terminate": "end",
    "prior to": "before",
    "subsequent to": "after",
    "assistance": "help",
    "inquire": "ask",
    "purchase": "buy",
    "numerous": "many",
    "demonstrate": "show",
    "facilitate": "make easier",
    "sufficient": "enough",
    "insufficient": "not enough",
    "endeavor": "try",
    "indicate": "show",
    "initiate": "start",
    "conclude": "end",
    "obtain": "get",
    "objective": "goal",
    "predominantly": "mostly",
    "methodology": "method",
    "component": "part",
    "substantial": "large",
    "comprehend": "understand",
    "articulate": "explain clearly",
    "analyze": "examine",
    "interpret": "explain",
    "validate": "confirm",
    "necessitate": "require",
    "implement": "put into action",
    "acknowledge": "admit",
    "proficient": "skilled",
    "demographic": "group",
    "controversial": "disputed",
    "authentic": "real",
    "acquire": "get",
    "allocate": "give",
    "consume": "use",
    "determine": "decide",
    "evaluate": "judge",
    "implication": "consequence",
    "mandatory": "required",
    "voluntary": "optional",
    "compile": "gather",
    "disseminate": "share",
    "fluctuate": "change",
    "impede": "slow down",
    "inhibit": "prevent",
    "mitigate": "reduce",
    "manifest": "show",
    "convey": "express",
    "pertain": "relate to",
    "accompany": "go with",
    "relevant": "related",
    "coincide": "happen at the same time",
    "reside": "live",
    "subsequently": "later",
    "emerge": "come out",
    "perspective": "view",
    "reiterate": "repeat",
    "viable": "possible",
    "optimize": "improve",
    "constitute": "make up",
    "adverse": "harmful",
    "approximate": "about",
    "indigenous": "native",
    "prevalent": "common",
    "synthesize": "combine",
    "advocate": "support",
    "detrimental": "harmful",
    "eradicate": "remove",
    "alleviate": "ease",
    "feasible": "possible",
    "paradigm": "model",
    "parameter": "limit",
    "contemporary": "modern",
    "aggregate": "total",
    "stipulate": "require",
    "ameliorate": "improve",
    "collaborate": "work together",
    "assert": "claim",
    "accomplish": "achieve",
    "artificial": "man-made"
}

    for word, simple in replacements.items():
        text = re.sub(rf"\b{word}\b", simple, text, flags=re.IGNORECASE)
    
    return text

def highlight_keywords(text):
    words = text.split()
    important = []

    for word in words:
        clean = word.lower().strip(".,")
        if clean not in stop_words and len(clean) > 5:
            important.append(clean)

    keywords = list(set(important[:6]))  
    highlighted = text

    for kw in keywords:
        highlighted = re.sub(rf"\b({kw})\b", r"**\1**", highlighted, flags=re.IGNORECASE)

    return highlighted


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
        reputation = check_source_reputation(text)
        summary = generate_summary(text, persona, reputation if persona == "detective" else None)


        return render_template("chat.html",
                               user_input=text,
                               bias=bias,
                               confidence=f"{confidence}%",
                               sensational=sensational,
                               summary=summary,
                               reputation=reputation if persona == "detective" else None,
                               persona_label=persona_labels.get(persona, "🤖 Neutral AI"))
    return render_template("chat.html")


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    app.run(debug=True, host="0.0.0.0", port=port)


