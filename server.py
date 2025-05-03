from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from transformers import MarianMTModel, MarianTokenizer
import re
import os
import uuid
import shutil
import time
import torch

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load translation models (Hindi and Punjabi to English)
MODEL_PATHS = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "pa": "Helsinki-NLP/opus-mt-pa-en"
}
models = {}
tokenizers = {}
for lang in MODEL_PATHS:
    models[lang] = MarianMTModel.from_pretrained(MODEL_PATHS[lang])
    tokenizers[lang] = MarianTokenizer.from_pretrained(MODEL_PATHS[lang])

# Create files/ folder if it doesn't exist
FILES_DIR = "files"
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)

# Clean up old files (older than 10 minutes)
def cleanup_files():
    now = time.time()
    for filename in os.listdir(FILES_DIR):
        file_path = os.path.join(FILES_DIR, filename)
        if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > 600:
            os.remove(file_path)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def translate_text(text, src_lang):
    tokenizer = tokenizers[src_lang]
    model = models[src_lang]
    # Split text into sentences for better translation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    translated = []
    for sentence in sentences:
        if sentence.strip():
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                translated_ids = model.generate(**inputs)
            translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated.append(translated_sentence)
    return " ".join(translated)

def verify_translation(original, translated, src_lang):
    # Back-translate for verification
    back_model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{src_lang}")
    back_tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{src_lang}")
    inputs = back_tokenizer(translated, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        back_ids = back_model.generate(**inputs)
    back_translated = back_tokenizer.decode(back_ids[0], skip_special_tokens=True)
    original_terms = re.findall(r'\b(she|he|her|him|his)\b', original, re.IGNORECASE)
    back_terms = re.findall(r'\b(she|he|her|him|his)\b', back_translated, re.IGNORECASE)
    is_accurate = len(original_terms) == len(back_terms) and all(t1.lower() == t2.lower() for t1, t2 in zip(original_terms, back_terms))
    return {
        "is_accurate": is_accurate,
        "original_terms": original_terms,
        "back_translated_terms": back_terms
    }

def create_pdf(text, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
    doc.build(story)

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    cleanup_files()  # Clean up old files
    progress = []
    if "file" not in request.files or "language" not in request.form:
        return jsonify({"error": "File and language selection are required"}), 400

    file = request.files["file"]
    lang = request.form["language"]
    if lang not in ["hi", "pa"]:
        return jsonify({"error": "Invalid language selected. Choose Hindi or Punjabi."}), 400

    # Check file size (limit to 2 MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > 2 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 2 MB limit."}), 400
    file.seek(0)

    file_id = str(uuid.uuid4())
    original_path = os.path.join(FILES_DIR, f"{file_id}_original.pdf")
    translated_path = os.path.join(FILES_DIR, f"{file_id}_translated.pdf")
    file.save(original_path)
    progress.append("File uploaded successfully.")

    # Extract text
    progress.append("Extracting text from PDF...")
    original_text = extract_text_from_pdf(original_path)

    # Translate
    progress.append(f"Translating from {lang == 'hi' and 'Hindi' or 'Punjabi'} to English...")
    translated_text = translate_text(original_text, lang)

    # Verify translation
    progress.append("Verifying translation accuracy...")
    verification_result = verify_translation(original_text, translated_text, lang)
    if not verification_result["is_accurate"]:
        progress.append("Warning: Translation may have inaccuracies in pronouns.")
    else:
        progress.append("Translation verified successfully.")

    # Create translated PDF
    progress.append("Generating translated PDF...")
    create_pdf(translated_text, translated_path)

    return jsonify({
        "progress": progress,
        "download_url": f"/download/{file_id}",
        "verification_result": verification_result
    })

@app.route("/download/<file_id>", methods=["GET"])
def download_file(file_id):
    translated_path = os.path.join(FILES_DIR, f"{file_id}_translated.pdf")
    if not os.path.exists(translated_path):
        return jsonify({"error": "File not available. It may have expired due to server restart. Please upload again."}), 404

    return send_file(translated_path, as_attachment=True, download_name=f"translated_{file_id}.pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
