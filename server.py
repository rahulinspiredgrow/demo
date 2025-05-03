from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from transformers import MarianTokenizer, MarianMTModel
import re
import os
import uuid
import shutil
import torch

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load translation model
MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode

# Create files/ folder if it doesn't exist
FILES_DIR = "files"
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)

def preprocess_text(text):
    """Clean input text before translation."""
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    return text

def translate_text(text, src_lang="hi"):
    """Translate text using the Hugging Face model."""
    if src_lang != "hi":  # Only Hindi supported for now
        return text  # Placeholder for Punjabi (extend later)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

def verify_translation(original, translated, src_lang):
    """Verify pronoun accuracy (simplified for demo)."""
    original_terms = re.findall(r'\b(she|he|her|him|his)\b', original, re.IGNORECASE)
    translated_terms = re.findall(r'\b(she|he|her|him|his)\b', translated, re.IGNORECASE)
    is_accurate = len(original_terms) == len(translated_terms) and all(t1.lower() == t2.lower() for t1, t2 in zip(original_terms, translated_terms))
    return {
        "is_accurate": is_accurate,
        "original_terms": original_terms,
        "translated_terms": translated_terms
    }

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return preprocess_text(text)

def create_pdf(text, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=1*inch, bottomMargin=1*inch)
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = styles["Heading1"]
    story = []

    # Structure the letter
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.2*inch))
            continue
        if line.startswith("Date:"):
            story.append(Paragraph(line, normal))
        elif line.startswith("Varisha") or line.startswith("+91") or "@" in line:
            story.append(Paragraph(line, normal))
        elif line.startswith("Dear"):
            story.append(Paragraph(line, heading))
        elif line.startswith("Sincerely") or line == "Keshav" or line == "Founder":
            story.append(Paragraph(line, normal))
        else:
            story.append(Paragraph(line, normal))
        story.append(Spacer(1, 0.1*inch))

    doc.build(story)

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    progress = []
    if "file" not in request.files or "language" not in request.form:
        return jsonify({"error": "File and language selection are required"}), 400

    file = request.files["file"]
    lang = request.form["language"]
    if lang not in ["hi", "pa"]:
        return jsonify({"error": "Invalid language selected. Choose Hindi or Punjabi."}), 400

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
