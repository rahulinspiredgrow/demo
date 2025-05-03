from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import googletrans
from googletrans import Translator
import re
import io
import os
import uuid
import shutil

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

translator = Translator()

# Create files/ folder if it doesn't exist
FILES_DIR = "files"
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def translate_text(text, src_lang, dest_lang="en"):
    chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
    translated = ""
    for chunk in chunks:
        result = translator.translate(chunk, src=src_lang, dest=dest_lang)
        translated += result.text
    return translated

def verify_translation(original, translated, src_lang):
    back_translated = translator.translate(translated, src="en", dest=src_lang).text
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
