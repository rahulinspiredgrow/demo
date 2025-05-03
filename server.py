from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import googletrans
from googletrans import Translator
import re
import io
import sqlite3
import uuid
import os

app = Flask(__name__)
CORS(app)

translator = Translator()

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(":memory:")  # In-memory DB for Render free tier
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files (id TEXT PRIMARY KEY, content BLOB, translated_content TEXT)''')
    conn.commit()
    return conn

conn = init_db()

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
    return len(original_terms) == len(back_terms) and all(t1.lower() == t2.lower() for t1, t2 in zip(original_terms, back_terms))

def create_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route("/upload", methods=["POST"])
def upload_file():
    progress = []
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_id = str(uuid.uuid4())
    file_content = file.read()
    progress.append("File uploaded successfully.")

    # Extract text
    progress.append("Extracting text from PDF...")
    original_text = extract_text_from_pdf(io.BytesIO(file_content))

    # Detect language
    lang = "hi" if any(ord(char) >= 2304 and ord(char) <= 2431 for char in original_text) else "pa"
    progress.append(f"Detected language: {'Hindi' if lang == 'hi' else 'Punjabi'}")

    # Translate
    progress.append("Translating to English...")
    translated_text = translate_text(original_text, lang)

    # Verify translation
    progress.append("Verifying translation accuracy...")
    if not verify_translation(original_text, translated_text, lang):
        return jsonify({"error": "Translation verification failed. Possible meaning mismatch."}), 500
    progress.append("Translation verified successfully.")

    # Store in SQLite
    c = conn.cursor()
    c.execute("INSERT INTO files (id, content, translated_content) VALUES (?, ?, ?)",
              (file_id, file_content, translated_text))
    conn.commit()

    # Create translated PDF
    progress.append("Creating translated PDF...")
    pdf_buffer = create_pdf(translated_text)

    return jsonify({
        "progress": progress,
        "download_url": f"/download/{file_id}"
    })

@app.route("/download/<file_id>", methods=["GET"])
def download_file(file_id):
    c = conn.cursor()
    c.execute("SELECT translated_content FROM files WHERE id = ?", (file_id,))
    result = c.fetchone()
    if not result:
        return jsonify({"error": "File not found"}), 404

    pdf_buffer = create_pdf(result[0])
    return send_file(pdf_buffer, as_attachment=True, download_name=f"translated_{file_id}.pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)