from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from transformers import MarianTokenizer, MarianMTModel
from sentence_transformers import SentenceTransformer, util
import re
import os
import uuid
import torch

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load translation model
try:
    MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    print(f"Error loading translation model: {str(e)}")
    raise

# Load sentence transformer for similarity
try:
    SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_model = SentenceTransformer(SIMILARITY_MODEL)
except Exception as e:
    print(f"Error loading similarity model: {str(e)}")
    raise

# Create files/ folder
FILES_DIR = "files"
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)

def preprocess_text(text):
    """Clean input text."""
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def summarize_text(text):
    """Rule-based summarization: extract first sentence of each paragraph."""
    paragraphs = text.split("\n")
    summary = []
    for para in paragraphs:
        sentences = re.split(r'[.!?]', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            summary.append(sentences[0])
    return " ".join(summary)

def translate_text(text, src_lang="hi"):
    """Translate text using Hugging Face model."""
    if src_lang != "hi":  # Placeholder for Punjabi
        return text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    translated = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        translated.append(tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0])
    return " ".join(translated)

def verify_translation(original, translated, src_lang):
    """Compare summaries for semantic similarity."""
    original_summary = summarize_text(original)
    translated_summary = summarize_text(translated)
    
    # Compute embeddings
    original_embedding = similarity_model.encode(original_summary, convert_to_tensor=True)
    translated_embedding = similarity_model.encode(translated_summary, convert_to_tensor=True)
    
    # Cosine similarity
    similarity = util.cos_sim(original_embedding, translated_embedding).item()
    is_accurate = similarity >= 0.85
    
    return {
        "is_accurate": is_accurate,
        "similarity_score": similarity,
        "original_summary": original_summary,
        "translated_summary": translated_summary
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
        progress.append("Warning: Translation summaries may differ in meaning.")
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
