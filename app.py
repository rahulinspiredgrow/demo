from flask import Flask, request, jsonify, send_file, render_template
from pdfminer.high_level import extract_text as pdfminer_extract_text
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from googletrans import Translator  # Consider more advanced options for production
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from transformers import pipeline  # More advanced translation and summarization
import os
import time  # For basic performance monitoring

app = Flask(__name__)
translator = Translator()
# Advanced translation model (requires significant resources and setup)
try:
    nmt_pipeline = pipeline('translation', model='Helsinki-NLP/opus-mt-hi-en')
except Exception as e:
    print(f"Warning: Failed to load advanced translation model: {e}")
    nmt_pipeline = None

# Advanced summarization model (requires significant resources and setup)
try:
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Warning: Failed to load advanced summarization model: {e}")
    summarization_pipeline = None

# Configuration options
TEMP_PDF_PREFIX = "temp_"
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30

def time_it(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

@time_it
def extract_text_from_pdf(pdf_path):
    """Attempts to extract text directly from PDF, then uses OCR if needed."""
    text = ""
    try:
        text = pdfminer_extract_text(pdf_path)
        if text.strip():
            print("Successfully extracted text directly from PDF.")
            return text
        else:
            print("Direct text extraction failed, attempting OCR.")
            images = convert_from_path(pdf_path)
            ocr_text = ""
            for i, img in enumerate(images):
                print(f"Performing OCR on page {i+1}/{len(images)}")
                ocr_text += pytesseract.image_to_string(img, lang='hin+eng') + "\n"
            if ocr_text.strip():
                print("OCR successful.")
                return ocr_text
            else:
                print("OCR also failed to extract significant text.")
                return ""
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        try:
            print("Attempting OCR as a fallback due to initial error.")
            images = convert_from_path(pdf_path)
            ocr_text = ""
            for i, img in enumerate(images):
                print(f"Performing fallback OCR on page {i+1}/{len(images)}")
                ocr_text += pytesseract.image_to_string(img, lang='hin+eng') + "\n"
            print("Fallback OCR completed.")
            return ocr_text
        except Exception as ocr_e:
            print(f"Fallback OCR Error: {ocr_e}")
            return ""

@time_it
def translate_text(text, target_language='en'):
    """Translates text using a more advanced model if available, otherwise falls back to googletrans."""
    if nmt_pipeline:
        try:
            print("Using advanced translation model.")
            result = nmt_pipeline(text, max_length=512)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Error with advanced translation model: {e}. Falling back to googletrans.")
            return _fallback_translate(text, target_language)
    else:
        return _fallback_translate(text, target_language)

def _fallback_translate(text, target_language='en'):
    """Fallback translation using googletrans."""
    try:
        print("Using googletrans for translation.")
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        print(f"Translation error (googletrans): {e}")
        return ""

@time_it
def generate_summary(text, language='en'):
    """Generates a summary using a more advanced model if available, otherwise falls back to a basic approach."""
    if summarization_pipeline:
        try:
            print("Using advanced summarization model.")
            summary = summarization_pipeline(text, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error with advanced summarization model: {e}. Falling back to basic summarization.")
            return _fallback_summary(text)
    else:
        return _fallback_summary(text)

def _fallback_summary(text):
    """Basic summarization by taking the first few sentences."""
    sentences = text.split('.')
    if len(sentences) > 2:
        return '. '.join(sentences[:3]) + '...'
    return text

@time_it
def create_pdf(text):
    """Generates a PDF from the given text with better formatting."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    textobject = p.beginText(30, 750)
    textobject.setFont("Helvetica", 12)
    lines = text.split('\n')
    for line in lines:
        textobject.textLine(line.strip())
        if textobject.getY() < 50:  # Start a new page if running out of space
            p.drawText(textobject)
            p.showPage()
            textobject = p.beginText(30, 750)
            textobject.setFont("Helvetica", 12)
    p.drawText(textobject)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
@time_it
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        temp_pdf_path = f"{TEMP_PDF_PREFIX}{time.time()}.pdf"
        try:
            file.save(temp_pdf_path)
            hindi_text = extract_text_from_pdf(temp_pdf_path)
            english_text = translate_text(hindi_text, 'en')
            hindi_summary = generate_summary(hindi_text, 'hi')
            english_summary = generate_summary(english_text, 'en')
            return jsonify({
                'hindi_text': hindi_text,
                'english_text': english_text,
                'hindi_summary': hindi_summary,
                'english_summary': english_summary
            })
        except Exception as e:
            print(f"Error processing PDF request: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/generate_pdf', methods=['POST'])
@time_it
def generate_pdf_route():
    data = request.get_json()
    english_text = data.get('text')
    if english_text:
        pdf_buffer = create_pdf(english_text)
        return send_file(pdf_buffer, download_name='translated_document.pdf', as_attachment=True, mimetype='application/pdf')
    return jsonify({'error': 'No text provided for PDF generation'}), 400

if __name__ == '__main__':
    app.run(debug=True)
