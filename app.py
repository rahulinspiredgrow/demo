from flask import Flask, render_template, request, send_file
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import transformers
from transformers import MarianMTModel, MarianTokenizer
import pdfkit
import spacy
from summarizer import Summarizer

app = Flask(__name__)

# Initialize the translation model
model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize summarizer
bert_model = Summarizer()
nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_path):
    text = ""
    # Try direct PDF text extraction
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
    except:
        # If direct extraction fails, use OCR
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image, lang='hin+eng')
    return text

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def generate_summary(text, ratio=0.3):
    summary = bert_model(text, ratio=ratio)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'
    
    # Save uploaded file
    temp_path = 'temp.pdf'
    file.save(temp_path)
    
    # Extract text
    original_text = extract_text_from_pdf(temp_path)
    
    # Translate
    translated_text = translate_text(original_text)
    
    # Generate summaries
    hindi_summary = generate_summary(original_text)
    english_summary = generate_summary(translated_text)
    
    # Create result PDF
    html_content = f"""
    <h1>Translation Results</h1>
    <h2>Original Text (Hindi)</h2>
    <p>{original_text}</p>
    <h2>Translated Text (English)</h2>
    <p>{translated_text}</p>
    <h2>Summaries</h2>
    <h3>Hindi Summary</h3>
    <p>{hindi_summary}</p>
    <h3>English Summary</h3>
    <p>{english_summary}</p>
    """
    
    pdfkit.from_string(html_content, 'result.pdf')
    
    return render_template('result.html',
                         original_text=original_text,
                         translated_text=translated_text,
                         hindi_summary=hindi_summary,
                         english_summary=english_summary)

if __name__ == '__main__':
    app.run(debug=True)
