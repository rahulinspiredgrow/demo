FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    wkhtmltopdf

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download spacy model
RUN python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["python", "app.py"]
