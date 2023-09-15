import os
from flask import Flask, request, jsonify
import pdfplumber
import spacy
from spacy.util import is_package
from io import BytesIO
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check if the language model is downloaded
if not is_package('en_core_web_sm'):
    try:
        spacy.cli.download('en_core_web_sm')
    except Exception as e:
        logging.error(f"Failed to download language model: {e}")
        raise

nlp = spacy.load("en_core_web_sm")

def extract_information(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables to store extracted information
    name = ""
    credit_score = ""
    open_accounts = ""
    accounts_ever_late = ""

    # Find the entity with the label "PERSON"
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            name = entity.text
            break

    # Extract Credit Score
    for sent in doc.sents:
        if "FICO" in sent.text and "Score" in sent.text:
            credit_score = sent.text.split("Score")[-1].strip()
            break

    # Extract Open Accounts
    for sent in doc.sents:
        if "Open accounts" in sent.text:
            open_accounts = sent.text.split(":")[-1].strip()
            break

    # Extract Accounts Ever Late
    for sent in doc.sents:
        if "Accounts ever late" in sent.text:
            accounts_ever_late = sent.text.split(":")[-1].strip()
            break

    # Return the extracted information as a dictionary
    return {
        "name": name,
        "credit_score": credit_score,
        "open_accounts": open_accounts,
        "accounts_ever_late": accounts_ever_late
    }

@app.route("/extract_and_summarize", methods=["POST"])
def extract_and_summarize():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser might submit an empty file part without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read the PDF content from the uploaded file
    pdf_content = file.read()
    
    text = ""
    try:
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        return jsonify({"error": f"Failed to extract text from PDF: {e}"}), 400

    info = extract_information(text)
    return jsonify(info)

if __name__ == "__main__":
    app.run(debug=False, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', 5000)))
