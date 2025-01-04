from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import fitz  # PyMuPDF

app = Flask(__name__)

# Chemin vers le modèle
model_path = "/Users/nicolasdimitriu/Desktop/PFE/results/checkpoint-15/"

# Charger le modèle et le tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Configurer le dispositif (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Variable globale pour stocker le Doctype
selected_doctype = None

@app.route('/')
def index():
    return render_template('FrontEnd.html')

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    global selected_doctype  # Utiliser une variable globale pour stocker le Doctype

    if 'file' not in request.files or 'doctype' not in request.form:
        return jsonify({'message': 'Fichier ou type de document manquant'}), 400

    file = request.files['file']
    selected_doctype = request.form['doctype']  # Récupération du type de document
    print(f"Type de document sélectionné : {selected_doctype}")  # Affichage dans la console

    if not file.filename.endswith('.pdf'):
        return jsonify({'message': 'Veuillez fournir un fichier PDF'}), 400

    try:
        # Extraire le texte du PDF et effectuer l'analyse
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)

        result = "Vice de procédure détecté" if prediction.item() == 1 else "Pas de vice de procédure"
        return jsonify({'message': result})

    except Exception as e:
        return jsonify({'message': f'Erreur lors du traitement : {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
