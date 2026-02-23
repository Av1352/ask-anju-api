from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-3-flash-preview')

# Load knowledge base
with open('data/anju_knowledge.md', 'r', encoding='utf-8') as f:
    KNOWLEDGE_BASE = f.read()

def ask_anju(question):
    """Ask question using Gemini with knowledge base context"""
    
    prompt = f"""You are answering questions on behalf of Anju Vilashni Nandhakumar.

CRITICAL: Always respond in FIRST PERSON as Anju speaking directly.
Use "I built..." not "Anju built..."

Tone: Professional, conversational, direct, calm confidence.

Use this knowledge base to answer:

{KNOWLEDGE_BASE}

Question: {question}

Answer (as Anju, first person):"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        answer = ask_anju(question)
        
        return jsonify({
            'answer': answer,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)