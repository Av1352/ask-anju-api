from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import re

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-3-flash-preview')

# Load and chunk knowledge base
with open('data/anju_knowledge.md', 'r', encoding='utf-8') as f:
    FULL_KB = f.read()

# Pre-chunk by sections
CHUNKS = {
    'projects': FULL_KB[FULL_KB.find('## Flagship Projects'):FULL_KB.find('## Published Research')],
    'skills': FULL_KB[FULL_KB.find('## Technical Skills'):FULL_KB.find('## Flagship Projects')],
    'experience': FULL_KB[FULL_KB.find('## Work Experience'):FULL_KB.find('## Education')],
    'education': FULL_KB[FULL_KB.find('## Education'):FULL_KB.find('## Why I Focus')],
    'bio': FULL_KB[:FULL_KB.find('## What Sets Me Apart')],
    'demos': FULL_KB[FULL_KB.find('## What Sets Me Apart'):FULL_KB.find('## Technical Skills')],
    'visa': FULL_KB[FULL_KB.find('Work Authorization'):FULL_KB.find('**Availability:**')] + '\n\n2+ years work authorization, no sponsorship cost.',
    'full': FULL_KB[:2000]  # First 2000 chars for general questions
}

def get_relevant_context(question):
    """Get relevant knowledge base sections based on question keywords"""
    q_lower = question.lower()
    
    # Keyword matching
    if any(word in q_lower for word in ['project', 'build', 'built', 'system', 'deploy']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['projects']
    elif any(word in q_lower for word in ['skill', 'technical', 'technology', 'framework', 'tool']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['skills']
    elif any(word in q_lower for word in ['experience', 'work', 'job', 'role', 'position']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['experience']
    elif any(word in q_lower for word in ['education', 'degree', 'school', 'university', 'northeastern']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['education']
    elif any(word in q_lower for word in ['visa', 'sponsor', 'authorization', 'opt', 'h1b']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['visa']
    elif any(word in q_lower for word in ['demo', 'custom', 'different', 'unique', 'standout']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['demos']
    elif any(word in q_lower for word in ['healthcare', 'medical', 'clinical', 'doctor', 'patient']):
        return CHUNKS['bio'] + '\n\n' + CHUNKS['projects']
    else:
        return CHUNKS['full']  # General questions get summary

def ask_anju(question):
    """Ask question with smart context selection"""
    
    context = get_relevant_context(question)
    
    prompt = f"""You are answering on behalf of Anju Vilashni Nandhakumar.

CRITICAL: Always respond in FIRST PERSON as Anju.
Use "I built..." not "Anju built..."

Tone: Professional, direct, calm confidence.

Context: {context}

Question: {question}

Answer (as Anju, first person, concise):"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        answer = ask_anju(question)
        
        return jsonify({'answer': answer, 'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)