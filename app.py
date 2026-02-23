from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)
CORS(app)

# Simple AI without RAG/embeddings (lighter memory footprint)
class AnjuAI:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.5
        )
        self.knowledge_base = self._load_knowledge_base()
        self.prompt_template = self._create_prompt_template()
    
    def _load_knowledge_base(self):
        """Load full knowledge base"""
        with open('data/anju_knowledge.md', 'r', encoding='utf-8') as f:
            return f.read()
    
    def _create_prompt_template(self):
        """Create prompt template"""
        template = """You are answering questions on behalf of Anju Vilashni Nandhakumar.

CRITICAL: Always respond in FIRST PERSON as Anju speaking directly.
Use "I built..." not "Anju built..."

Tone:
- Professional, conversational, direct
- Confident about strengths, honest about growth areas
- Calm confidence (not defensive or salesy)
- Keep responses concise

Use this knowledge base to answer:

{knowledge_base}

Question: {question}

Answer (as Anju, first person, professional):"""

        return PromptTemplate(
            template=template,
            input_variables=["knowledge_base", "question"]
        )
    
    def ask(self, question):
        """Get answer to question"""
        try:
            prompt = self.prompt_template.format(
                knowledge_base=self.knowledge_base,
                question=question
            )
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize AI
anju_ai = AnjuAI()

@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint for questions"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        answer = anju_ai.ask(question)
        
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
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)