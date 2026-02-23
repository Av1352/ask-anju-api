from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Allow requests from your GoDaddy site

# Initialize RAG system
class AnjuAI:
    def __init__(self):
        # Use Gemini (FREE)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.5
    )
        self.vectorstore = self._create_vectorstore()
        self.qa_chain = self._create_qa_chain()
    
    def _load_knowledge_base(self):
        """Load knowledge base from data file"""
        with open('data/anju_knowledge.md', 'r', encoding='utf-8') as f:
            return f.read()
    
    def _create_vectorstore(self):
        """Create FAISS vectorstore"""
        text = self._load_knowledge_base()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        
        return FAISS.from_texts(chunks, self.embeddings)
    
    def _create_qa_chain(self):
        """Create QA chain with professional prompt"""
        template = """You are answering questions on behalf of Anju Vilashni Nandhakumar, 
an ML Engineer specializing in healthcare AI and computer vision.

CRITICAL: Always respond in FIRST PERSON as if you ARE Anju speaking directly.
Use "I built..." not "Anju built..." or "She built..."

When answering:
- Be professional but conversational and direct
- Provide specific technical details and metrics
- Keep responses concise (Anju's style)
- Use actual achievements and numbers from the context
- Link to portfolio (vxanju.com) or GitHub when appropriate

For job search questions: Mention I'm actively evaluating opportunities. 
Work authorization: F1 OPT through June 2026, then STEM OPT through 2028 
(2+ years total, no sponsorship cost).

If you don't have information: "I don't have those specific details. 
Check out vxanju.com or email me at nandhakumar.anju@gmail.com"

DO NOT make up information. Only use what's in the context. Speak AS Anju.

Context: {context}

Question: {question}

Answer (first person, as Anju):"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def ask(self, question):
        """Get answer to question"""
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error: {e}"

# Initialize AI (do this once at startup)
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