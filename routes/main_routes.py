from flask import Blueprint, request, render_template, jsonify  
from services.retrieval import retrieve_documents
from services.rag_pipeline import answer_query

main_bp = Blueprint('main', __name__)

# Route for the index (search page)
@main_bp.route('/', methods=['GET', 'POST'])
def index():
    results = None
    query = ''
    model = 'legalbert'
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
       # model = request.form.get('model', 'legalbert')
        if query:
            results = retrieve_documents(query, model)
    return render_template('index.html', results=results, query=query, model=model)

# Route to serve the chatbot UI (HTML page)
@main_bp.route('/chatbot', methods=['GET'])  # <-- Don't reuse the route name
def chatbot_ui():
    return render_template('chatbot.html')

# API endpoint to handle chatbot POST requests
@main_bp.route("/chatbot", methods=["POST"])
def chatbot_api():
    data = request.get_json()
    user_message = data.get("message", "")
    model_name="legalbert"
    try:
        response = answer_query(user_message,model_name=model_name)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500
