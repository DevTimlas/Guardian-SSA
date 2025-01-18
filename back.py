from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from app_v2 import chat_response

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Flask route to handle user queries
@app.route("/query", methods=["POST"])
def query_ssa_guardian():
    try:
        # Parse the incoming JSON payload
        data = request.get_json()
        user_query = data.get("query", "")
        user_id = data.get("user_id", "1")
        thread_id = data.get("thread_id", "1")

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        # Generate response from SSA Guardian
        response = chat_response(user_query, user_id, thread_id)

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'response': 'Please provide a query.'}), 400

    response = chat_response(user_query)
    return jsonify({'response': response})

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the templates folder


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
