from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(dotenv_path='.env')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000'),
    'http://127.0.0.1:3000',
    'http://localhost:5173',
    'http://127.0.0.1:5173'
]}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization", "x-api-key"], expose_headers=["Content-Type"], supports_credentials=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_TOKEN")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or os.getenv("API_AUTH_TOKEN")

if not GOOGLE_API_KEY:
    print("⚠️ GOOGLE_API_KEY is not set. Gemini requests will fail.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


def _check_internal_auth(req):
    if not INTERNAL_API_KEY:
        return None  # auth not required
    if req.method == 'OPTIONS' or req.path == '/health':
        return None

    auth_header = req.headers.get('Authorization', '')
    bearer = auth_header[7:] if isinstance(auth_header, str) and auth_header.startswith('Bearer ') else None
    x_api_key = req.headers.get('x-api-key') if isinstance(req.headers.get('x-api-key'), str) else None
    query_key = req.args.get('api_key') if isinstance(req.args.get('api_key'), str) else None
    body_json = req.get_json(silent=True) or {}
    body_key = body_json.get('api_key') if isinstance(body_json.get('api_key'), str) else None

    provided = bearer or x_api_key or query_key or body_key
    if not provided or provided != INTERNAL_API_KEY:
        return jsonify({
            'error': 'Invalid or missing internal API key',
            'hint': 'Provide Authorization: Bearer <INTERNAL_API_KEY> or x-api-key, or api_key in query/body'
        }), 401
    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Gemini Proxy API (Flask)',
        'llm_integration': 'Google Generative AI',
        'api_key_status': 'configured' if GOOGLE_API_KEY else 'not configured',
        'model': 'gemini-1.5-flash-latest'
    })


@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return '', 204
    # Internal auth (optional)
    auth_err = _check_internal_auth(request)
    if auth_err:
        return auth_err

    if not GOOGLE_API_KEY:
        return jsonify({'error': 'GOOGLE_API_KEY is not configured on the server'}), 500

    data = request.get_json(silent=True) or {}
    mode = (data.get('mode') or 'prompt').strip()
    inputs = (data.get('inputs') or '').strip()
    model_name = data.get('model') or 'gemini-1.5-flash-latest'
    parameters = data.get('parameters') or {}

    if mode != 'random' and not inputs:
        return jsonify({'error': "Missing 'inputs' parameter"}), 400

    if mode == 'random' and not inputs:
        inputs = 'Write a short, whimsical story suitable for kids.'

    # Map parameters to generation_config
    generation_config = {}
    if 'temperature' in parameters:
        generation_config['temperature'] = parameters['temperature']
    if 'top_p' in parameters:
        generation_config['top_p'] = parameters['top_p']
    if 'top_k' in parameters:
        generation_config['top_k'] = parameters['top_k']
    if 'max_new_tokens' in parameters:
        generation_config['max_output_tokens'] = parameters['max_new_tokens']

    # Clean None values
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(inputs, generation_config=generation_config)

        # Handle safety blocks or empty responses
        text = getattr(resp, 'text', None)
        if not text:
            return jsonify({'error': 'Empty response', 'raw': resp.to_dict() if hasattr(resp, 'to_dict') else None}), 502

        return jsonify({
            'candidates': [
                {
                    'content': {
                        'parts': [{'text': text}],
                        'role': 'model'
                    }
                }
            ],
            'text': text
        })
    except Exception as e:
        print(f"❌ Google API error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    print(f"🚀 Gemini Proxy API running on http://localhost:{port}")
    print("📡 Health check: /health")
    print("🤖 Generation API: /api/generate")
    app.run(host='0.0.0.0', port=port, debug=True) 