import os
from flask import Flask, request, jsonify, render_template
import openai
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # '/chat' 경로에 대한 CORS 허용

# OpenAI API 키 설정 (환경 변수 사용)
openai.api_key = os.environ.get("OPENAI_API_KEY")
@app.route('/')
def index():
    return render_template('talking GPT.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful private detective. Speak in 100 characters or less"},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100,
            temperature=0.7
        )

        bot_reply = response.choices[0].message['content'].strip()
        return jsonify({'reply': bot_reply})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)