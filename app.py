import os
import uuid
from flask import Flask, request, jsonify, render_template, session
import openai
from flask_cors import CORS
import json
import tiktoken




app = Flask(__name__)
CORS(app)

# OpenAI API 키 설정
openai.api_key = os.environ.get("OPENAI_API_KEY")
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key_here")



# 프롬프트를 파일에서 읽어오는 함수
def load_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            prompt = file.read()
            print("[DEBUG] Prompt loaded successfully.")
            return prompt
    except FileNotFoundError:
        print("[ERROR] Prompt file not found.")
        return "기본 프롬프트가 없습니다. prompt.txt 파일을 확인해주세요."

# 토큰 계산 함수
def calculate_tokens(messages, model="gpt-3.5-turbo"):
    try:
        encoder = tiktoken.encoding_for_model(model)
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoder.encode(message["content"]))
        print(f"[DEBUG] Calculated total tokens: {total_tokens}")
        return total_tokens
    except Exception as e:
        print(f"[ERROR] Error calculating tokens: {e}")
        raise

# 대화 요약 함수
def summarize_conversation(conversation):
    try:
        if len(conversation) > 10:
            messages_to_summarize = conversation[:-5]
            print("[DEBUG] Summarizing conversation.")
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "다음 대화를 간단히 요약하세요."},
                    {"role": "user", "content": str(messages_to_summarize)},
                ],
                max_tokens=300,
                temperature=0.5,
            )
            summary = summary_response.choices[0].message["content"]
            print(f"[DEBUG] Summary created: {summary}")
            return [{"role": "assistant", "content": summary}] + conversation[-5:]
        return conversation
    except Exception as e:
        print(f"[ERROR] Error summarizing conversation: {e}")
        raise

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["conversation"] = []
    print("[DEBUG] Session initialized.")
    return render_template("talking GPT.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        print("[WARNING] No message provided by user.")
        return jsonify({"error": "No message provided"}), 400

    conversation = session.get("conversation", [])
    conversation.append({"role": "user", "content": user_message})
    print(f"[DEBUG] User message added: {user_message}")

    try:
        # 프롬프트 읽기
        system_prompt = load_prompt()

        # 토큰 제한 관리
        max_allowed_tokens = 3000
        conversation_tokens = calculate_tokens(conversation)
        
        # 대화 기록 요약 또는 최신 메시지 제한
        if conversation_tokens > max_allowed_tokens:
            print("[INFO] Conversation tokens exceeded limit, summarizing...")
            conversation = summarize_conversation(conversation)

        # OpenAI Chat API 호출
        print("[DEBUG] Calling OpenAI API...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}] + conversation,
            max_tokens=3000 - calculate_tokens([{"role": "system", "content": system_prompt}]),
            temperature=0.7,
        )

        bot_reply = response.choices[0].message["content"].strip()
        print(f"[DEBUG] Bot reply received: {bot_reply}")
        conversation.append({"role": "assistant", "content": bot_reply})

        # 대화 기록 갱신
        session["conversation"] = conversation

        # 응답 반환
        return app.response_class(
            response=json.dumps({"reply": bot_reply}, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        # 예외 발생 시 디버깅 정보 기록
        print(f"[ERROR] Error during chat handling: {e}")
        return app.response_class(
            response=json.dumps({"error": str(e)}, ensure_ascii=False),
            status=500,
            mimetype="application/json"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
