import os
import uuid
from flask import Flask, request, jsonify, render_template, session
import openai
from flask_cors import CORS
import json  # 한글 처리를 위해 json 모듈 사용
import tiktoken  # 토큰 계산을 위해 필요

app = Flask(__name__)
CORS(app)

# OpenAI API 키 설정 (환경 변수 사용)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 세션을 위한 secret key 설정
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key_here")

# 프롬프트를 파일에서 읽어오는 함수
def load_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "기본 프롬프트가 없습니다. prompt.txt 파일을 확인해주세요."

# 토큰 계산 함수
def calculate_tokens(messages, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoder.encode(message["content"]))
    return total_tokens

# 대화 요약 함수
def summarize_conversation(conversation):
    if len(conversation) > 10:  # 메시지가 10개 초과 시 요약
        messages_to_summarize = conversation[:-5]  # 최신 5개 제외한 이전 메시지 요약
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
        # 요약된 메시지와 최신 5개 결합
        return [{"role": "assistant", "content": summary}] + conversation[-5:]
    return conversation

@app.route("/")
def index():
    # 학생에게 고유 세션 ID 부여
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["conversation"] = []  # 개별 대화 기록 초기화
    return render_template("talking GPT.html")

@app.route("/chat", methods=["POST"])
def chat():
    # 클라이언트 요청 데이터 가져오기
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return app.response_class(
            response=json.dumps({"error": "No message provided"}, ensure_ascii=False),
            status=400,
            mimetype="application/json"
        )

    # 세션 ID를 기반으로 대화 이력 관리
    conversation = session.get("conversation", [])
    conversation.append({"role": "user", "content": user_message})

    try:
        # 프롬프트 읽어오기
        system_prompt = load_prompt()

        # 토큰 제한 관리
        max_allowed_tokens = 3000  # 전체 토큰 제한
        conversation_tokens = calculate_tokens(conversation)
        
        # 대화 기록 요약 또는 최신 메시지 제한
        if conversation_tokens > max_allowed_tokens:
            conversation = summarize_conversation(conversation)

        # OpenAI Chat API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}] + conversation,
            max_tokens=3000 - calculate_tokens([{"role": "system", "content": system_prompt}]),
            temperature=0.7,
        )

        # GPT 응답 처리
        bot_reply = response.choices[0].message["content"].strip()
        conversation.append({"role": "assistant", "content": bot_reply})

        # 대화 기록 갱신
        session["conversation"] = conversation

        # JSON 응답 반환 (ensure_ascii=False 적용)
        return app.response_class(
            response=json.dumps({"reply": bot_reply}, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        # 예외 처리 및 오류 메시지 반환
        return app.response_class(
            response=json.dumps({"error": str(e)}, ensure_ascii=False),
            status=500,
            mimetype="application/json"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
