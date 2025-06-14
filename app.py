"""
레시피 챗봇 Flask 웹 앱 (KoGPT2 버전)
"""
from flask import Flask, render_template, request, jsonify
import json
from chatbot_kogpt2 import KoGPT2RecipeChatbot  # KoGPT2 챗봇 사용

app = Flask(__name__)

# 챗봇 초기화
print("KoGPT2 챗봇 초기화 중...")
chatbot = KoGPT2RecipeChatbot()
print("KoGPT2 챗봇 준비 완료!")

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': '메시지가 비어있습니다.'
            })
        
        # 챗봇 응답 생성
        bot_response = chatbot.chat(user_message)
        
        return jsonify({
            'success': True,
            'response': bot_response
        })
        
    except Exception as e:
        print(f"채팅 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'model_type': 'KoGPT2',
        'recipes_count': len(chatbot.recipes) if hasattr(chatbot, 'recipes') else 0,
        'model_loaded': hasattr(chatbot, 'model') and chatbot.model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)