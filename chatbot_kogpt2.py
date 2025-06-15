"""
KoGPT2 기반 레시피 챗봇
"""
import json
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from config_kogpt2 import *

class KoGPT2RecipeChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.load_recipe_data()
    
    def load_model(self):
        """훈련된 KoGPT2 모델 로드"""
        try:
            # 훈련된 모델 로드
            self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ 훈련된 KoGPT2 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 훈련된 모델 로드 실패: {e}")
            print("기본 KoGPT2 모델 사용")
            
            # 기본 모델 로드
            self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_recipe_data(self):
        """레시피 데이터 로드 (폴백용)"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            print(f"레시피 데이터 로드: {len(self.recipes)}개")
        except:
            self.recipes = []
            print("레시피 데이터 로드 실패")
    
    def generate_answer(self, question):
        """KoGPT2로 답변 생성"""
        try:
            # 프롬프트 생성
            prompt = PROMPT_FORMAT.format(question=question)
            
            # 토크나이징
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 생성
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + GENERATION_MAX_LENGTH,
                    temperature=GENERATION_CONFIG['temperature'],
                    top_p=GENERATION_CONFIG['top_p'],
                    top_k=GENERATION_CONFIG['top_k'],
                    repetition_penalty=GENERATION_CONFIG['repetition_penalty'],
                    do_sample=GENERATION_CONFIG['do_sample'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 답변 부분만 추출
            if "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1].strip()
                
                # 불필요한 반복 제거
                if len(answer) > 5:
                    return answer
            
            return None
            
        except Exception as e:
            print(f"생성 오류: {e}")
            return None
    
    def fallback_answer(self, question):
        """모델 실패시 간단한 폴백 답변"""
        if not self.recipes:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 레시피 이름 검색
        for recipe in self.recipes:
            if recipe['name'] in question:
                if any(word in question for word in ['만들', '조리', '방법']):
                    return " ".join(recipe['steps']) if recipe['steps'] else "조리 방법 정보가 없습니다."
                elif any(word in question for word in ['재료', '들어가']):
                    return ", ".join(recipe['ingredients']) if recipe['ingredients'] else "재료 정보가 없습니다."
        
        return "죄송합니다. 관련된 레시피를 찾을 수 없습니다."
    
    def chat(self, question):
        """사용자 질문에 답변"""
        if not question.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 물어보세요! 🍳"
        
        # 인사말 처리
        if any(word in question for word in ['안녕', '헬로', '처음']):
            return "안녕하세요! 레시피 챗봇입니다. 요리에 대해 무엇이든 물어보세요! 🍳"
        
        # KoGPT2로 답변 생성 시도
        answer = self.generate_answer(question)
        
        if answer and len(answer) > 5:
            return answer
        else:
            # 폴백 답변
            print("모델 생성 실패 - 폴백 답변 사용")
            return self.fallback_answer(question)

# 테스트
if __name__ == "__main__":
    chatbot = KoGPT2RecipeChatbot()
    
    test_questions = [
        "안녕하세요",
        "김치찌개 어떻게 만들어?",
        "불고기에 뭐가 들어가?",
        "감자로 뭐 만들 수 있어?",
        "부대찌개 어떻게 만들어?",
        "불고기 어떻게 만들어? 자세하게"
    ]
    
    print("🤖 KoGPT2 챗봇 테스트!")
    print("=" * 50)
    
    for question in test_questions:
        answer = chatbot.chat(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")