"""
KoGPT2 기반 레시피 챗봇 (수정된 버전) - 생성 문제 해결
"""
import json
import torch
import os
import re
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from config_kogpt2 import *

class KoGPT2RecipeChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")
        
        # 모델 상태 추적
        self.model_loaded = False
        self.is_trained_model = False
        
        self.load_model()
        self.load_recipe_data()
    
    def check_trained_model_exists(self):
        """훈련된 모델 파일들이 존재하는지 확인"""
        required_files = [
            'config.json',
            'pytorch_model.bin',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 모델 디렉토리가 없습니다: {MODEL_PATH}")
            return False
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(MODEL_PATH, file)
            if not os.path.exists(file_path):
                if file == 'pytorch_model.bin':
                    alt_path = os.path.join(MODEL_PATH, 'model.safetensors')
                    if os.path.exists(alt_path):
                        continue
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 누락된 모델 파일들: {missing_files}")
            return False
        
        print("✅ 훈련된 모델 파일들이 모두 존재합니다")
        return True
    
    def load_model(self):
        """훈련된 KoGPT2 모델 로드"""
        print("🔄 모델 로드 시작...")
        
        if self.check_trained_model_exists():
            print("📂 훈련된 모델을 로드합니다...")
            
            try:
                self.model = GPT2LMHeadModel.from_pretrained(
                    MODEL_PATH,
                    local_files_only=True
                )
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    MODEL_PATH,
                    local_files_only=True
                )
                
                self.model.to(self.device)
                self.model.eval()
                
                # 패딩 토큰 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model_loaded = True
                self.is_trained_model = True
                
                print("✅ 훈련된 KoGPT2 모델 로드 완료!")
                print(f"  - EOS 토큰 ID: {self.tokenizer.eos_token_id}")
                print(f"  - PAD 토큰 ID: {self.tokenizer.pad_token_id}")
                
                return
                
            except Exception as e:
                print(f"❌ 훈련된 모델 로드 실패: {e}")
                print("🔄 기본 모델로 전환합니다...")
        
        # 기본 모델 로드
        print("📂 기본 KoGPT2 모델을 로드합니다...")
        
        try:
            self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            self.is_trained_model = False
            
            print("✅ 기본 KoGPT2 모델 로드 완료")
            print("⚠️ 주의: 훈련되지 않은 기본 모델을 사용합니다")
            
        except Exception as e:
            print(f"❌ 기본 모델 로드도 실패: {e}")
            self.model_loaded = False
            raise e
    
    def load_recipe_data(self):
        """레시피 데이터 로드"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            print(f"📚 레시피 데이터 로드: {len(self.recipes)}개")
        except Exception as e:
            print(f"⚠️ 레시피 데이터 로드 실패: {e}")
            self.recipes = []
    
    def clean_generated_text(self, text):
        """생성된 텍스트를 정리"""
        # EOS 토큰에서 자르기
        if '<|endoftext|>' in text:
            text = text.split('<|endoftext|>')[0]
        
        # 이상한 토큰 패턴 제거
        text = re.sub(r'<\|[^>]*\|>', '', text)
        text = re.sub(r'[<>|{}]+', '', text)
        
        # 반복되는 구두점이나 문자 제거
        text = re.sub(r'([.!?])\1{2,}', r'\1', text)
        text = re.sub(r'(\w)\1{4,}', r'\1', text)
        
        # 줄바꿈과 공백 정리
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_answer(self, question):
        """KoGPT2로 답변 생성 - 수정된 버전"""
        if not self.model_loaded:
            return None
        
        try:
            # 프롬프트 생성
            prompt = PROMPT_FORMAT.format(question=question)
            
            # 토크나이징
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 입력 길이 제한
            if input_ids.shape[1] > 500:  # 더 짧게 제한
                input_ids = input_ids[:, -500:]
            
            # 수정된 생성 파라미터
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=300,  # max_length 대신 max_new_tokens 사용
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    use_cache=True
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            
            # 답변 부분만 추출
            if "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1]
                
                # 텍스트 정리
                answer = self.clean_generated_text(answer)
                
                # 최소 길이 확인 및 품질 체크
                if len(answer) > 10 and self.is_valid_answer(answer):
                    return answer
            
            return None
            
        except Exception as e:
            print(f"❌ 생성 오류: {e}")
            return None
    
    def is_valid_answer(self, text):
        """생성된 답변이 유효한지 확인"""
        # 너무 짧은 답변
        if len(text) < 10:
            return False
        
        # 의미없는 토큰들로만 구성된 경우
        if re.match(r'^[<>|{}\s]+$', text):
            return False
        
        # 반복이 너무 많은 경우
        words = text.split()
        if len(words) < 3:
            return False
        
        # 같은 단어가 연속으로 3번 이상 나오는 경우
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return False
        
        return True
    
    def find_recipe_by_name(self, question):
        """요리 이름으로 레시피 찾기"""
        for recipe in self.recipes:
            if recipe['name'] in question:
                return recipe
        return None
    
    def find_recipes_by_ingredient(self, question):
        """재료로 레시피 찾기"""
        matching_recipes = []
        for recipe in self.recipes:
            for ingredient in recipe['ingredients']:
                if len(ingredient) >= 2 and ingredient in question:
                    matching_recipes.append(recipe)
                    break
        return matching_recipes[:5]
    
    def format_complete_recipe(self, recipe):
        """완전한 레시피 정보 포맷팅"""
        result = f"🍳 **{recipe['name']}**\n\n"
        
        if recipe.get('category'):
            result += f"📂 카테고리: {recipe['category']}\n"
        if recipe.get('cooking_method'):
            result += f"🔥 조리법: {recipe['cooking_method']}\n"
        
        if recipe['ingredients']:
            result += f"\n📋 **재료 ({len(recipe['ingredients'])}개):**\n"
            for i, ingredient in enumerate(recipe['ingredients'], 1):
                result += f"{i}. {ingredient}\n"
        
        if recipe['steps']:
            result += f"\n👨‍🍳 **조리 과정 ({len(recipe['steps'])}단계):**\n"
            for i, step in enumerate(recipe['steps'], 1):
                result += f"{i}. {step}\n"
        
        return result
    
    def format_ingredients_only(self, recipe):
        """재료만 표시"""
        if not recipe['ingredients']:
            return f"{recipe['name']}의 재료 정보가 없습니다."
        
        result = f"🥬 **{recipe['name']} 재료 ({len(recipe['ingredients'])}개):**\n\n"
        for i, ingredient in enumerate(recipe['ingredients'], 1):
            result += f"{i}. {ingredient}\n"
        
        return result
    
    def format_cooking_steps_only(self, recipe):
        """조리 과정만 표시"""
        if not recipe['steps']:
            return f"{recipe['name']}의 조리 방법 정보가 없습니다."
        
        result = f"👨‍🍳 **{recipe['name']} 만드는 법 ({len(recipe['steps'])}단계):**\n\n"
        for i, step in enumerate(recipe['steps'], 1):
            result += f"{i}. {step}\n"
        
        return result
    
    def fallback_answer(self, question):
        """폴백 답변"""
        if not self.recipes:
            return "죄송합니다. 현재 레시피 정보를 로드할 수 없습니다."
        
        # 레시피 이름으로 직접 검색
        recipe = self.find_recipe_by_name(question)
        if recipe:
            if any(word in question for word in ['만들', '조리', '방법', '어떻게', '과정', '순서']):
                return self.format_cooking_steps_only(recipe)
            elif any(word in question for word in ['재료', '들어가', '필요', '넣어']):
                return self.format_ingredients_only(recipe)
            else:
                return self.format_complete_recipe(recipe)
        
        # 재료로 검색
        matching_recipes = self.find_recipes_by_ingredient(question)
        if matching_recipes:
            if len(matching_recipes) == 1:
                return self.format_complete_recipe(matching_recipes[0])
            else:
                result = f"🔍 해당 재료로 만들 수 있는 요리들:\n\n"
                for i, recipe in enumerate(matching_recipes, 1):
                    result += f"{i}. **{recipe['name']}**\n"
                    if recipe['ingredients']:
                        preview_ingredients = recipe['ingredients'][:3]
                        result += f"   재료: {', '.join(preview_ingredients)}"
                        if len(recipe['ingredients']) > 3:
                            result += f" 외 {len(recipe['ingredients'])-3}개"
                        result += "\n"
                    result += "\n"
                
                result += "💡 구체적인 요리 이름을 말씀하시면 자세한 레시피를 알려드려요!"
                return result
        
        # 일반적인 요리 추천
        if any(word in question for word in ['추천', '뭐', '무엇', '어떤']):
            popular_recipes = self.recipes[:10]
            result = "🍳 **인기 요리 추천:**\n\n"
            for i, recipe in enumerate(popular_recipes, 1):
                result += f"{i}. {recipe['name']}"
                if recipe.get('category'):
                    result += f" ({recipe['category']})"
                result += "\n"
            
            result += "\n💡 원하시는 요리 이름을 말씀하시면 자세한 레시피를 알려드려요!"
            return result
        
        return "죄송합니다. 관련된 레시피를 찾을 수 없습니다.\n\n💡 이렇게 물어보세요:\n• '김치찌개 어떻게 만들어?'\n• '불고기 재료가 뭐야?'\n• '감자로 뭐 만들 수 있어?'"
    
    def chat(self, question):
        """사용자 질문에 답변"""
        if not question.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 물어보세요! 🍳"
        
        # 인사말 처리
        if any(word in question for word in ['안녕', '헬로', '처음', '반가워']):
            model_status = "훈련된 모델" if self.is_trained_model else "기본 모델"
            return f"안녕하세요! KoGPT2 레시피 챗봇입니다 ({model_status}). 요리에 대해 무엇이든 물어보세요! 🍳\n\n💡 예시:\n• '김치찌개 어떻게 만들어?'\n• '불고기 재료가 뭐야?'\n• '감자로 뭐 만들 수 있어?'"
        
        # 도움말 처리
        if any(word in question for word in ['도움', '사용법', '명령어']):
            return "📖 **사용 방법:**\n\n🍳 **조리법 질문:**\n• '김치찌개 어떻게 만들어?'\n• '불고기 만드는 방법'\n• '된장찌개 조리법'\n\n🥬 **재료 질문:**\n• '김치찌개 재료가 뭐야?'\n• '불고기에 뭐가 들어가?'\n\n🔍 **요리 검색:**\n• '감자로 뭐 만들 수 있어?'\n• '닭고기 요리 추천'\n\n모든 조리 과정과 재료를 완전히 보여드립니다!"
        
        # KoGPT2로 답변 생성 시도 (훈련된 모델인 경우에만)
        if self.is_trained_model:
            answer = self.generate_answer(question)
            
            if answer and len(answer.strip()) > 10:
                # 생성된 답변이 유효하면 반환
                return answer
            else:
                print("⚠️ 모델 생성 실패 - 폴백 답변 사용")
        else:
            print("ℹ️ 기본 모델 사용 - 폴백 답변 사용")
        
        # 폴백 답변
        return self.fallback_answer(question)

# 테스트
if __name__ == "__main__":
    print("🤖 수정된 KoGPT2 레시피 챗봇 테스트!")
    print("=" * 60)
    
    chatbot = KoGPT2RecipeChatbot()
    
    test_questions = [
        "안녕하세요",
        "김치찌개 어떻게 만들어?",
        "김치찌개 재료가 뭐야?",
        "감자로 뭐 만들 수 있어?",
        "요리 추천해줘",
        "도움말",
        "약과 만드는 방법",
        "청국장 만드는 방법",
        "부대찌개 만드는 방법",
        "부대찌개 재료",
        "칼국수 만드는 방법 자세히"
    ]
    
    for question in test_questions:
        print(f"\n💬 Q: {question}")
        answer = chatbot.chat(question)
        print(f"🤖 A: {answer}")
        print("-" * 60)