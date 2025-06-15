"""
KoGPT2 기반 레시피 챗봇 (완전한 답변 버전)
"""
import json
import torch
import os
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
            'pytorch_model.bin',  # 또는 model.safetensors
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
                # pytorch_model.bin 대신 model.safetensors가 있을 수 있음
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
        
        # 1단계: 훈련된 모델 존재 확인
        if self.check_trained_model_exists():
            print("📂 훈련된 모델을 로드합니다...")
            
            try:
                # 훈련된 모델 로드
                self.model = GPT2LMHeadModel.from_pretrained(
                    MODEL_PATH,
                    local_files_only=True  # 로컬 파일만 사용
                )
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    MODEL_PATH,
                    local_files_only=True
                )
                
                # GPU로 이동
                self.model.to(self.device)
                self.model.eval()
                
                # 패딩 토큰 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model_loaded = True
                self.is_trained_model = True
                
                print("✅ 훈련된 KoGPT2 모델 로드 완료!")
                print(f"  - 모델 경로: {MODEL_PATH}")
                print(f"  - 어휘 크기: {len(self.tokenizer)}")
                print(f"  - 패딩 토큰: {self.tokenizer.pad_token}")
                
                # 간단한 생성 테스트
                self.test_model()
                return
                
            except Exception as e:
                print(f"❌ 훈련된 모델 로드 실패: {e}")
                print("🔄 기본 모델로 전환합니다...")
        
        # 2단계: 기본 모델 로드
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
            print("   더 나은 성능을 위해서는 model_trainer_kogpt2.py로 모델을 훈련하세요")
            
        except Exception as e:
            print(f"❌ 기본 모델 로드도 실패: {e}")
            self.model_loaded = False
            raise e
    
    def test_model(self):
        """모델 동작 테스트"""
        try:
            test_prompt = "질문: 김치찌개 어떻게 만들어?\n답변:"
            
            input_ids = self.tokenizer.encode(test_prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 30,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"🧪 모델 테스트 성공")
            
        except Exception as e:
            print(f"⚠️ 모델 테스트 실패: {e}")
    
    def load_recipe_data(self):
        """레시피 데이터 로드 (폴백용)"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            print(f"📚 레시피 데이터 로드: {len(self.recipes)}개")
        except Exception as e:
            print(f"⚠️ 레시피 데이터 로드 실패: {e}")
            self.recipes = []
    
    def generate_answer(self, question):
        """KoGPT2로 답변 생성 - 길이 제한 없음"""
        if not self.model_loaded:
            return None
        
        try:
            # 프롬프트 생성
            prompt = PROMPT_FORMAT.format(question=question)
            
            # 토크나이징
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 너무 긴 입력 방지
            if input_ids.shape[1] > MAX_LENGTH - GENERATION_MAX_LENGTH:
                input_ids = input_ids[:, -(MAX_LENGTH - GENERATION_MAX_LENGTH):]
            
            # 생성 - 더 긴 답변 허용
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + GENERATION_MAX_LENGTH * 2,  # 더 긴 생성
                    temperature=GENERATION_CONFIG['temperature'],
                    top_p=GENERATION_CONFIG['top_p'],
                    top_k=GENERATION_CONFIG['top_k'],
                    repetition_penalty=GENERATION_CONFIG['repetition_penalty'],
                    do_sample=GENERATION_CONFIG['do_sample'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    num_return_sequences=1
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 답변 부분만 추출 - 길이 제한 제거
            if "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1].strip()
                
                # 최소 길이만 확인 (너무 짧지 않게)
                if len(answer) > 5 and not self.is_repetitive(answer):
                    return answer
            
            return None
            
        except Exception as e:
            print(f"❌ 생성 오류: {e}")
            return None
    
    def is_repetitive(self, text):
        """반복적인 텍스트인지 확인 - 더 관대하게"""
        words = text.split()
        if len(words) < 6:  # 너무 짧으면 반복 체크 안함
            return False
        
        # 같은 단어가 연속으로 4번 이상 나오면 반복적
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return True
        
        return False
    
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
        return matching_recipes[:5]  # 최대 5개
    
    def format_complete_recipe(self, recipe):
        """완전한 레시피 정보 포맷팅"""
        result = f"🍳 **{recipe['name']}**\n\n"
        
        # 카테고리와 조리법 정보
        if recipe.get('category'):
            result += f"📂 카테고리: {recipe['category']}\n"
        if recipe.get('cooking_method'):
            result += f"🔥 조리법: {recipe['cooking_method']}\n"
        
        # 모든 재료 표시
        if recipe['ingredients']:
            result += f"\n📋 **재료 ({len(recipe['ingredients'])}개):**\n"
            for i, ingredient in enumerate(recipe['ingredients'], 1):
                result += f"{i}. {ingredient}\n"
        
        # 모든 조리 과정 표시
        if recipe['steps']:
            result += f"\n👨‍🍳 **조리 과정 ({len(recipe['steps'])}단계):**\n"
            for i, step in enumerate(recipe['steps'], 1):
                result += f"{i}. {step}\n"
        
        return result
    
    def format_ingredients_only(self, recipe):
        """재료만 완전히 표시"""
        if not recipe['ingredients']:
            return f"{recipe['name']}의 재료 정보가 없습니다."
        
        result = f"🥬 **{recipe['name']} 재료 ({len(recipe['ingredients'])}개):**\n\n"
        for i, ingredient in enumerate(recipe['ingredients'], 1):
            result += f"{i}. {ingredient}\n"
        
        return result
    
    def format_cooking_steps_only(self, recipe):
        """조리 과정만 완전히 표시"""
        if not recipe['steps']:
            return f"{recipe['name']}의 조리 방법 정보가 없습니다."
        
        result = f"👨‍🍳 **{recipe['name']} 만드는 법 ({len(recipe['steps'])}단계):**\n\n"
        for i, step in enumerate(recipe['steps'], 1):
            result += f"{i}. {step}\n"
        
        return result
    
    def fallback_answer(self, question):
        """모델 실패시 완전한 폴백 답변 - 절대 자르지 않음"""
        if not self.recipes:
            return "죄송합니다. 현재 레시피 정보를 로드할 수 없습니다."
        
        # 1. 레시피 이름으로 직접 검색
        recipe = self.find_recipe_by_name(question)
        if recipe:
            # 조리법 질문
            if any(word in question for word in ['만들', '조리', '방법', '어떻게', '과정', '순서']):
                return self.format_cooking_steps_only(recipe)
            
            # 재료 질문
            elif any(word in question for word in ['재료', '들어가', '필요', '넣어']):
                return self.format_ingredients_only(recipe)
            
            # 전체 레시피 질문
            else:
                return self.format_complete_recipe(recipe)
        
        # 2. 재료로 검색
        matching_recipes = self.find_recipes_by_ingredient(question)
        if matching_recipes:
            if len(matching_recipes) == 1:
                recipe = matching_recipes[0]
                return self.format_complete_recipe(recipe)
            else:
                result = f"🔍 해당 재료로 만들 수 있는 요리들:\n\n"
                for i, recipe in enumerate(matching_recipes, 1):
                    result += f"{i}. **{recipe['name']}**\n"
                    # 주요 재료 몇 개만 미리보기
                    if recipe['ingredients']:
                        preview_ingredients = recipe['ingredients'][:3]
                        result += f"   재료: {', '.join(preview_ingredients)}"
                        if len(recipe['ingredients']) > 3:
                            result += f" 외 {len(recipe['ingredients'])-3}개"
                        result += "\n"
                    result += "\n"
                
                result += "💡 구체적인 요리 이름을 말씀하시면 자세한 레시피를 알려드려요!"
                return result
        
        # 3. 일반적인 요리 추천
        if any(word in question for word in ['추천', '뭐', '무엇', '어떤']):
            popular_recipes = self.recipes[:10]  # 처음 10개 레시피
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
        """사용자 질문에 답변 - 완전한 답변 제공"""
        if not question.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 물어보세요! 🍳"
        
        # 인사말 처리
        if any(word in question for word in ['안녕', '헬로', '처음', '반가워']):
            model_status = "훈련된 모델" if self.is_trained_model else "기본 모델"
            return f"안녕하세요! KoGPT2 레시피 챗봇입니다 ({model_status}). 요리에 대해 무엇이든 물어보세요! 🍳\n\n💡 예시:\n• '김치찌개 어떻게 만들어?'\n• '불고기 재료가 뭐야?'\n• '감자로 뭐 만들 수 있어?'"
        
        # 도움말 처리
        if any(word in question for word in ['도움', '사용법', '명령어']):
            return "📖 **사용 방법:**\n\n🍳 **조리법 질문:**\n• '김치찌개 어떻게 만들어?'\n• '불고기 만드는 방법'\n• '된장찌개 조리법'\n\n🥬 **재료 질문:**\n• '김치찌개 재료가 뭐야?'\n• '불고기에 뭐가 들어가?'\n\n🔍 **요리 검색:**\n• '감자로 뭐 만들 수 있어?'\n• '닭고기 요리 추천'\n\n모든 조리 과정과 재료를 완전히 보여드립니다!"
        
        # KoGPT2로 답변 생성 시도 (훈련된 모델인 경우)
        if self.is_trained_model:
            answer = self.generate_answer(question)
            
            if answer and len(answer.strip()) > 3:
                return answer
            else:
                print("⚠️ 모델 생성 실패 - 완전한 폴백 답변 사용")
        else:
            print("ℹ️ 기본 모델 사용 - 완전한 폴백 답변 사용")
        
        # 완전한 폴백 답변
        return self.fallback_answer(question)

# 테스트
if __name__ == "__main__":
    print("🤖 KoGPT2 레시피 챗봇 테스트 시작!")
    print("=" * 60)
    
    chatbot = KoGPT2RecipeChatbot()
    
    test_questions = [
        "안녕하세요",
        "김치찌개 어떻게 만들어?",
        "김치찌개 재료가 뭐야?",
        "감자로 뭐 만들 수 있어?",
        "요리 추천해줘",
        "도움말"
    ]
    
    for question in test_questions:
        print(f"\n💬 Q: {question}")
        answer = chatbot.chat(question)
        print(f"🤖 A: {answer}")
        print("-" * 60)