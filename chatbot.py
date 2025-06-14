"""
레시피 QA 챗봇 추론 클래스
"""
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_trainer import RecipeQAModel
from config import *
import re

class RecipeChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.load_recipe_data()
    
    def load_model(self):
        """훈련된 모델 로드"""
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # 모델 로드
            self.model = RecipeQAModel()
            self.model.load_state_dict(torch.load(
                os.path.join(MODEL_PATH, 'pytorch_model.bin'),
                map_location=self.device
            ))
            self.model.to(self.device)
            self.model.eval()
            
            print("모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            # 기본 모델 사용
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = RecipeQAModel()
            self.model.to(self.device)
            self.model.eval()
            print("기본 모델 사용")
    
    def load_recipe_data(self):
        """레시피 데이터 로드"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            print(f"레시피 데이터 로드: {len(self.recipes)}개")
        except:
            self.recipes = []
            print("레시피 데이터 로드 실패")
    
    def find_relevant_recipes(self, question):
        """질문과 관련된 레시피 찾기"""
        relevant_recipes = []
        
        # 레시피 이름이 질문에 포함된 경우
        for recipe in self.recipes:
            if recipe['name'] in question:
                relevant_recipes.append(recipe)
        
        # 재료가 질문에 포함된 경우
        if not relevant_recipes:
            for recipe in self.recipes:
                for ingredient in recipe['ingredients']:
                    if len(ingredient) >= 2 and ingredient in question:
                        relevant_recipes.append(recipe)
                        break
        
        # 조리 방법이 질문에 포함된 경우
        if not relevant_recipes:
            cooking_methods = ['볶음', '구이', '찜', '조림', '무침', '탕', '국', '찌개']
            for method in cooking_methods:
                if method in question:
                    for recipe in self.recipes:
                        if method in recipe.get('cooking_method', ''):
                            relevant_recipes.append(recipe)
        
        return relevant_recipes[:3]  # 최대 3개
    
    def create_context(self, question, recipes):
        """질문에 맞는 context 생성"""
        if not recipes:
            return "죄송합니다. 관련된 레시피를 찾을 수 없습니다."
        
        context_parts = []
        
        for recipe in recipes:
            recipe_info = f"{recipe['name']}: "
            
            # 조리법 질문인 경우
            if any(word in question for word in ['만들', '조리', '방법', '어떻게']):
                if recipe['steps']:
                    recipe_info += " ".join(recipe['steps'][:3])
            
            # 재료 질문인 경우
            elif any(word in question for word in ['재료', '들어가', '필요']):
                if recipe['ingredients']:
                    recipe_info += ", ".join(recipe['ingredients'][:5])
            
            # 일반적인 경우
            else:
                recipe_info += f"재료: {', '.join(recipe['ingredients'][:3])}. "
                if recipe['steps']:
                    recipe_info += f"조리법: {recipe['steps'][0]}"
            
            context_parts.append(recipe_info)
        
        return " ".join(context_parts)
    
    def extract_answer(self, question, context, start_logits, end_logits):
        """모델 출력에서 답변 추출"""
        # '[CLS] 질문 [SEP] 지문 [SEP]' 형태로 인코딩
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        
        # 소프트맥스 적용
        start_probs = F.softmax(start_logits[0], dim=0)
        end_probs = F.softmax(end_logits[0], dim=0)
        
        # 가장 높은 확률의 시작/끝 위치 찾기
        start_idx = torch.argmax(start_probs).item()
        end_idx = torch.argmax(end_probs).item()
        
        # 시작이 끝보다 뒤에 있으면 조정
        if start_idx > end_idx:
            end_idx = start_idx
        
        # 답변 토큰 추출
        answer_tokens = input_ids[start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_simple_answer(self, question, recipes):
        """간단한 규칙 기반 답변 생성 (모델이 실패할 경우)"""
        if not recipes:
            return "죄송합니다. 관련된 레시피를 찾을 수 없습니다."
        
        recipe = recipes[0]  # 첫 번째 레시피 사용
        
        # 조리법 질문
        if any(word in question for word in ['만들', '조리', '방법', '어떻게']):
            if recipe['steps']:
                return f"{recipe['name']} 만드는 법: {recipe['steps'][0]}"
        
        # 재료 질문
        elif any(word in question for word in ['재료', '들어가', '필요']):
            if recipe['ingredients']:
                return f"{recipe['name']}의 주요 재료: {', '.join(recipe['ingredients'][:4])}"
        
        # 추천 질문
        else:
            recipe_names = [r['name'] for r in recipes[:3]]
            return f"추천 요리: {', '.join(recipe_names)}"
        
        return f"{recipe['name']}에 대한 정보를 찾았습니다."
    
    def chat(self, question):
        """사용자 질문에 답변"""
        if not question.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 물어보세요! 🍳"
        
        # 인사말 처리
        if any(word in question for word in ['안녕', '헬로', '처음']):
            return "안녕하세요! 레시피 챗봇입니다. 요리에 대해 무엇이든 물어보세요! 🍳"
        
        # 관련 레시피 찾기
        relevant_recipes = self.find_relevant_recipes(question)
        
        # context 생성
        context = self.create_context(question, relevant_recipes)
        
        try:
            # 모델 추론
            encoding = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            with torch.no_grad():
                start_logits, end_logits = self.model(
                    input_ids, attention_mask, token_type_ids
                )
            
            # 답변 추출
            answer = self.extract_answer(question, context, start_logits, end_logits)
            
            # 답변이 너무 짧거나 의미없으면 규칙 기반 답변 사용
            if len(answer.strip()) < 3 or answer.strip() in ['[CLS]', '[SEP]', '[PAD]']:
                answer = self.generate_simple_answer(question, relevant_recipes)
            
            return answer
            
        except Exception as e:
            print(f"모델 추론 오류: {e}")
            # 규칙 기반 답변으로 폴백
            return self.generate_simple_answer(question, relevant_recipes)

# 테스트
if __name__ == "__main__":
    chatbot = RecipeChatbot()
    
    test_questions = [
        "안녕하세요",
        "김치찌개 어떻게 만들어?",
        "불고기에 뭐가 들어가?",
        "감자로 뭐 만들 수 있어?"
    ]
    
    for question in test_questions:
        answer = chatbot.chat(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")