"""
레시피 데이터 전처리 (KoGPT2 버전) - Context와 Answer 동일
"""
import json
import random
from collections import defaultdict
from config_kogpt2 import *

class KoGPT2DataProcessor:
    def __init__(self):
        self.qa_templates = {
            'cooking_method': [
                "{recipe_name} 어떻게 만들어?",
                "{recipe_name} 만드는 방법 알려줘",
                "{recipe_name} 조리법이 뭐야?",
                "{recipe_name} 요리하는 법",
                "{recipe_name} 만드는 순서"
            ],
            'ingredients': [
                "{recipe_name}에 뭐가 들어가?",
                "{recipe_name} 재료가 뭐야?",
                "{recipe_name} 만들 때 필요한 재료",
                "{recipe_name}의 재료 목록",
                "{recipe_name} 들어가는 재료"
            ],
            'recipe_search': [
                "{ingredient}로 뭐 만들 수 있어?",
                "{ingredient} 요리 추천해줘",
                "{ingredient}를 사용한 음식",
                "{ingredient} 들어간 요리",
                "{ingredient} 넣어서 뭐 만들까?"
            ]
        }
        
        # 일반 대화
        self.general_qa = [
            {
                'question': '안녕하세요',
                'answer': '안녕하세요! 레시피 챗봇입니다. 요리에 대해 무엇이든 물어보세요!'
            },
            {
                'question': '안녕',
                'answer': '안녕하세요! 오늘은 어떤 요리를 만들어보고 싶으신가요?'
            },
            {
                'question': '도움말',
                'answer': '저는 요리 관련 질문에 답변해드려요! 예: 김치찌개 어떻게 만들어?'
            }
        ]
    
    def load_recipes(self):
        """레시피 데이터 로드"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            print(f"레시피 데이터 로드: {len(recipes)}개")
            return recipes
        except FileNotFoundError:
            print("레시피 데이터 파일이 없습니다.")
            return []
    
    def create_cooking_method_data(self, recipe):
        """조리법 데이터 생성 - 모든 스텝 포함"""
        training_data = []
        recipe_name = recipe['name']
        
        # 답변 = 모든 조리 과정
        full_steps = " ".join(recipe['steps']) if recipe['steps'] else "조리 방법 정보가 없습니다."
        
        for template in self.qa_templates['cooking_method']:
            question = template.format(recipe_name=recipe_name)
            
            # KoGPT2 형태: "질문: ... 답변: ..."
            full_text = f"질문: {question}\n답변: {full_steps}{END_TOKEN}"
            
            training_data.append({
                'text': full_text,
                'question': question,
                'answer': full_steps,  # context와 동일
                'type': 'cooking_method'
            })
        
        return training_data
    
    def create_ingredients_data(self, recipe):
        """재료 데이터 생성 - 모든 재료 포함"""
        training_data = []
        recipe_name = recipe['name']
        
        # 답변 = 모든 재료
        all_ingredients = ", ".join(recipe['ingredients']) if recipe['ingredients'] else "재료 정보가 없습니다."
        
        for template in self.qa_templates['ingredients']:
            question = template.format(recipe_name=recipe_name)
            
            full_text = f"질문: {question}\n답변: {all_ingredients}{END_TOKEN}"
            
            training_data.append({
                'text': full_text,
                'question': question,
                'answer': all_ingredients,  # context와 동일
                'type': 'ingredients'
            })
        
        return training_data
    
    def create_recipe_search_data(self, recipes):
        """재료 검색 데이터 생성"""
        training_data = []
        
        # 재료별로 레시피 그룹화
        ingredient_recipes = defaultdict(list)
        for recipe in recipes:
            for ingredient in recipe['ingredients']:
                if len(ingredient) >= 2:
                    clean_ingredient = ingredient.strip()
                    ingredient_recipes[clean_ingredient].append(recipe['name'])
        
        # 2개 이상의 레시피가 있는 재료만 사용
        for ingredient, recipe_names in ingredient_recipes.items():
            if len(recipe_names) >= 2:
                # 중복 제거
                unique_recipes = list(set(recipe_names))
                
                # 답변 = 모든 관련 요리들
                all_recipes = ", ".join(unique_recipes)
                
                for template in self.qa_templates['recipe_search']:
                    question = template.format(ingredient=ingredient)
                    
                    full_text = f"질문: {question}\n답변: {all_recipes}{END_TOKEN}"
                    
                    training_data.append({
                        'text': full_text,
                        'question': question,
                        'answer': all_recipes,  # context와 동일
                        'type': 'recipe_search'
                    })
        
        return training_data
    
    def create_qa_dataset(self):
        """KoGPT2 QA 데이터셋 생성"""
        print("KoGPT2 QA 데이터셋 생성 시작...")
        
        recipes = self.load_recipes()
        if not recipes:
            print("유효한 레시피 데이터가 없습니다.")
            return
        
        all_training_data = []
        
        # 각 레시피에 대해 조리법, 재료 데이터 생성
        for recipe in recipes:
            if recipe.get('name'):
                all_training_data.extend(self.create_cooking_method_data(recipe))
                all_training_data.extend(self.create_ingredients_data(recipe))
        
        # 재료 검색 데이터 생성
        all_training_data.extend(self.create_recipe_search_data(recipes))
        
        # 일반 대화 추가
        for qa in self.general_qa:
            full_text = f"질문: {qa['question']}\n답변: {qa['answer']}{END_TOKEN}"
            
            all_training_data.append({
                'text': full_text,
                'question': qa['question'],
                'answer': qa['answer'],
                'type': 'general'
            })
        
        print(f"총 {len(all_training_data)}개 훈련 데이터 생성")
        
        # 데이터 셔플
        random.shuffle(all_training_data)
        
        # 저장
        with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, ensure_ascii=False, indent=2)
        
        print(f"KoGPT2 데이터셋 저장 완료: {len(all_training_data)}개")
        
        # 샘플 출력
        for i, sample in enumerate(all_training_data[:30]):
            print(f"\n샘플 {i+1}:")
            print(f"질문: {sample['question']}")
            print(f"답변: {sample['answer']}..." if len(sample['answer']) > 100 else f"답변: {sample['answer']}")
            print(f"타입: {sample['type']}")

if __name__ == "__main__":
    processor = KoGPT2DataProcessor()
    processor.create_qa_dataset()