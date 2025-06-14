"""
레시피 데이터 전처리 및 QA 데이터셋 생성
"""
import json
import random
from collections import defaultdict
from config import *

class RecipeQAProcessor:
    def __init__(self):
        self.qa_templates = {
            'cooking_method': [
                "{recipe_name} 어떻게 만들어?",
                "{recipe_name} 만드는 방법 알려줘",
                "{recipe_name} 조리법이 뭐야?",
                "{recipe_name} 요리하는 법",
                "{recipe_name} 만드는 순서",
                "{recipe_name} 조리 과정"
            ],
            'ingredients': [
                "{recipe_name}에 뭐가 들어가?",
                "{recipe_name} 재료가 뭐야?",
                "{recipe_name} 만들 때 필요한 재료",
                "{recipe_name}의 재료 목록",
                "{recipe_name} 들어가는 재료",
                "{recipe_name} 필요한 것들"
            ],
            'recipe_search': [
                "{ingredient}로 뭐 만들 수 있어?",
                "{ingredient} 요리 추천해줘",
                "{ingredient}를 사용한 음식",
                "{ingredient} 들어간 요리",
                "{ingredient} 넣어서 뭐 만들까?",
                "{ingredient}가 들어간 요리는?"
            ]
        }
        
        # 일반 QA 추가
        self.general_qa = [
            {
                'question': '안녕하세요',
                'context': '레시피 챗봇 인사말',
                'answer': '안녕하세요! 레시피 챗봇입니다. 요리에 대해 무엇이든 물어보세요!',
                'type': 'greeting'
            },
            {
                'question': '안녕',
                'context': '레시피 챗봇 인사말',
                'answer': '안녕하세요! 오늘은 어떤 요리를 만들어보고 싶으신가요?',
                'type': 'greeting'
            },
            {
                'question': '도움말',
                'context': '레시피 챗봇 사용법',
                'answer': '레시피 챗봇 사용법: 1. 재료로 요리 검색 2. 특정 요리 레시피 3. 요리 재료 확인',
                'type': 'help'
            }
        ]
    
    def load_recipes(self):
        """레시피 데이터 로드"""
        try:
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            
            if not recipes:
                print("레시피 데이터가 비어있습니다.")
                return []
            
            print(f"레시피 데이터 로드: {len(recipes)}개")
            return recipes
            
        except FileNotFoundError:
            print("레시피 데이터 파일이 없습니다. 먼저 data_collector.py를 실행하세요.")
            return []
        except Exception as e:
            print(f"레시피 데이터 로드 실패: {e}")
            return []
    
    def generate_cooking_method_qa(self, recipe):
        """조리법 QA 생성"""
        qa_pairs = []
        recipe_name = recipe['name']
        steps = recipe['steps']
        
        if len(steps) < 1:
            return qa_pairs
        
        # 조리 과정을 context로 만들기
        context = f"{recipe_name} 만드는 방법: " + " ".join(steps[:3])
        
        for template in self.qa_templates['cooking_method']:
            question = template.format(recipe_name=recipe_name)
            
            # 답변은 첫 번째 조리 단계로 설정
            answer = steps[0] if steps else ""
            
            if answer:
                qa_pairs.append({
                    'question': question,
                    'context': context,
                    'answer': answer,
                    'type': 'cooking_method'
                })
        
        return qa_pairs
    
    def generate_ingredients_qa(self, recipe):
        """재료 QA 생성"""
        qa_pairs = []
        recipe_name = recipe['name']
        ingredients = recipe['ingredients']
        
        if len(ingredients) < 1:
            return qa_pairs
        
        # 재료 목록을 context로 만들기
        ingredients_text = ", ".join(ingredients[:5])
        context = f"{recipe_name}의 재료: {ingredients_text}"
        
        for template in self.qa_templates['ingredients']:
            question = template.format(recipe_name=recipe_name)
            
            # 답변은 주요 재료 3개로 설정
            answer = ", ".join(ingredients[:3])
            
            qa_pairs.append({
                'question': question,
                'context': context,
                'answer': answer,
                'type': 'ingredients'
            })
        
        return qa_pairs
    
    def generate_recipe_search_qa(self, recipes):
        """재료로 레시피 검색 QA 생성"""
        qa_pairs = []
        
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
                
                # context는 해당 재료로 만들 수 있는 요리들
                context = f"{ingredient}로 만들 수 있는 요리: " + ", ".join(unique_recipes[:5])
                
                for template in self.qa_templates['recipe_search']:
                    question = template.format(ingredient=ingredient)
                    
                    # 답변은 대표 요리 2개
                    answer = ", ".join(unique_recipes[:2])
                    
                    qa_pairs.append({
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'type': 'recipe_search'
                    })
        
        return qa_pairs
    
    def process_for_bert(self, qa_pairs):
        """BERT 입력 형태로 변환"""
        processed_data = []
        
        for qa in qa_pairs:
            question = qa['question']
            context = qa['context']
            answer = qa['answer']
            
            # BERT 입력 형태로 변환
            input_text = f"[CLS] {question} [SEP] {context} [SEP]"
            
            # 답변의 시작과 끝 위치 찾기
            context_start = input_text.find(context)
            answer_start = context.find(answer)
            
            if answer_start != -1:
                # 전체 텍스트에서의 실제 위치 계산
                actual_answer_start = context_start + answer_start
                actual_answer_end = actual_answer_start + len(answer) - 1
                
                processed_data.append({
                    'input_text': input_text,
                    'question': question,
                    'context': context,
                    'answer': answer,
                    'answer_start': actual_answer_start,
                    'answer_end': actual_answer_end,
                    'type': qa['type']
                })
        
        return processed_data
    
    def create_qa_dataset(self):
        """QA 데이터셋 생성"""
        print("QA 데이터셋 생성 시작...")
        
        recipes = self.load_recipes()
        if not recipes:
            print("유효한 레시피 데이터가 없습니다.")
            return
        
        all_qa_pairs = []
        
        # 각 레시피에 대해 조리법, 재료 QA 생성
        for recipe in recipes:
            if recipe.get('name'):
                all_qa_pairs.extend(self.generate_cooking_method_qa(recipe))
                all_qa_pairs.extend(self.generate_ingredients_qa(recipe))
        
        # 재료 검색 QA 생성
        all_qa_pairs.extend(self.generate_recipe_search_qa(recipes))
        
        # 일반 QA 추가
        all_qa_pairs.extend(self.general_qa)
        
        print(f"총 {len(all_qa_pairs)}개 QA 쌍 생성")
        
        if not all_qa_pairs:
            print("생성된 QA가 없습니다.")
            return
        
        # BERT 형태로 변환
        processed_data = self.process_for_bert(all_qa_pairs)
        
        # 데이터 셔플
        random.shuffle(processed_data)
        
        # 저장
        with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"QA 데이터셋 저장 완료: {len(processed_data)}개")
        
        # 샘플 출력
        for i, sample in enumerate(processed_data[:3]):
            print(f"\n샘플 {i+1}:")
            print(f"질문: {sample['question']}")
            print(f"답변: {sample['answer']}")
            print(f"타입: {sample['type']}")

if __name__ == "__main__":
    processor = RecipeQAProcessor()
    processor.create_qa_dataset()