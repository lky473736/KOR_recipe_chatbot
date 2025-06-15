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
        
        # 일반 QA 추가 (더 길고 자세하게)
        self.general_qa = [
            {
                'question': '안녕하세요',
                'context': '레시피 챗봇 인사말: 안녕하세요! 농림축산식품부 공공데이터 기반 레시피 챗봇입니다. 537개의 다양한 한국 요리 레시피를 알고 있어서, 요리 방법이나 재료에 대해 무엇이든 물어보세요! 맛있는 요리 만들기를 도와드릴게요',
                'answer': '레시피 챗봇 인사말: 안녕하세요! 농림축산식품부 공공데이터 기반 레시피 챗봇입니다. 537개의 다양한 한국 요리 레시피를 알고 있어서, 요리 방법이나 재료에 대해 무엇이든 물어보세요! 맛있는 요리 만들기를 도와드릴게요',
                'type': 'greeting'
            },
            {
                'question': '안녕',
                'context': '레시피 챗봇 인사말: 안녕하세요! 오늘은 어떤 요리를 만들어보고 싶으신가요? 한국의 전통 요리부터 간단한 집밥까지 다양한 레시피를 알려드릴 수 있어요. 편하게 물어보세요!',
                'answer': '레시피 챗봇 인사말: 안녕하세요! 오늘은 어떤 요리를 만들어보고 싶으신가요? 한국의 전통 요리부터 간단한 집밥까지 다양한 레시피를 알려드릴 수 있어요. 편하게 물어보세요!',
                'type': 'greeting'
            },
            {
                'question': '도움말',
                'context': '레시피 챗봇 사용법: 1. 재료로 요리 검색 - 감자로 뭐 만들 수 있어? 2. 특정 요리 만드는 법 - 김치찌개 어떻게 만들어? 3. 요리 재료 확인 - 불고기 재료가 뭐야? 이런 식으로 자연스럽게 물어보시면 친절하게 답변해드릴게요!',
                'answer': '레시피 챗봇 사용법: 1. 재료로 요리 검색 - 감자로 뭐 만들 수 있어? 2. 특정 요리 만드는 법 - 김치찌개 어떻게 만들어? 3. 요리 재료 확인 - 불고기 재료가 뭐야? 이런 식으로 자연스럽게 물어보시면 친절하게 답변해드릴게요!',
                'type': 'help'
            },
            {
                'question': '뭐 해줄 수 있어',
                'context': '레시피 챗봇 기능: 저는 요리와 관련해서 많은 도움을 드릴 수 있어요! 특정 재료로 만들 수 있는 요리 추천, 요리 만드는 방법 상세 설명, 필요한 재료 목록 안내, 요리 카테고리별 추천 등을 할 수 있습니다. 537개의 한국 요리 레시피가 준비되어 있으니 언제든 물어보세요!',
                'answer': '레시피 챗봇 기능: 저는 요리와 관련해서 많은 도움을 드릴 수 있어요! 특정 재료로 만들 수 있는 요리 추천, 요리 만드는 방법 상세 설명, 필요한 재료 목록 안내, 요리 카테고리별 추천 등을 할 수 있습니다. 537개의 한국 요리 레시피가 준비되어 있으니 언제든 물어보세요!',
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
        
        # 조리 과정을 context로 만들기 (더 상세하게)
        context = f"{recipe_name} 만드는 방법: " + " ".join(steps)
        
        for template in self.qa_templates['cooking_method']:
            question = template.format(recipe_name=recipe_name)
            
            # context 전체를 answer로 사용!
            answer = context
            
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
        
        # 재료 목록을 context로 만들기 (모든 재료 포함)
        ingredients_text = ", ".join(ingredients)
        context = f"{recipe_name}의 재료: {ingredients_text}"
        
        for template in self.qa_templates['ingredients']:
            question = template.format(recipe_name=recipe_name)
            
            # context 전체를 answer로 사용!
            answer = context
            
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
                
                # context는 해당 재료로 만들 수 있는 요리들 (더 많이 포함)
                context = f"{ingredient}로 만들 수 있는 요리: " + ", ".join(unique_recipes)
                
                for template in self.qa_templates['recipe_search']:
                    question = template.format(ingredient=ingredient)
                    
                    # context 전체를 answer로 사용!
                    answer = context
                    
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
            
            # 답변의 시작과 끝 위치 찾기 (context = answer이므로 context 시작 위치)
            context_start = input_text.find(context)
            
            if context_start != -1:
                # context 전체가 답변이므로
                actual_answer_start = context_start
                actual_answer_end = context_start + len(context) - 1
                
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
            print(f"답변: {sample['answer'][:100]}..." if len(sample['answer']) > 100 else f"답변: {sample['answer']}")
            print(f"타입: {sample['type']}")

if __name__ == "__main__":
    processor = RecipeQAProcessor()
    processor.create_qa_dataset()