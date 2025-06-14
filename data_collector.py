"""
농림축산식품 공공데이터 수집기
"""
import requests
import json
import time
from config import *

class RecipeDataCollector:
    def __init__(self):
        self.api_key = MAFRA_API_KEY
        self.base_url = MAFRA_BASE_URL
        self.session = requests.Session()
        
        # 요청 헤더 설정
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/json, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
    def build_url(self, service_id, start_idx=1, end_idx=1000):
        """API URL 생성"""
        return f"{self.base_url}/{self.api_key}/json/{service_id}/{start_idx}/{end_idx}"
    
    def test_api_connection(self):
        """API 연결 테스트"""
        test_url = self.build_url(RECIPE_BASIC_SERVICE, 1, 5)
        try:
            print(f"API 연결 테스트...")
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if RECIPE_BASIC_SERVICE in data and 'row' in data[RECIPE_BASIC_SERVICE]:
                    print("✅ API 연결 성공")
                    return True
            return False
        except:
            return False
    
    def fetch_data(self, service_id, max_records=1000):
        """데이터 가져오기"""
        all_data = []
        
        if service_id == RECIPE_BASIC_SERVICE:
            # 기본정보는 한 번에
            url = self.build_url(service_id, 1, max_records)
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if service_id in data and 'row' in data[service_id]:
                    rows = data[service_id]['row']
                    all_data = rows if isinstance(rows, list) else [rows]
                    
            except Exception as e:
                print(f"기본정보 수집 실패: {e}")
        
        else:
            # 재료정보와 과정정보는 분할 수집
            for start_idx in range(1, max_records + 1, 1000):
                end_idx = min(start_idx + 999, max_records)
                url = self.build_url(service_id, start_idx, end_idx)
                
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if service_id in data and 'row' in data[service_id]:
                        rows = data[service_id]['row']
                        batch_data = rows if isinstance(rows, list) else [rows]
                        all_data.extend(batch_data)
                    else:
                        break  # 더 이상 데이터 없음
                        
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"배치 {start_idx}-{end_idx} 실패: {e}")
                    break
                    
        return all_data
    
    def collect_all_data(self):
        """모든 레시피 데이터 수집"""
        print("농림축산식품 레시피 데이터 수집 시작...")
        
        # API 연결 테스트
        if not self.test_api_connection():
            print("❌ API 연결 실패")
            return []
        
        # 기본정보 수집
        basic_data = self.fetch_data(RECIPE_BASIC_SERVICE, 1000)
        print(f"기본정보: {len(basic_data)}개")
        
        # 재료정보 수집 (더 많이)
        ingredient_data = self.fetch_data(RECIPE_INGREDIENT_SERVICE, 7000)
        print(f"재료정보: {len(ingredient_data)}개")
        
        # 과정정보 수집 (더 많이)
        process_data = self.fetch_data(RECIPE_PROCESS_SERVICE, 4000)
        print(f"과정정보: {len(process_data)}개")
        
        # 데이터 통합
        recipes = self.integrate_data(basic_data, ingredient_data, process_data)
        
        # 저장
        with open(RAW_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)
        
        print(f"데이터 저장 완료: {len(recipes)}개 레시피")
        return recipes
    
    def integrate_data(self, basic_data, ingredient_data, process_data):
        """데이터 통합"""
        recipes = {}
        
        # 기본정보 처리
        for item in basic_data:
            recipe_id = item.get('RECIPE_ID', '')
            if recipe_id:
                recipes[recipe_id] = {
                    'id': recipe_id,
                    'name': item.get('RECIPE_NM_KO', ''),
                    'category': item.get('RECIPE_TY_NM', ''),
                    'cooking_method': item.get('COOKING_MTH_NM', ''),
                    'ingredients': [],
                    'steps': []
                }
        
        # 재료정보 추가
        for item in ingredient_data:
            recipe_id = item.get('RECIPE_ID', '')
            if recipe_id in recipes:
                ingredient = item.get('IRDNT_NM', '')
                if ingredient:
                    recipes[recipe_id]['ingredients'].append(ingredient)
        
        # 과정정보 추가
        process_dict = {}
        for item in process_data:
            recipe_id = item.get('RECIPE_ID', '')
            if recipe_id in recipes:
                step_no = int(item.get('COOKING_NO', 0))
                step_desc = item.get('COOKING_DC', '')
                if step_desc:
                    if recipe_id not in process_dict:
                        process_dict[recipe_id] = {}
                    process_dict[recipe_id][step_no] = step_desc
        
        # 과정을 순서대로 정렬하여 추가
        for recipe_id, steps in process_dict.items():
            if recipe_id in recipes:
                sorted_steps = [steps[i] for i in sorted(steps.keys())]
                recipes[recipe_id]['steps'] = sorted_steps
        
        # 유효한 레시피만 반환
        valid_recipes = []
        for recipe in recipes.values():
            if (recipe['name'] and 
                len(recipe['ingredients']) >= 1 and 
                len(recipe['steps']) >= 1):
                valid_recipes.append(recipe)
        
        return valid_recipes

if __name__ == "__main__":
    collector = RecipeDataCollector()
    collector.collect_all_data()