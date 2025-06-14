"""
레시피 챗봇 설정 파일
"""
import os

# 농림축산식품 API 설정
MAFRA_API_KEY = "c43f9e43df898ac83c17fecf1abcd3e0af0bf29087be02128cf82a9e8679c90c"
MAFRA_BASE_URL = "http://211.237.50.150:7080/openapi"

# API 서비스 ID
RECIPE_BASIC_SERVICE = "Grid_20150827000000000226_1"      # 레시피 기본정보
RECIPE_INGREDIENT_SERVICE = "Grid_20150827000000000227_1"  # 레시피 재료정보
RECIPE_PROCESS_SERVICE = "Grid_20150827000000000228_1"     # 레시피 과정정보

# 모델 설정
MODEL_NAME = "beomi/kcbert-base"
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# 디렉토리 설정
DATA_DIR = "data"
MODEL_DIR = "models"
TEMPLATES_DIR = "templates"

# 파일 경로
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_recipes.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "qa_dataset.json")
MODEL_PATH = os.path.join(MODEL_DIR, "recipe_qa_model")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)