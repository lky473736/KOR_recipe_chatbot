"""
레시피 챗봇 설정 파일 (KoGPT2 버전)
"""
import os

# 농림축산식품 API 설정
MAFRA_API_KEY = "c43f9e43df898ac83c17fecf1abcd3e0af0bf29087be02128cf82a9e8679c90c"
MAFRA_BASE_URL = "http://211.237.50.150:7080/openapi"

# API 서비스 ID
RECIPE_BASIC_SERVICE = "Grid_20150827000000000226_1"      # 레시피 기본정보
RECIPE_INGREDIENT_SERVICE = "Grid_20150827000000000227_1"  # 레시피 재료정보
RECIPE_PROCESS_SERVICE = "Grid_20150827000000000228_1"     # 레시피 과정정보

# KoGPT2 모델 설정
MODEL_NAME = "skt/kogpt2-base-v2"  # 또는 "kakaobrain/kogpt"
MAX_LENGTH = 512  # 생성형이므로 좀 더 길게
BATCH_SIZE = 8    # GPT2는 메모리를 많이 사용하므로 작게
LEARNING_RATE = 5e-5  # GPT2는 보통 조금 더 높은 LR 사용
NUM_EPOCHS = 5
GENERATION_MAX_LENGTH = 150  # 답변 생성 최대 길이

# 특수 토큰들
PROMPT_FORMAT = "질문: {question}\n답변:"
END_TOKEN = "<|endoftext|>"

# 디렉토리 설정
DATA_DIR = "data"
MODEL_DIR = "models_kogpt2"
TEMPLATES_DIR = "templates"

# 파일 경로
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_recipes.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "kogpt2_dataset.json")
MODEL_PATH = os.path.join(MODEL_DIR, "recipe_kogpt2_model")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 생성 설정
GENERATION_CONFIG = {
    'max_length': GENERATION_MAX_LENGTH,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.2,
    'do_sample': True,
    'pad_token_id': 0,
    'eos_token_id': 1
}