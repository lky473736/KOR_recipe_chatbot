"""
레시피 챗봇 설정 파일 (KoGPT2 버전) - 개선됨
"""
import os

# 농림축산식품 API 설정
MAFRA_API_KEY = "c43f9e43df898ac83c17fecf1abcd3e0af0bf29087be02128cf82a9e8679c90c"
MAFRA_BASE_URL = "http://211.237.50.150:7080/openapi"

# API 서비스 ID
RECIPE_BASIC_SERVICE = "Grid_20150827000000000226_1"
RECIPE_INGREDIENT_SERVICE = "Grid_20150827000000000227_1"
RECIPE_PROCESS_SERVICE = "Grid_20150827000000000228_1"

# KoGPT2 모델 설정
MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LENGTH = 512
BATCH_SIZE = 4  # GPU 메모리에 따라 조절
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
GENERATION_MAX_LENGTH = 150  # 답변 생성 최대 길이

# 특수 토큰
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

# 생성 설정 (개선됨)
GENERATION_CONFIG = {
    'max_length': GENERATION_MAX_LENGTH,
    'temperature': 0.8,  # 조금 더 창의적으로
    'top_p': 0.85,      # 더 집중된 선택
    'top_k': 40,        # 적절한 다양성
    'repetition_penalty': 1.3,  # 반복 방지 강화
    'do_sample': True,
    'pad_token_id': 1,  # 기본값, 런타임에서 재설정됨
    'eos_token_id': 1   # 기본값, 런타임에서 재설정됨
}

# 디버그 설정
DEBUG = True
VERBOSE_LOGGING = True

# 성능 설정
TORCH_SEED = 42
USE_MIXED_PRECISION = False  # GPU 메모리가 충분하지 않으면 True로 설정

print(f"📁 설정 로드 완료:")
print(f"  - 데이터 디렉토리: {DATA_DIR}")
print(f"  - 모델 디렉토리: {MODEL_DIR}")
print(f"  - 모델 경로: {MODEL_PATH}")
print(f"  - 원본 데이터: {RAW_DATA_PATH}")
print(f"  - 전처리 데이터: {PROCESSED_DATA_PATH}")