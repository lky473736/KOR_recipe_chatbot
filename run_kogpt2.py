"""
KoGPT2 레시피 챗봇 통합 실행 스크립트
"""
import os
import sys
from config_kogpt2 import *

def run_data_collection():
    """데이터 수집 실행"""
    print("🔄 농림축산식품 데이터 수집 중...")
    try:
        from data_collector import RecipeDataCollector
        collector = RecipeDataCollector()
        recipes = collector.collect_all_data()
        
        if recipes:
            print(f"✅ 데이터 수집 완료: {len(recipes)}개 레시피")
            return True
        else:
            print("❌ 데이터 수집 실패")
            return False
    except Exception as e:
        print(f"❌ 데이터 수집 오류: {e}")
        return False

def run_data_processing():
    """KoGPT2 데이터 전처리 실행"""
    print("🔄 KoGPT2 데이터셋 생성 중...")
    try:
        from data_processor_kogpt2 import KoGPT2DataProcessor
        processor = KoGPT2DataProcessor()
        processor.create_qa_dataset()
        
        if os.path.exists(PROCESSED_DATA_PATH):
            print("✅ KoGPT2 데이터셋 생성 완료")
            return True
        else:
            print("❌ KoGPT2 데이터셋 생성 실패")
            return False
    except Exception as e:
        print(f"❌ 데이터 전처리 오류: {e}")
        return False

def run_model_training():
    """KoGPT2 모델 훈련 실행"""
    print("🔄 KoGPT2 모델 훈련 중...")
    try:
        from model_trainer_kogpt2 import KoGPT2Trainer
        trainer = KoGPT2Trainer()
        trainer.train()
        
        # 모델 파일 존재 확인
        model_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        pytorch_model = os.path.join(MODEL_PATH, 'pytorch_model.bin')
        safetensors_model = os.path.join(MODEL_PATH, 'model.safetensors')
        
        files_exist = all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
        model_exists = os.path.exists(pytorch_model) or os.path.exists(safetensors_model)
        
        if files_exist and model_exists:
            print("✅ KoGPT2 모델 훈련 완료")
            return True
        else:
            print("❌ KoGPT2 모델 훈련 실패")
            return False
    except Exception as e:
        print(f"❌ 모델 훈련 오류: {e}")
        return False

def run_chatbot_test():
    """KoGPT2 챗봇 테스트"""
    print("🔄 KoGPT2 챗봇 테스트 중...")
    try:
        from chatbot_kogpt2 import KoGPT2RecipeChatbot
        chatbot = KoGPT2RecipeChatbot()
        
        # 테스트 질문들
        test_questions = [
            "안녕하세요",
            "김치찌개 어떻게 만들어?",
            "감자로 뭐 만들 수 있어?"
        ]
        
        print("\n📝 KoGPT2 챗봇 테스트 결과:")
        print("-" * 50)
        for question in test_questions:
            answer = chatbot.chat(question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
        
        print("✅ KoGPT2 챗봇 테스트 완료")
        print(f"📊 챗봇 상태:")
        print(f"  - 모델 로드됨: {chatbot.model_loaded}")
        print(f"  - 훈련된 모델: {chatbot.is_trained_model}")
        print(f"  - 레시피 수: {len(chatbot.recipes)}")
        
        return True
    except Exception as e:
        print(f"❌ KoGPT2 챗봇 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_requirements():
    """필요한 패키지 확인"""
    print("🔍 패키지 확인 중...")
    
    required_packages = [
        'torch', 'transformers', 'flask', 'requests', 
        'scikit-learn', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지: {missing_packages}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ 모든 필요한 패키지가 설치되어 있습니다")
        return True

def main():
    """메인 실행 함수"""
    print("🍳 KoGPT2 레시피 챗봇 설정 시작!")
    print("=" * 60)
    
    # 패키지 확인
    if not check_requirements():
        print("\n❌ 필요한 패키지를 먼저 설치해주세요.")
        return
    
    # 1단계: 데이터 수집
    if not os.path.exists(RAW_DATA_PATH):
        print("\n📋 1단계: 데이터 수집")
        if not run_data_collection():
            print("⚠️ 데이터 수집이 필요합니다. 나중에 data_collector.py를 실행하세요.")
        print()
    else:
        print("✅ 레시피 데이터가 이미 존재합니다.")
    
    # 2단계: 데이터 전처리
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("📋 2단계: KoGPT2 데이터 전처리")
        if not run_data_processing():
            print("⚠️ KoGPT2 데이터 생성이 필요합니다.")
            print("나중에 data_processor_kogpt2.py를 실행하세요.")
        print()
    else:
        print("✅ KoGPT2 데이터셋이 이미 존재합니다.")
    
    # 3단계: 모델 훈련 (선택사항)
    print("📋 3단계: KoGPT2 모델 훈련")
    
    # 모델 파일 존재 확인 (더 정확하게)
    model_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    pytorch_model = os.path.join(MODEL_PATH, 'pytorch_model.bin')
    safetensors_model = os.path.join(MODEL_PATH, 'model.safetensors')
    
    files_exist = all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
    model_exists = os.path.exists(pytorch_model) or os.path.exists(safetensors_model)
    
    if not (files_exist and model_exists):
        print("현재 훈련된 모델이 없습니다.")
        train_choice = input("KoGPT2 모델을 훈련하시겠습니까? (y/n): ").lower().strip()
        if train_choice == 'y':
            if run_model_training():
                print("🎉 모델 훈련이 완료되었습니다!")
            else:
                print("❌ 모델 훈련에 실패했습니다.")
        else:
            print("⏭️ 모델 훈련을 건너뜁니다. (기본 모델 + 룰 기반 답변 사용)")
    else:
        print("✅ 훈련된 KoGPT2 모델이 이미 존재합니다.")
    
    print()
    
    # 4단계: 챗봇 테스트
    print("📋 4단계: KoGPT2 챗봇 테스트")
    if run_chatbot_test():
        print("🎉 모든 설정이 완료되었습니다!")
        
        # 웹 앱 실행 여부 확인
        web_choice = input("\nKoGPT2 웹 챗봇을 실행하시겠습니까? (y/n): ").lower().strip()
        if web_choice == 'y':
            print("\n🌐 KoGPT2 웹 챗봇을 시작합니다...")
            print("브라우저에서 http://localhost:5000 에 접속하세요!")
            print("(종료하려면 Ctrl+C)")
            
            # Flask 앱 실행
            try:
                os.system("python app.py")
            except KeyboardInterrupt:
                print("\n👋 챗봇을 종료합니다.")
        else:
            print("\n수동으로 실행하려면: python app.py")
    else:
        print("❌ 챗봇 테스트에 실패했습니다.")
    
    print("\n🍳 KoGPT2 레시피 챗봇 설정 완료!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()