"""
레시피 챗봇 통합 실행 스크립트
"""
import os
import sys
from config import *

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
    """데이터 전처리 실행"""
    print("🔄 QA 데이터셋 생성 중...")
    try:
        from data_processor import RecipeQAProcessor
        processor = RecipeQAProcessor()
        processor.create_qa_dataset()
        
        if os.path.exists(PROCESSED_DATA_PATH):
            print("✅ QA 데이터셋 생성 완료")
            return True
        else:
            print("❌ QA 데이터셋 생성 실패")
            return False
    except Exception as e:
        print(f"❌ 데이터 전처리 오류: {e}")
        return False

def run_model_training():
    """모델 훈련 실행"""
    print("🔄 모델 훈련 중...")
    try:
        from model_trainer import RecipeQATrainer
        trainer = RecipeQATrainer()
        trainer.train()
        
        if os.path.exists(os.path.join(MODEL_PATH, 'pytorch_model.bin')):
            print("✅ 모델 훈련 완료")
            return True
        else:
            print("❌ 모델 훈련 실패")
            return False
    except Exception as e:
        print(f"❌ 모델 훈련 오류: {e}")
        return False

def run_chatbot_test():
    """챗봇 테스트"""
    print("🔄 챗봇 테스트 중...")
    try:
        from chatbot import RecipeChatbot
        chatbot = RecipeChatbot()
        
        # 테스트 질문들
        test_questions = [
            "안녕하세요",
            "김치찌개 어떻게 만들어?",
            "감자로 뭐 만들 수 있어?"
        ]
        
        print("\n📝 챗봇 테스트 결과:")
        for question in test_questions:
            answer = chatbot.chat(question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
        
        print("✅ 챗봇 테스트 완료")
        return True
    except Exception as e:
        print(f"❌ 챗봇 테스트 오류: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🍳 레시피 챗봇 설정 시작!")
    print("=" * 50)
    
    # 1단계: 데이터 수집
    if not os.path.exists(RAW_DATA_PATH):
        if not run_data_collection():
            print("데이터 수집이 필요합니다. 나중에 data_collector.py를 실행하세요.")
    else:
        print("✅ 레시피 데이터가 이미 존재합니다.")
    
    # 2단계: 데이터 전처리
    if not os.path.exists(PROCESSED_DATA_PATH):
        if not run_data_processing():
            print("QA 데이터 생성이 필요합니다. 나중에 data_processor.py를 실행하세요.")
    else:
        print("✅ QA 데이터셋이 이미 존재합니다.")
    
    # 3단계: 모델 훈련 (선택사항)
    model_exists = os.path.exists(os.path.join(MODEL_PATH, 'pytorch_model.bin'))
    
    if not model_exists:
        train_choice = input("\n모델을 훈련하시겠습니까? (y/n): ").lower().strip()
        if train_choice == 'y':
            run_model_training()
        else:
            print("⏭️ 모델 훈련을 건너뜁니다. (기본 룰 기반 답변 사용)")
    else:
        print("✅ 훈련된 모델이 이미 존재합니다.")
    
    # 4단계: 챗봇 테스트
    if run_chatbot_test():
        print("🎉 모든 설정이 완료되었습니다!")
        
        # 웹 앱 실행 여부 확인
        web_choice = input("\n웹 챗봇을 실행하시겠습니까? (y/n): ").lower().strip()
        if web_choice == 'y':
            print("\n🌐 웹 챗봇을 시작합니다...")
            print("브라우저에서 http://localhost:5000 에 접속하세요!")
            
            # Flask 앱 실행
            os.system("python3 app.py")
        else:
            print("\n수동으로 실행하려면: python3 app.py")
    
    print("\n🍳 레시피 챗봇 설정 완료!")

if __name__ == "__main__":
    main()