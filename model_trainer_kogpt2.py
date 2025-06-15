"""
KoGPT2 모델 훈련 (패딩 토큰 오류 완전 해결)
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config_kogpt2 import *

class KoGPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 토크나이저가 이미 설정되어 있는지 확인
        print(f"Dataset - 패딩 토큰: {self.tokenizer.pad_token}")
        print(f"Dataset - 패딩 토큰 ID: {self.tokenizer.pad_token_id}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 토크나이징 (더 안전한 방법)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # GPT2는 input_ids와 labels가 동일
        labels = input_ids.clone()
        
        # 패딩 토큰에 대해서는 loss 계산 안함
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class KoGPT2Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
        
        # 패딩 토큰 강제 설정 (여러 방법 시도)
        print("토크나이저 패딩 토큰 설정 중...")
        
        # 방법 1: EOS 토큰을 패딩으로 사용
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 방법 2: 새로운 패딩 토큰 추가 (위 방법이 안되면)
        if self.tokenizer.pad_token is None:
            special_tokens_dict = {'pad_token': '<pad>'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print("새로운 패딩 토큰 <pad> 추가")
        
        # 최종 확인
        print(f"✅ 패딩 토큰 설정 완료:")
        print(f"  - pad_token: {self.tokenizer.pad_token}")
        print(f"  - pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"  - eos_token: {self.tokenizer.eos_token}")
        print(f"  - eos_token_id: {self.tokenizer.eos_token_id}")
        print(f"  - vocab_size: {len(self.tokenizer)}")
        
        # 모델 로드
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
        # 토큰 수가 변경되었으면 임베딩 크기 조정
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        print(f"모델 로드 완료: {MODEL_NAME}")
        
        # 간단한 토크나이징 테스트
        self.test_tokenizer()
    
    def test_tokenizer(self):
        """토크나이저 테스트"""
        test_text = "질문: 김치찌개 어떻게 만들어?\n답변: 김치와 돼지고기를 볶으세요.<|endoftext|>"
        
        try:
            encoding = self.tokenizer(
                test_text,
                truncation=True,
                padding='max_length',
                max_length=50,
                return_tensors='pt'
            )
            print("✅ 토크나이저 테스트 성공!")
            print(f"  - input_ids shape: {encoding['input_ids'].shape}")
            print(f"  - attention_mask shape: {encoding['attention_mask'].shape}")
            
        except Exception as e:
            print(f"❌ 토크나이저 테스트 실패: {e}")
            raise e
    
    def load_data(self):
        """데이터 로드"""
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"전체 데이터: {len(data)}개")
        
        # 데이터 유효성 검사 및 길이 제한
        valid_data = []
        for item in data:
            text = item.get('text', '')
            if text and len(text.strip()) > 10:
                # 너무 긴 텍스트는 자르기
                if len(text) > 1000:
                    text = text[:1000] + END_TOKEN
                    item['text'] = text
                valid_data.append(item)
        
        print(f"유효한 데이터: {len(valid_data)}개")
        
        if len(valid_data) < 10:
            raise ValueError("유효한 데이터가 너무 적습니다!")
        
        # 훈련/검증 분할
        train_data, val_data = train_test_split(valid_data, test_size=0.1, random_state=42)
        
        # 데이터셋 생성 (이미 설정된 토크나이저 전달)
        train_dataset = KoGPT2Dataset(train_data, self.tokenizer, MAX_LENGTH)
        val_dataset = KoGPT2Dataset(val_data, self.tokenizer, MAX_LENGTH)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0,
            pin_memory=False  # 메모리 문제 방지
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"훈련: {len(train_data)}개, 검증: {len(val_data)}개")
        print(f"배치 수: 훈련 {len(train_loader)}, 검증 {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self):
        """모델 훈련"""
        print("KoGPT2 모델 훈련 시작...")
        
        train_loader, val_loader = self.load_data()
        
        # 옵티마이저
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        best_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n📖 Epoch {epoch+1}/{NUM_EPOCHS} 시작")
            
            # 훈련
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 디버그: 첫 번째 배치에서 확인
                    if batch_idx == 0 and epoch == 0:
                        print(f"\n🔍 첫 번째 배치 정보:")
                        print(f"  - input_ids shape: {input_ids.shape}")
                        print(f"  - attention_mask shape: {attention_mask.shape}")
                        print(f"  - labels shape: {labels.shape}")
                    
                    # 모델 출력
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # NaN 체크
                    if torch.isnan(loss):
                        print(f"⚠️ NaN loss 발생 (배치 {batch_idx})")
                        continue
                    
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/num_batches:.4f}'
                    })
                    
                except Exception as e:
                    print(f"❌ 배치 {batch_idx} 처리 오류: {e}")
                    continue
            
            if num_batches > 0:
                avg_train_loss = total_loss / num_batches
                print(f"\n📊 Epoch {epoch+1} 결과:")
                print(f"  - 평균 훈련 손실: {avg_train_loss:.4f}")
                
                # 검증
                val_loss = self.validate(val_loader)
                print(f"  - 검증 손실: {val_loss:.4f}")
                
                # 모델 저장
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model()
                    print("  - ✅ 최고 성능 모델 저장!")
                else:
                    print(f"  - 성능 개선 없음 (최고: {best_loss:.4f})")
            else:
                print("⚠️ 유효한 배치가 없었습니다")
            
            print("-" * 60)
    
    def validate(self, val_loader):
        """모델 검증"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    if not torch.isnan(outputs.loss):
                        total_loss += outputs.loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def save_model(self):
        """모델 저장"""
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # 모델과 토크나이저 저장
        self.model.save_pretrained(MODEL_PATH)
        self.tokenizer.save_pretrained(MODEL_PATH)
        
        print(f"💾 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    try:
        trainer = KoGPT2Trainer()
        trainer.train()
        print("\n🎉 훈련 완료!")
    except Exception as e:
        print(f"\n❌ 훈련 실패: {e}")
        import traceback
        traceback.print_exc()