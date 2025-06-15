"""
KoGPT2 모델 훈련
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config_kogpt2 import *

class KoGPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # GPT2는 input_ids와 labels가 동일 (다음 토큰 예측)
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
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        
        print(f"모델 로드 완료: {MODEL_NAME}")
    
    def load_data(self):
        """데이터 로드"""
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"전체 데이터: {len(data)}개")
        
        # 훈련/검증 분할
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        
        # 데이터셋 생성
        train_dataset = KoGPT2Dataset(train_data, self.tokenizer, MAX_LENGTH)
        val_dataset = KoGPT2Dataset(val_data, self.tokenizer, MAX_LENGTH)
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"훈련: {len(train_data)}개, 검증: {len(val_data)}개")
        
        return train_loader, val_loader
    
    def train(self):
        """모델 훈련"""
        print("KoGPT2 모델 훈련 시작...")
        
        train_loader, val_loader = self.load_data()
        
        # 옵티마이저
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        best_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            # 훈련
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 모델 출력
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - 평균 훈련 손실: {avg_train_loss:.4f}")
            
            # 검증
            val_loss = self.validate(val_loader)
            print(f"검증 손실: {val_loss:.4f}")
            
            # 모델 저장
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model()
                print("✅ 최고 성능 모델 저장")
            
            print("-" * 50)
    
    def validate(self, val_loader):
        """모델 검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self):
        """모델 저장"""
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # 모델과 토크나이저 저장
        self.model.save_pretrained(MODEL_PATH)
        self.tokenizer.save_pretrained(MODEL_PATH)
        
        print(f"모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    trainer = KoGPT2Trainer()
    trainer.train()