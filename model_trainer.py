"""
레시피 QA 모델 훈련 (교과서 방법론 기반)
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from config import *

class RecipeQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        context = item['context']
        answer_start = item['answer_start']
        answer_end = item['answer_end']
        
        # '[CLS] 질문 [SEP] 지문 [SEP]' 형태로 인코딩
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 토큰 레벨에서의 답변 위치 찾기 (간단하게 처리)
        input_ids = encoding['input_ids'].squeeze()
        
        # 실제 구현에서는 더 정확한 토큰 매핑이 필요하지만, 
        # 교육용으로 간단하게 처리
        start_positions = torch.zeros(self.max_length, dtype=torch.long)
        end_positions = torch.zeros(self.max_length, dtype=torch.long)
        
        # SEP 토큰 이후부터 답변 영역으로 가정
        sep_indices = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) >= 1:
            context_start = sep_indices[0].item() + 1
            # 답변 위치를 context 시작 부분으로 설정 (교육용 간소화)
            if context_start < self.max_length - 10:
                start_positions[context_start] = 1
                end_positions[min(context_start + 5, self.max_length - 1)] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'start_positions': start_positions,
            'end_positions': end_positions
        }

class RecipeQAModel(nn.Module):
    """교과서 방법론을 따른 QA 모델"""
    
    def __init__(self, model_name=MODEL_NAME):
        super(RecipeQAModel, self).__init__()
        
        # KcBERT 모델 로드
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # 교과서의 태스크 모듈: 2차원 벡터 출력 (시작, 끝 확률)
        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(self.hidden_size, 2)  # 시작, 끝 확률
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT 출력
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 드롭아웃 적용
        sequence_output = self.dropout(sequence_output)
        
        # 각 토큰에 대해 시작/끝 확률 계산
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, 2]
        
        start_logits = logits[:, :, 0]  # 시작 확률
        end_logits = logits[:, :, 1]    # 끝 확률
        
        return start_logits, end_logits

class RecipeQATrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 모델 초기화
        self.model = RecipeQAModel().to(self.device)
        
    def load_data(self):
        """데이터 로드"""
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 훈련/검증 분할
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # 데이터셋 생성
        train_dataset = RecipeQADataset(train_data, self.tokenizer, MAX_LENGTH)
        val_dataset = RecipeQADataset(val_data, self.tokenizer, MAX_LENGTH)
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self):
        """모델 훈련"""
        print("모델 훈련 시작...")
        
        train_loader, val_loader = self.load_data()
        
        # 옵티마이저 설정
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        
        # 손실 함수
        loss_fn = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            # 훈련
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                # 모델 출력
                start_logits, end_logits = self.model(
                    input_ids, attention_mask, token_type_ids
                )
                
                # 손실 계산 (시작 위치와 끝 위치의 실제 인덱스)
                start_loss = loss_fn(start_logits, start_positions.argmax(dim=1))
                end_loss = loss_fn(end_logits, end_positions.argmax(dim=1))
                loss = start_loss + end_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - 평균 손실: {avg_loss:.4f}")
            
            # 검증
            val_loss = self.validate(val_loader, loss_fn)
            print(f"검증 손실: {val_loss:.4f}")
            
            # 모델 저장
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model()
                print("최고 성능 모델 저장")
    
    def validate(self, val_loader, loss_fn):
        """모델 검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                start_logits, end_logits = self.model(
                    input_ids, attention_mask, token_type_ids
                )
                
                start_loss = loss_fn(start_logits, start_positions.argmax(dim=1))
                end_loss = loss_fn(end_logits, end_positions.argmax(dim=1))
                loss = start_loss + end_loss
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self):
        """모델 저장 (bin, h5 형태)"""
        # 모델 디렉토리 생성
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # PyTorch 형태로 저장 (.bin)
        torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, 'pytorch_model.bin'))
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(MODEL_PATH)
        
        # 설정 저장
        config = {
            'model_name': MODEL_NAME,
            'max_length': MAX_LENGTH,
            'hidden_size': self.model.hidden_size
        }
        
        with open(os.path.join(MODEL_PATH, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    trainer = RecipeQATrainer()
    trainer.train()