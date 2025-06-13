# 모듈 6: 소형 LLM 학습 및 도메인 어댑테이션

## 학습 목표
- Full 파인튜닝과 PEFT(LoRA)의 차이점을 이해한다
- 도메인 특화 데이터로 LLM을 효율적으로 학습한다
- 실제 프로젝트에서 LoRA를 활용한 모델 경량화를 수행한다

## 1. LLM 학습 방법 비교

### 1.1 Full 파인튜닝
- 전체 모델 파라미터 업데이트
- 높은 컴퓨팅 자원 요구
- 큰 저장 공간 필요
- 일반적으로 더 나은 성능

### 1.2 PEFT (Parameter-Efficient Fine-Tuning)
- 일부 파라미터만 업데이트
- 적은 컴퓨팅 자원
- 작은 저장 공간
- 효율적인 학습 가능

### 1.3 LoRA (Low-Rank Adaptation)
- 저순위 분해를 통한 효율적 학습
- 기존 가중치는 동결
- 적응 행렬만 학습
- 빠른 학습과 적은 메모리 사용

## 2. 실습 예제

### 2.1 데이터 준비
```python
# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

class DataPreparator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, filepath):
        """데이터 로드 및 기본 전처리"""
        df = pd.read_csv(filepath)
        return df
    
    def clean_text(self, text):
        """텍스트 정제"""
        # 기본 정제 작업
        text = str(text)
        text = text.strip()
        text = ' '.join(text.split())
        return text
    
    def prepare_dataset(self, df, text_column, label_column=None):
        """데이터셋 준비"""
        # 텍스트 정제
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # 데이터셋 분할
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # HuggingFace 데이터셋 형식으로 변환
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """토큰화 함수"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

# 사용 예시
def prepare_data_example():
    from transformers import AutoTokenizer
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    
    # 데이터 준비 객체 생성
    preparator = DataPreparator(tokenizer)
    
    # 샘플 데이터 생성
    sample_data = {
        'text': [
            '이 제품은 품질이 매우 좋습니다.',
            '배송이 너무 늦어요.',
            '가격대비 성능이 훌륭해요.'
        ],
        'label': [1, 0, 1]
    }
    df = pd.DataFrame(sample_data)
    
    # 데이터셋 준비
    train_dataset, val_dataset = preparator.prepare_dataset(df, 'text', 'label')
    
    return train_dataset, val_dataset
```

### 2.2 LoRA 구현
```python
# lora_training.py

import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

class LoRATrainer:
    def __init__(
        self, 
        base_model_name,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 기본 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16
        )
        
        # LoRA 설정
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # LoRA 모델 생성
        self.model = get_peft_model(self.base_model, self.peft_config)
    
    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4
    ):
        """모델 학습"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=100,
            logging_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=3
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        
        return trainer
    
    def save_model(self, output_dir):
        """모델 저장"""
        self.model.save_pretrained(output_dir)
    
    def evaluate(self, test_dataset):
        """모델 평가"""
        trainer = Trainer(
            model=self.model,
            eval_dataset=test_dataset
        )
        
        metrics = trainer.evaluate()
        return metrics

# 사용 예시
def train_model_example():
    # 데이터 준비
    train_dataset, val_dataset = prepare_data_example()
    
    # LoRA 트레이너 초기화
    trainer = LoRATrainer("beomi/kcbert-base")
    
    # 모델 학습
    trained_model = trainer.train(
        train_dataset,
        val_dataset,
        "output/lora_model"
    )
    
    # 모델 저장
    trainer.save_model("output/lora_model")
    
    # 평가
    metrics = trainer.evaluate(val_dataset)
    print("Evaluation metrics:", metrics)
```

### 2.3 성능 비교 도구
```python
# model_comparison.py

import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelComparator:
    def __init__(self):
        self.results = {}
    
    def measure_memory(self, model):
        """모델 메모리 사용량 측정"""
        torch.cuda.empty_cache()
        
        # GPU 메모리 측정
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            gpu_memory = 0
            
        # CPU 메모리 측정
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**2  # MB
        
        return {
            "gpu_memory_mb": gpu_memory,
            "cpu_memory_mb": cpu_memory
        }
    
    def measure_inference_time(self, model, tokenizer, text, num_runs=100):
        """추론 시간 측정"""
        inputs = tokenizer(text, return_tensors="pt")
        
        # 워밍업
        model.generate(**inputs, max_length=50)
        
        # 시간 측정
        start_time = time.time()
        for _ in range(num_runs):
            model.generate(**inputs, max_length=50)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def compare_models(self, base_model_path, lora_model_path, test_text):
        """모델 비교"""
        # 기본 모델
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        base_results = {
            "memory": self.measure_memory(base_model),
            "inference_time": self.measure_inference_time(
                base_model, tokenizer, test_text
            )
        }
        
        # LoRA 모델
        lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path)
        
        lora_results = {
            "memory": self.measure_memory(lora_model),
            "inference_time": self.measure_inference_time(
                lora_model, tokenizer, test_text
            )
        }
        
        return {
            "base_model": base_results,
            "lora_model": lora_results
        }
    
    def print_comparison(self, results):
        """비교 결과 출력"""
        print("=== 모델 비교 결과 ===")
        print("\n1. 메모리 사용량")
        print("기본 모델:")
        print(f"- GPU: {results['base_model']['memory']['gpu_memory_mb']:.2f} MB")
        print(f"- CPU: {results['base_model']['memory']['cpu_memory_mb']:.2f} MB")
        print("\nLoRA 모델:")
        print(f"- GPU: {results['lora_model']['memory']['gpu_memory_mb']:.2f} MB")
        print(f"- CPU: {results['lora_model']['memory']['cpu_memory_mb']:.2f} MB")
        
        print("\n2. 추론 시간")
        print(f"기본 모델: {results['base_model']['inference_time']*1000:.2f} ms")
        print(f"LoRA 모델: {results['lora_model']['inference_time']*1000:.2f} ms")

# 사용 예시
def compare_models_example():
    comparator = ModelComparator()
    
    results = comparator.compare_models(
        "beomi/kcbert-base",
        "output/lora_model",
        "이 제품의 특징을 설명해주세요."
    )
    
    comparator.print_comparison(results)
```

## 3. 실습 과제

### 3.1 도메인 데이터 수집
1. 데이터 수집 계획 수립
   - 목표 도메인 선정
   - 필요 데이터 양 결정
   - 수집 방법 선택

2. 데이터 정제
   - 중복 제거
   - 노이즈 제거
   - 포맷 통일

3. 데이터셋 구성
   - 학습/검증/테스트 분할
   - 레이블링 (필요한 경우)
   - 데이터 증강 (필요한 경우)

### 3.2 LoRA 학습
1. 기본 모델 선택
   - 모델 크기 고려
   - 도메인 적합성 검토
   - 리소스 요구사항 확인

2. LoRA 하이퍼파라미터 설정
   - rank (r) 설정
   - alpha 값 조정
   - learning rate 최적화

3. 학습 실행
   - 학습 모니터링
   - 체크포인트 저장
   - 성능 평가

### 3.3 성능 평가
1. 정량적 평가
   - 정확도/손실 측정
   - 추론 속도 비교
   - 메모리 사용량 분석

2. 정성적 평가
   - 출력 품질 평가
   - 도메인 적합성 검토
   - 사용자 피드백 수집

## 참고 자료
- [PEFT 문서](https://huggingface.co/docs/peft)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PyTorch 문서](https://pytorch.org/docs)

## 추가 리소스
- 예제 코드 저장소
- 데이터셋 샘플
- 하이퍼파라미터 설정 가이드
- 트러블슈팅 가이드
