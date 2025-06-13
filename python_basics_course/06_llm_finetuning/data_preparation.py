import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

class DataPreparator:
    def __init__(self, tokenizer, max_length=512):
        """데이터 준비 클래스 초기화

        Args:
            tokenizer: 사용할 토크나이저
            max_length (int): 최대 시퀀스 길이
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, filepath):
        """데이터 로드 및 기본 전처리

        Args:
            filepath (str): 데이터 파일 경로

        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        df = pd.read_csv(filepath)
        return df
    
    def clean_text(self, text):
        """텍스트 정제

        Args:
            text (str): 정제할 텍스트

        Returns:
            str: 정제된 텍스트
        """
        # 기본 정제 작업
        text = str(text)
        text = text.strip()
        text = ' '.join(text.split())
        return text
    
    def prepare_dataset(self, df, text_column, label_column=None):
        """데이터셋 준비

        Args:
            df (pd.DataFrame): 원본 데이터프레임
            text_column (str): 텍스트 컬럼명
            label_column (str, optional): 레이블 컬럼명

        Returns:
            tuple: (학습용 데이터셋, 검증용 데이터셋)
        """
        # 텍스트 정제
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # 데이터셋 분할
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # HuggingFace 데이터셋 형식으로 변환
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # 토큰화
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """토큰화 함수

        Args:
            examples (dict): 텍스트 예시들

        Returns:
            dict: 토큰화된 결과
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
    
    def create_dummy_dataset(self, size=1000):
        """테스트용 더미 데이터셋 생성

        Args:
            size (int): 생성할 데이터 크기

        Returns:
            pd.DataFrame: 생성된 더미 데이터프레임
        """
        # 샘플 텍스트 템플릿
        templates = [
            "이 제품은 {}한 특징이 있으며, {}한 장점이 있습니다.",
            "서비스 품질이 {}하고 {}합니다.",
            "사용해본 결과 {}하며 {}한 것 같습니다."
        ]
        
        # 형용사 리스트
        adjectives = [
            "우수", "훌륭", "뛰어난", "독특", "특별", "효과적",
            "편리", "실용적", "혁신적", "안정적", "효율적", "신뢰할 만"
        ]
        
        texts = []
        labels = []
        
        for _ in range(size):
            template = np.random.choice(templates)
            adj1 = np.random.choice(adjectives)
            adj2 = np.random.choice(adjectives)
            
            text = template.format(adj1, adj2)
            label = np.random.randint(0, 2)  # 이진 분류를 위한 레이블
            
            texts.append(text)
            labels.append(label)
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })

def example_usage():
    """사용 예시"""
    from transformers import AutoTokenizer
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    
    # 데이터 준비 객체 생성
    preparator = DataPreparator(tokenizer)
    
    # 더미 데이터 생성
    dummy_df = preparator.create_dummy_dataset()
    
    # 데이터셋 준비
    train_dataset, val_dataset = preparator.prepare_dataset(
        dummy_df, 'text', 'label'
    )
    
    print("학습 데이터셋 크기:", len(train_dataset))
    print("검증 데이터셋 크기:", len(val_dataset))
    print("\n데이터셋 샘플:")
    print(train_dataset[0])

if __name__ == "__main__":
    example_usage()
