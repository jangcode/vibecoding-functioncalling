import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    gpu_memory: float
    cpu_memory: float
    inference_time: float
    model_size: float
    perplexity: float = 0.0

class ModelComparator:
    def __init__(self):
        """모델 비교 도구 초기화"""
        self.results = {}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장 디렉토리 생성
        self.results_dir = "comparison_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def measure_memory(self, model) -> Dict[str, float]:
        """모델 메모리 사용량 측정

        Args:
            model: 측정할 모델

        Returns:
            Dict[str, float]: GPU와 CPU 메모리 사용량 (MB)
        """
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
    
    def measure_inference_time(
        self,
        model,
        tokenizer,
        text: str,
        num_runs: int = 100
    ) -> float:
        """추론 시간 측정

        Args:
            model: 측정할 모델
            tokenizer: 토크나이저
            text (str): 테스트 텍스트
            num_runs (int): 측정 반복 횟수

        Returns:
            float: 평균 추론 시간 (초)
        """
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 워밍업
        with torch.no_grad():
            model.generate(**inputs, max_length=50)
        
        # 시간 측정
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                model.generate(**inputs, max_length=50)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return sum(times) / len(times)
    
    def calculate_perplexity(
        self,
        model,
        tokenizer,
        test_texts: List[str]
    ) -> float:
        """모델의 펄플렉시티 계산

        Args:
            model: 평가할 모델
            tokenizer: 토크나이저
            test_texts (List[str]): 테스트 텍스트 목록

        Returns:
            float: 평균 펄플렉시티
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    def get_model_size(self, model) -> float:
        """모델 파일 크기 측정

        Args:
            model: 측정할 모델

        Returns:
            float: 모델 크기 (MB)
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / (1024**2)  # MB로 변환
    
    def compare_models(
        self,
        base_model_path: str,
        lora_model_path: str,
        test_texts: List[str]
    ) -> Dict[str, PerformanceMetrics]:
        """모델 비교 수행

        Args:
            base_model_path (str): 기본 모델 경로
            lora_model_path (str): LoRA 모델 경로
            test_texts (List[str]): 테스트 텍스트 목록

        Returns:
            Dict[str, PerformanceMetrics]: 각 모델의 성능 메트릭
        """
        results = {}
        
        # 기본 모델 평가
        self.logger.info("Evaluating base model...")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        if torch.cuda.is_available():
            base_model = base_model.cuda()
        
        memory = self.measure_memory(base_model)
        inference_time = self.measure_inference_time(
            base_model, tokenizer, test_texts[0]
        )
        perplexity = self.calculate_perplexity(base_model, tokenizer, test_texts)
        model_size = self.get_model_size(base_model)
        
        results["base_model"] = PerformanceMetrics(
            gpu_memory=memory["gpu_memory_mb"],
            cpu_memory=memory["cpu_memory_mb"],
            inference_time=inference_time,
            model_size=model_size,
            perplexity=perplexity
        )
        
        # LoRA 모델 평가
        self.logger.info("Evaluating LoRA model...")
        lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path)
        
        if torch.cuda.is_available():
            lora_model = lora_model.cuda()
        
        memory = self.measure_memory(lora_model)
        inference_time = self.measure_inference_time(
            lora_model, tokenizer, test_texts[0]
        )
        perplexity = self.calculate_perplexity(lora_model, tokenizer, test_texts)
        model_size = self.get_model_size(lora_model)
        
        results["lora_model"] = PerformanceMetrics(
            gpu_memory=memory["gpu_memory_mb"],
            cpu_memory=memory["cpu_memory_mb"],
            inference_time=inference_time,
            model_size=model_size,
            perplexity=perplexity
        )
        
        return results
    
    def visualize_comparison(
        self,
        results: Dict[str, PerformanceMetrics]
    ):
        """비교 결과 시각화

        Args:
            results (Dict[str, PerformanceMetrics]): 비교 결과
        """
        # 데이터 준비
        metrics_df = pd.DataFrame({
            'Metric': ['GPU Memory (MB)', 'CPU Memory (MB)', 
                      'Inference Time (s)', 'Model Size (MB)', 'Perplexity'],
            'Base Model': [
                results['base_model'].gpu_memory,
                results['base_model'].cpu_memory,
                results['base_model'].inference_time,
                results['base_model'].model_size,
                results['base_model'].perplexity
            ],
            'LoRA Model': [
                results['lora_model'].gpu_memory,
                results['lora_model'].cpu_memory,
                results['lora_model'].inference_time,
                results['lora_model'].model_size,
                results['lora_model'].perplexity
            ]
        })
        
        # 그래프 생성
        plt.figure(figsize=(12, 8))
        metrics_df_melted = pd.melt(
            metrics_df, 
            id_vars=['Metric'], 
            var_name='Model', 
            value_name='Value'
        )
        
        sns.barplot(
            data=metrics_df_melted,
            x='Metric',
            y='Value',
            hue='Model'
        )
        
        plt.xticks(rotation=45)
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.results_dir}/comparison_{timestamp}.png")
        metrics_df.to_csv(f"{self.results_dir}/metrics_{timestamp}.csv")
    
    def print_comparison(self, results: Dict[str, PerformanceMetrics]):
        """비교 결과 출력

        Args:
            results (Dict[str, PerformanceMetrics]): 비교 결과
        """
        print("\n=== 모델 비교 결과 ===")
        
        print("\n1. 메모리 사용량")
        print("기본 모델:")
        print(f"- GPU: {results['base_model'].gpu_memory:.2f} MB")
        print(f"- CPU: {results['base_model'].cpu_memory:.2f} MB")
        print("\nLoRA 모델:")
        print(f"- GPU: {results['lora_model'].gpu_memory:.2f} MB")
        print(f"- CPU: {results['lora_model'].cpu_memory:.2f} MB")
        
        print("\n2. 추론 시간")
        print(f"기본 모델: {results['base_model'].inference_time*1000:.2f} ms")
        print(f"LoRA 모델: {results['lora_model'].inference_time*1000:.2f} ms")
        
        print("\n3. 모델 크기")
        print(f"기본 모델: {results['base_model'].model_size:.2f} MB")
        print(f"LoRA 모델: {results['lora_model'].model_size:.2f} MB")
        
        print("\n4. 성능 (펄플렉시티)")
        print(f"기본 모델: {results['base_model'].perplexity:.2f}")
        print(f"LoRA 모델: {results['lora_model'].perplexity:.2f}")
        
        # 개선율 계산
        memory_reduction = (
            (results['base_model'].gpu_memory - results['lora_model'].gpu_memory)
            / results['base_model'].gpu_memory * 100
        )
        size_reduction = (
            (results['base_model'].model_size - results['lora_model'].model_size)
            / results['base_model'].model_size * 100
        )
        
        print("\n=== 개선율 ===")
        print(f"메모리 사용량 감소: {memory_reduction:.1f}%")
        print(f"모델 크기 감소: {size_reduction:.1f}%")

def example_usage():
    """사용 예시"""
    # 테스트 텍스트 준비
    test_texts = [
        "이 제품의 특징을 설명해주세요.",
        "서비스의 장단점을 분석해주세요.",
        "이 기술의 미래 전망은 어떨까요?"
    ]
    
    # 비교 도구 초기화
    comparator = ModelComparator()
    
    # 모델 비교 실행
    results = comparator.compare_models(
        "beomi/kcbert-base",
        "output/lora_model",
        test_texts
    )
    
    # 결과 출력 및 시각화
    comparator.print_comparison(results)
    comparator.visualize_comparison(results)

if __name__ == "__main__":
    example_usage()
