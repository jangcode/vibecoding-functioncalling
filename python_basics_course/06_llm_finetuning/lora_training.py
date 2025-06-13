import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import os
import logging
from typing import Dict, Any

class LoRATrainer:
    def __init__(
        self,
        base_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 2e-4
    ):
        """LoRA 트레이너 초기화

        Args:
            base_model_name (str): 기본 모델 이름
            lora_r (int): LoRA rank
            lora_alpha (int): LoRA alpha
            lora_dropout (float): LoRA dropout
            learning_rate (float): 학습률
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Using device: {self.device}")
        
        try:
            # 기본 모델과 토크나이저 로드
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # LoRA 설정
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"]  # 모델에 따라 조정 필요
            )
            
            # LoRA 모델 생성
            self.model = get_peft_model(self.base_model, self.peft_config)
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    def print_trainable_parameters(self):
        """학습 가능한 파라미터 정보 출력"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        self.logger.info(
            f"trainable params: {trainable_params} || "
            f"all params: {all_params} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )
    
    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 100,
        logging_steps: int = 100,
        eval_steps: int = 500
    ) -> Trainer:
        """모델 학습

        Args:
            train_dataset: 학습 데이터셋
            val_dataset: 검증 데이터셋
            output_dir (str): 출력 디렉토리
            num_train_epochs (int): 학습 에폭 수
            per_device_train_batch_size (int): 배치 크기
            gradient_accumulation_steps (int): 그래디언트 누적 스텝
            save_steps (int): 모델 저장 주기
            logging_steps (int): 로깅 주기
            eval_steps (int): 평가 주기

        Returns:
            Trainer: 학습된 트레이너 객체
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                save_steps=save_steps,
                logging_steps=logging_steps,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                evaluation_strategy="steps",
                eval_steps=eval_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            self.logger.info("Starting training...")
            trainer.train()
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_dir: str):
        """모델 저장

        Args:
            output_dir (str): 저장할 디렉토리 경로
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    def evaluate(self, test_dataset) -> Dict[str, Any]:
        """모델 평가

        Args:
            test_dataset: 테스트 데이터셋

        Returns:
            Dict[str, Any]: 평가 메트릭
        """
        try:
            trainer = Trainer(
                model=self.model,
                eval_dataset=test_dataset
            )
            
            metrics = trainer.evaluate()
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """텍스트 생성

        Args:
            prompt (str): 입력 프롬프트
            max_length (int): 최대 생성 길이

        Returns:
            str: 생성된 텍스트
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

def example_usage():
    """사용 예시"""
    from data_preparation import DataPreparator
    
    # 데이터 준비
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    preparator = DataPreparator(tokenizer)
    dummy_df = preparator.create_dummy_dataset()
    train_dataset, val_dataset = preparator.prepare_dataset(dummy_df, 'text', 'label')
    
    # LoRA 트레이너 초기화
    trainer = LoRATrainer("beomi/kcbert-base")
    
    # 학습 가능한 파라미터 정보 출력
    trainer.print_trainable_parameters()
    
    # 모델 학습
    trainer.train(
        train_dataset,
        val_dataset,
        "output/lora_model",
        num_train_epochs=1  # 예시를 위해 1 에폭만 학습
    )
    
    # 모델 저장
    trainer.save_model("output/lora_model")
    
    # 평가
    metrics = trainer.evaluate(val_dataset)
    print("평가 결과:", metrics)
    
    # 텍스트 생성 테스트
    prompt = "이 제품의 특징은"
    generated = trainer.generate_text(prompt)
    print(f"\n입력: {prompt}")
    print(f"출력: {generated}")

if __name__ == "__main__":
    example_usage()
