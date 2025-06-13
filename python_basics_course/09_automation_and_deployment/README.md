# 7주차: 자동화 워크플로우 & AWS 배포·CI/CD·모니터링

## 학습 목표
- n8n을 활용한 워크플로우 자동화 구현
- Task Master로 작업 스케줄링 관리
- Docker 컨테이너화 및 AWS 배포
- GitHub Actions를 통한 CI/CD 파이프라인 구축
- AWS CloudWatch와 Prometheus를 활용한 모니터링

## 프로젝트 구조
```
09_automation_and_deployment/
├── n8n/
│   ├── flows/
│   │   ├── data_preprocessing.json
│   │   └── gpt_workflow.json
│   └── docker-compose.yml
├── taskmaster/
│   ├── tasks/
│   │   ├── daily_report.py
│   │   └── data_cleanup.py
│   └── config.yaml
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
└── monitoring/
    ├── prometheus/
    │   └── prometheus.yml
    └── grafana/
        └── dashboards/
```

## 설치 방법

### 1. n8n 설정
```bash
# Docker로 n8n 실행
docker-compose -f n8n/docker-compose.yml up -d
```

### 2. Task Master 설정
```bash
pip install taskmaster-scheduler
```

### 3. AWS CLI 설정
```bash
aws configure
```

## 주요 기능 구현

### 1. n8n 워크플로우
- 데이터 수집 및 전처리
- GPT API 호출
- 결과 저장 및 알림

### 2. Task Master 작업
- 정기적인 데이터 클린업
- 일일 리포트 생성
- 조건부 작업 트리거

### 3. Docker 컨테이너화
- 멀티스테이지 빌드
- 최적화된 이미지 생성
- 보안 베스트 프랙티스

### 4. AWS 배포
- ECR 레포지토리 생성
- ECS Fargate 서비스 설정
- 로드 밸런서 구성

### 5. CI/CD 파이프라인
- 자동 테스트
- 보안 스캔
- 자동 배포

### 6. 모니터링 설정
- CloudWatch 메트릭 구성
- Prometheus + Grafana 대시보드
- 알람 설정

## 실습 과제
1. n8n 워크플로우 구현
   - 데이터 수집 플로우
   - GPT 처리 플로우
   - 결과 저장 플로우

2. Task Master 스케줄링
   - 일간 작업 설정
   - 주간 작업 설정
   - 조건부 트리거 설정

3. AWS 인프라 구축
   - Terraform 스크립트 작성
   - ECS 클러스터 설정
   - 로드 밸런서 구성

4. CI/CD 파이프라인 구축
   - GitHub Actions 워크플로우 작성
   - 테스트 자동화
   - 배포 자동화

5. 모니터링 시스템 구축
   - CloudWatch 대시보드 설정
   - Prometheus 메트릭 수집
   - Grafana 대시보드 구성

## 참고 자료
- [n8n 공식 문서](https://docs.n8n.io/)
- [AWS ECS 가이드](https://docs.aws.amazon.com/ecs/)
- [GitHub Actions 문서](https://docs.github.com/actions)
- [Prometheus 가이드](https://prometheus.io/docs/)
