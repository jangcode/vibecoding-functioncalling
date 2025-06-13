# 모듈 3: Pandas를 활용한 데이터 분석

## 학습 목표
- Pandas의 기본 데이터 구조를 이해하고 활용할 수 있다
- 데이터 로드, 정제, 변환 작업을 수행할 수 있다
- 기본적인 데이터 분석과 시각화를 수행할 수 있다
- 실제 데이터셋을 활용한 분석 프로젝트를 수행할 수 있다

## 1. Pandas 기초

### 1.1 Series와 DataFrame
```python
import pandas as pd
import numpy as np

# Series 생성
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series 예제:")
print(s)

# DataFrame 생성
dates = pd.date_range('20250101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), 
                 index=dates,
                 columns=list('ABCD'))
print("\nDataFrame 예제:")
print(df)
```

### 1.2 데이터 불러오기
```python
# CSV 파일 읽기
df_csv = pd.read_csv('data.csv')

# Excel 파일 읽기
df_excel = pd.read_excel('data.xlsx')

# JSON 파일 읽기
df_json = pd.read_json('data.json')

# 데이터베이스에서 읽기
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df_sql = pd.read_sql('SELECT * FROM table', engine)
```

## 2. 데이터 탐색과 전처리

### 2.1 기본 정보 확인
```python
# 데이터 미리보기
print(df.head())
print(df.tail())

# 기본 정보 확인
print(df.info())
print(df.describe())

# 결측치 확인
print(df.isnull().sum())
```

### 2.2 데이터 선택과 필터링
```python
# 열 선택
df['A']                  # 단일 열 선택
df[['A', 'B']]          # 복수 열 선택

# 행 선택
df.loc[dates[0]]        # 라벨로 선택
df.iloc[0]              # 위치로 선택

# 조건부 선택
df[df['A'] > 0]        # A열의 값이 0보다 큰 행 선택
```

### 2.3 데이터 변환
```python
# 결측치 처리
df.fillna(value=0)               # 0으로 채우기
df.dropna(how='any')            # 결측치가 있는 행 제거

# 데이터 타입 변환
df['column'] = df['column'].astype('int64')

# 중복 제거
df.drop_duplicates()
```

## 3. 데이터 분석

### 3.1 기술 통계
```python
# 기본 통계량
df.mean()              # 평균
df.median()           # 중앙값
df.std()              # 표준편차
df.var()              # 분산
df.mode()             # 최빈값

# 그룹화와 집계
df.groupby('category').mean()
df.groupby(['A', 'B']).sum()
```

### 3.2 데이터 병합
```python
# 데이터프레임 연결
pd.concat([df1, df2])

# 데이터프레임 병합
pd.merge(df1, df2, on='key')
```

## 4. 데이터 시각화

### 4.1 기본 플롯
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 선 그래프
df.plot(kind='line')

# 막대 그래프
df.plot(kind='bar')

# 산점도
df.plot(kind='scatter', x='A', y='B')

# 히스토그램
df['A'].hist()
```

### 4.2 Seaborn 활용
```python
# 상관관계 히트맵
sns.heatmap(df.corr(), annot=True)

# 박스플롯
sns.boxplot(x='category', y='value', data=df)

# 바이올린 플롯
sns.violinplot(x='category', y='value', data=df)
```

## 실습 과제

### 1. 판매 데이터 분석
`sales_data.csv` 파일을 사용하여 다음 작업을 수행하세요:
- 월별, 제품별 판매량 분석
- 최대/최소 판매 기간 파악
- 판매 트렌드 시각화
- 제품 카테고리별 성과 분석

### 2. 고객 데이터 분석
`customer_data.csv` 파일을 사용하여 다음 작업을 수행하세요:
- 고객 세그먼트 분석
- 구매 패턴 파악
- 고객 생애 가치(CLV) 계산
- 이탈 고객 예측 모델 구축

## 추가 학습 자료
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Seaborn 공식 문서](https://seaborn.pydata.org/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

## 다음 단계
데이터 분석 기초를 마스터했다면, 다음 모듈에서는 LLM과 효과적인 프롬프트 작성법을 학습할 예정입니다.
