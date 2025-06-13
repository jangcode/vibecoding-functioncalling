import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_sales_data(file_path):
    """판매 데이터를 로드하고 분석합니다."""
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 기본 정보 출력
    print("\n=== 데이터 기본 정보 ===")
    print(df.info())
    
    # 기술 통계량
    print("\n=== 기술 통계량 ===")
    print(df.describe())
    
    # 월별 판매량 분석
    monthly_sales = df.groupby('month')['quantity'].sum()
    print("\n=== 월별 판매량 ===")
    print(monthly_sales)
    
    # 제품별 판매량 분석
    product_sales = df.groupby('product')['quantity'].sum().sort_values(ascending=False)
    print("\n=== 제품별 판매량 ===")
    print(product_sales)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 월별 판매량 트렌드
    plt.subplot(2, 2, 1)
    monthly_sales.plot(kind='line')
    plt.title('월별 판매량 트렌드')
    plt.xlabel('월')
    plt.ylabel('판매량')
    
    # 제품별 판매량
    plt.subplot(2, 2, 2)
    product_sales.plot(kind='bar')
    plt.title('제품별 판매량')
    plt.xlabel('제품')
    plt.ylabel('판매량')
    plt.xticks(rotation=45)
    
    # 판매량 분포
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='quantity')
    plt.title('판매량 분포')
    
    # 제품 카테고리별 박스플롯
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='category', y='quantity')
    plt.title('카테고리별 판매량 분포')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

def analyze_customer_data(file_path):
    """고객 데이터를 분석합니다."""
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 고객 세그먼트 분석
    print("\n=== 고객 세그먼트 분석 ===")
    
    # RFM 분석
    current_date = pd.Timestamp.now()
    
    rfm = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - pd.to_datetime(x.max())).days,  # Recency
        'order_id': 'count',  # Frequency
        'total_amount': 'sum'  # Monetary
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # RFM 점수 계산
    rfm['R'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['frequency'], q=5, labels=[1, 2, 3, 4, 5])
    rfm['M'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])
    
    # RFM 세그먼트 점수
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    print("\n=== RFM 분석 결과 ===")
    print(rfm.head())
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 고객 세그먼트별 분포
    plt.subplot(2, 2, 1)
    sns.histplot(data=rfm, x='RFM_Score')
    plt.title('고객 세그먼트 분포')
    plt.xticks(rotation=90)
    
    # Recency vs Frequency
    plt.subplot(2, 2, 2)
    plt.scatter(rfm['recency'], rfm['frequency'])
    plt.title('Recency vs Frequency')
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency (count)')
    
    # Monetary 분포
    plt.subplot(2, 2, 3)
    sns.histplot(data=rfm, x='monetary')
    plt.title('구매금액 분포')
    
    # CLV (Customer Lifetime Value) 계산
    rfm['CLV'] = rfm['frequency'] * rfm['monetary']
    
    # CLV 분포
    plt.subplot(2, 2, 4)
    sns.histplot(data=rfm, x='CLV')
    plt.title('고객 생애 가치 분포')
    
    plt.tight_layout()
    plt.show()
    
    return rfm

# 테스트용 데이터 생성
def create_sample_data():
    """테스트용 샘플 데이터를 생성합니다."""
    
    # 판매 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    products = ['A', 'B', 'C', 'D', 'E']
    categories = ['전자제품', '의류', '식품']
    
    sales_data = {
        'date': np.random.choice(dates, 1000),
        'product': np.random.choice(products, 1000),
        'category': np.random.choice(categories, 1000),
        'quantity': np.random.randint(1, 100, 1000),
        'price': np.random.uniform(10, 1000, 1000)
    }
    
    sales_df = pd.DataFrame(sales_data)
    sales_df['month'] = sales_df['date'].dt.month
    sales_df.to_csv('sales_data.csv', index=False)
    
    # 고객 데이터 생성
    customers = range(1, 101)
    purchase_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    
    customer_data = {
        'customer_id': np.random.choice(customers, 1000),
        'purchase_date': np.random.choice(purchase_dates, 1000),
        'order_id': range(1, 1001),
        'total_amount': np.random.uniform(10, 1000, 1000)
    }
    
    customer_df = pd.DataFrame(customer_data)
    customer_df.to_csv('customer_data.csv', index=False)
    
    return sales_df, customer_df

if __name__ == "__main__":
    # 샘플 데이터 생성
    sales_df, customer_df = create_sample_data()
    
    # 데이터 분석 실행
    print("\n=== 판매 데이터 분석 ===")
    sales_analysis = load_and_analyze_sales_data('sales_data.csv')
    
    print("\n=== 고객 데이터 분석 ===")
    customer_analysis = analyze_customer_data('customer_data.csv')
