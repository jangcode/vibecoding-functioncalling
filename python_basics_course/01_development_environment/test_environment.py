import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_environment():
    print(f"Python 버전: {sys.version}")
    print(f"NumPy 버전: {np.__version__}")
    print(f"Pandas 버전: {pd.__version__}")
    
    # 간단한 데이터 시각화 테스트
    data = np.random.normal(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    sns.histplot(data)
    plt.title("정규 분포 테스트")
    plt.show()

if __name__ == "__main__":
    test_environment()
