import os
import json
from datetime import datetime
import pandas as pd
from openai import OpenAI

def generate_daily_report():
    """일일 데이터 처리 결과 리포트 생성"""
    try:
        # 결과 데이터 로드
        with open('/data/output/results.json', 'r') as f:
            results = json.load(f)
        
        # 데이터 분석
        df = pd.DataFrame(results)
        
        # GPT를 사용한 리포트 생성
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        report_prompt = f"""
        오늘의 데이터 처리 결과:
        - 총 처리 건수: {len(df)}
        - 성공률: {(df['status'] == 'success').mean() * 100:.2f}%
        
        이 데이터를 바탕으로 간단한 분석 리포트를 작성해주세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": report_prompt}
            ]
        )
        
        # 리포트 저장
        report_date = datetime.now().strftime('%Y-%m-%d')
        report_path = f'/data/reports/daily_report_{report_date}.txt'
        
        with open(report_path, 'w') as f:
            f.write(response.choices[0].message.content)
            
        print(f"Daily report generated: {report_path}")
        return True
        
    except Exception as e:
        print(f"Error generating daily report: {str(e)}")
        return False

if __name__ == "__main__":
    generate_daily_report()
