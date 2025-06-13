class Calculator:
    def __init__(self):
        self.history = []
        self.last_result = None
    
    def add(self, x, y):
        """덧셈 연산을 수행합니다."""
        result = x + y
        self._update_history('add', x, y, result)
        return result
    
    def subtract(self, x, y):
        """뺄셈 연산을 수행합니다."""
        result = x - y
        self._update_history('subtract', x, y, result)
        return result
    
    def multiply(self, x, y):
        """곱셈 연산을 수행합니다."""
        result = x * y
        self._update_history('multiply', x, y, result)
        return result
    
    def divide(self, x, y):
        """나눗셈 연산을 수행합니다."""
        if y == 0:
            raise ValueError("0으로 나눌 수 없습니다.")
        result = x / y
        self._update_history('divide', x, y, result)
        return result
    
    def _update_history(self, operation, x, y, result):
        """계산 기록을 업데이트합니다."""
        self.history.append({
            'operation': operation,
            'x': x,
            'y': y,
            'result': result
        })
        self.last_result = result
    
    def get_last_result(self):
        """마지막 계산 결과를 반환합니다."""
        return self.last_result
    
    def get_history(self):
        """전체 계산 기록을 반환합니다."""
        return self.history

# 사용 예시
if __name__ == "__main__":
    # 계산기 인스턴스 생성
    calc = Calculator()
    
    # 덧셈
    result1 = calc.add(5, 3)
    print(f"5 + 3 = {result1}")
    
    # 뺄셈
    result2 = calc.subtract(10, 4)
    print(f"10 - 4 = {result2}")
    
    # 곱셈
    result3 = calc.multiply(6, 2)
    print(f"6 * 2 = {result3}")
    
    # 나눗셈
    result4 = calc.divide(15, 3)
    print(f"15 / 3 = {result4}")
    
    # 마지막 결과 확인
    print(f"마지막 결과: {calc.get_last_result()}")
    
    # 계산 기록 확인
    print("\n계산 기록:")
    for record in calc.get_history():
        print(f"연산: {record['operation']}, "
              f"x: {record['x']}, "
              f"y: {record['y']}, "
              f"결과: {record['result']}")
