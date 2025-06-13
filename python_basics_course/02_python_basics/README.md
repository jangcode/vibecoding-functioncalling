# 모듈 2: Python 기초 문법

## 학습 목표
- Python의 기본 자료형과 연산자를 이해하고 활용할 수 있다
- 제어문과 함수를 사용하여 논리적인 프로그램을 작성할 수 있다
- 클래스를 통해 객체지향 프로그래밍을 구현할 수 있다
- Python의 모듈과 패키지를 활용할 수 있다

## 1. 기본 자료형과 연산자

### 1.1 자료형
```python
# 숫자형
integer_num = 42        # 정수
float_num = 3.14       # 실수
complex_num = 1 + 2j   # 복소수

# 문자열
text = "Hello, World!"  # 큰따옴표
text2 = 'Python'       # 작은따옴표
multi_line = """
여러 줄의
문자열
작성
"""

# 불리언
is_true = True
is_false = False

# 시퀀스형
my_list = [1, 2, 3]           # 리스트 (가변)
my_tuple = (1, 2, 3)          # 튜플 (불변)
my_range = range(5)           # range

# 매핑형
my_dict = {"name": "Python"}  # 딕셔너리

# 집합형
my_set = {1, 2, 3}           # 집합
```

### 1.2 연산자
```python
# 산술 연산자
a = 10
b = 3
print(f"덧셈: {a + b}")       # 13
print(f"뺄셈: {a - b}")       # 7
print(f"곱셈: {a * b}")       # 30
print(f"나눗셈: {a / b}")     # 3.3333...
print(f"몫: {a // b}")        # 3
print(f"나머지: {a % b}")     # 1
print(f"제곱: {a ** b}")      # 1000

# 비교 연산자
print(a > b)    # True
print(a < b)    # False
print(a >= b)   # True
print(a <= b)   # False
print(a == b)   # False
print(a != b)   # True

# 논리 연산자
x = True
y = False
print(f"AND: {x and y}")  # False
print(f"OR: {x or y}")    # True
print(f"NOT: {not x}")    # False
```

## 2. 제어문

### 2.1 조건문
```python
# if-elif-else
score = 85

if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'

print(f"학점: {grade}")  # B
```

### 2.2 반복문
```python
# for 반복문
fruits = ['사과', '바나나', '오렌지']
for fruit in fruits:
    print(fruit)

# range를 사용한 반복
for i in range(3):
    print(i)  # 0, 1, 2

# while 반복문
count = 0
while count < 3:
    print(count)
    count += 1
```

## 3. 함수

### 3.1 함수 정의와 호출
```python
def greet(name):
    """
    인사말을 반환하는 함수
    
    Args:
        name (str): 사용자 이름
    
    Returns:
        str: 인사말 메시지
    """
    return f"안녕하세요, {name}님!"

# 함수 호출
message = greet("Python")
print(message)  # 안녕하세요, Python님!
```

### 3.2 함수 매개변수
```python
# 기본 매개변수
def power(base, exponent=2):
    return base ** exponent

print(power(2))      # 4 (2^2)
print(power(2, 3))   # 8 (2^3)

# 가변 매개변수
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # 6

# 키워드 매개변수
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Python", age=30)
```

## 4. 클래스와 객체

### 4.1 클래스 정의
```python
class Dog:
    # 클래스 변수
    species = "개"
    
    # 초기화 메서드
    def __init__(self, name, age):
        self.name = name  # 인스턴스 변수
        self.age = age
    
    # 인스턴스 메서드
    def bark(self):
        return f"{self.name}가 짖습니다!"
    
    # 인스턴스 메서드
    def info(self):
        return f"{self.name}는 {self.age}살입니다."

# 클래스 사용
my_dog = Dog("멍멍이", 3)
print(my_dog.bark())    # 멍멍이가 짖습니다!
print(my_dog.info())    # 멍멍이는 3살입니다.
```

### 4.2 상속
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        return f"{self.name}: 야옹!"

class Duck(Animal):
    def speak(self):
        return f"{self.name}: 꽥!"

# 상속 사용
cat = Cat("나비")
duck = Duck("도널드")

print(cat.speak())   # 나비: 야옹!
print(duck.speak())  # 도널드: 꽥!
```

## 실습 과제

### 1. 기본 계산기 만들기
다음 요구사항을 만족하는 계산기 클래스를 작성하세요:
- 덧셈, 뺄셈, 곱셈, 나눗셈 기능
- 계산 기록 저장 기능
- 최근 계산 결과 확인 기능

### 2. 도서 관리 시스템
다음 기능을 포함하는 도서 관리 시스템을 구현하세요:
- 도서 추가/삭제
- 도서 검색 (제목, 저자)
- 대출/반납 관리
- 연체 도서 확인

## 참고 자료
- [Python 공식 문서](https://docs.python.org/ko/3/)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)

## 다음 단계
기초 문법을 마스터했다면, 다음 모듈에서는 Pandas를 활용한 데이터 분석을 학습할 예정입니다.
