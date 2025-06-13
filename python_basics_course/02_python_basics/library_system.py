from datetime import datetime, timedelta

class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_available = True
        self.due_date = None
        self.borrower = None

class LibrarySystem:
    def __init__(self):
        self.books = {}  # ISBN을 키로 사용
        self.loan_period = 14  # 대출 기간 (일)
    
    def add_book(self, title, author, isbn):
        """도서를 시스템에 추가합니다."""
        if isbn in self.books:
            raise ValueError(f"ISBN {isbn}이(가) 이미 존재합니다.")
        self.books[isbn] = Book(title, author, isbn)
        return f"도서 '{title}'이(가) 추가되었습니다."
    
    def remove_book(self, isbn):
        """도서를 시스템에서 제거합니다."""
        if isbn not in self.books:
            raise ValueError(f"ISBN {isbn}을(를) 찾을 수 없습니다.")
        del self.books[isbn]
        return f"ISBN {isbn} 도서가 제거되었습니다."
    
    def search_by_title(self, title):
        """제목으로 도서를 검색합니다."""
        return [book for book in self.books.values() 
                if title.lower() in book.title.lower()]
    
    def search_by_author(self, author):
        """저자로 도서를 검색합니다."""
        return [book for book in self.books.values() 
                if author.lower() in book.author.lower()]
    
    def borrow_book(self, isbn, borrower):
        """도서를 대출합니다."""
        if isbn not in self.books:
            raise ValueError(f"ISBN {isbn}을(를) 찾을 수 없습니다.")
        
        book = self.books[isbn]
        if not book.is_available:
            return f"도서 '{book.title}'은(는) 현재 대출 중입니다."
        
        book.is_available = False
        book.borrower = borrower
        book.due_date = datetime.now() + timedelta(days=self.loan_period)
        
        return (f"도서 '{book.title}'이(가) 대출되었습니다.\n"
                f"반납 예정일: {book.due_date.strftime('%Y-%m-%d')}")
    
    def return_book(self, isbn):
        """도서를 반납합니다."""
        if isbn not in self.books:
            raise ValueError(f"ISBN {isbn}을(를) 찾을 수 없습니다.")
        
        book = self.books[isbn]
        if book.is_available:
            return f"도서 '{book.title}'은(는) 이미 반납되었습니다."
        
        book.is_available = True
        book.due_date = None
        book.borrower = None
        
        return f"도서 '{book.title}'이(가) 반납되었습니다."
    
    def get_overdue_books(self):
        """연체된 도서 목록을 반환합니다."""
        current_date = datetime.now()
        overdue_books = [
            book for book in self.books.values()
            if not book.is_available and book.due_date < current_date
        ]
        return overdue_books

# 사용 예시
if __name__ == "__main__":
    # 도서관 시스템 인스턴스 생성
    library = LibrarySystem()
    
    # 도서 추가
    print(library.add_book("Python 기초", "홍길동", "123456789"))
    print(library.add_book("데이터 분석의 정석", "김데이터", "987654321"))
    print(library.add_book("인공지능 입문", "이AI", "456789123"))
    
    # 도서 검색
    print("\n'Python' 키워드로 검색:")
    for book in library.search_by_title("Python"):
        print(f"- {book.title} (저자: {book.author})")
    
    # 도서 대출
    print("\n도서 대출:")
    print(library.borrow_book("123456789", "학생1"))
    
    # 연체 도서 확인 (시뮬레이션을 위해 due_date를 강제로 과거로 설정)
    book = library.books["123456789"]
    book.due_date = datetime.now() - timedelta(days=1)
    
    print("\n연체 도서 목록:")
    overdue_books = library.get_overdue_books()
    for book in overdue_books:
        print(f"- {book.title} (대출자: {book.borrower}, "
              f"반납예정일: {book.due_date.strftime('%Y-%m-%d')})")
