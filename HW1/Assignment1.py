import numpy as np
import random
from collections import deque
import heapq

class Geometry:
    count = 0
    def __init__(self, name = "Shape", points = None):
        self.name = name
        # name is string that is a name of gemoetry
        self.points = points
        # points is a list of tuple points = [(x0, y0), (x1, y1), ...]
        Geometry.count += 1

    def calculate_area(self):
        return 0.0

    def get_name(self):
        return self.name

    @classmethod
    def count_number_of_geometry(cls):
        # TODO: Your task is to implement the class method
        # to get the number of instance that have already created
        return cls.count

class Triangle(Geometry):
    def __init__(self, a, b, c):
        # a, b, c are tuples that represent for 3 vertices of a triangle
        # TODO: Your task is to implement the constructor
        super(Triangle, self).__init__("Triangle", [a,b,c])

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        points = self.points
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        x.append(x[0])
        y.append(y[0])
        area = 0.5 * abs(sum(i * j for i, j in zip(x[:-1], y[1:])) - sum(i * j for i, j in zip(x[1:], y[:-1])))
        return area

class Rectangle(Geometry):
    def __init__(self, a, b):
        # a, b are tuples that represent for top and bottom vertices of a rectangle
        # TODO: Your task is to implement the constructor
        c = (b[0],a[1])
        d = (a[0],b[1])
        super(Rectangle, self).__init__("Rectangle", [a,c,b,d])

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        width = abs(self.points[1][0] - self.points[0][0])
        height = abs(self.points[3][1] - self.points[0][1])
        return width * height

class Square(Rectangle):
    def __init__(self, a, length):
        # a is a tuple that represent a top vertex of a square
        # length is the side length of a square
        # TODO: Your task is to implement the constructor
        b = (a[0] + length, a[1] + length)
        super(Square, self).__init__(a, b)
        self.name = "Square"

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        length = abs(self.points[1][0] - self.points[0][0])
        return length * length

class Circle(Geometry):
    def __init__(self, o, r):
        # o is a tuple that represent a centre of a circle
        # r is the radius of a circle
        # TODO: Your task is to implement the constructor
        super(Circle, self).__init__("Circle", [o])
        self.radius = r

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        return np.pi * self.radius * self.radius

class Polygon(Geometry):
    def __init__(self, points):
        # points is a list of tuples that represent vertices of a polygon
        # TODO: Your task is to implement the constructor
        super(Polygon, self).__init__("Polygon", points)

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        x.append(x[0])
        y.append(y[0])
        area = 0.5 * abs(sum(i * j for i, j in zip(x[:-1], y[1:])) - sum(i * j for i, j in zip(x[1:], y[:-1])))
        return area

def test_geomery():
    ## Test cases for Problem 1

    triangle = Triangle((0, 1), (1, 0), (0, 0))
    print("Area of %s: %0.4f" % (triangle.name, triangle.calculate_area()))

    rectangle = Rectangle((0, 0), (2, 2))
    print("Area of %s: %0.4f" % (rectangle.name, rectangle.calculate_area()))

    square = Square((0, 0), 2)
    print("Area of %s: %0.4f" % (square.name, square.calculate_area()))

    circle = Circle((0, 0), 3)
    print("Area of %s: %0.4f" % (circle.name, circle.calculate_area()))

    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    print("Area of %s: %0.4f" % (polygon.name, polygon.calculate_area()))


def matrix_multiplication(A, B):
    # TODO: Your task is to required to implement
    # a matrix multiplication between A and B
    m, n = A.shape
    n2, k = B.shape
    if n != n2:
        raise ValueError("Matrix dimensions don't match")
    
    C = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            for p in range(n):
                C[i,j] += A[i,p] * B[p,j]
    return C

def test_matrix_mul():
    ## Test cases for matrix multplication ##

    for test in range(10):
        m, n, k = random.randint(3, 10), random.randint(3, 10), random.randint(3, 10)
        A = np.random.randn(m, n)
        B = np.random.randn(n, k)
        assert np.mean(np.abs(A.dot(B) - matrix_multiplication(A, B))) <= 1e-7, "Your implmentation is wrong!"
        print("[Test Case %d]. Your implementation is correct!" % test)

def recursive_pow(A, n):
    # TODO: Your task is required implementing
    # a recursive function
    if n == 0:
        return np.eye(A.shape[0])
    if n == 1:
        return A
    
    if n % 2 == 0:
        half = recursive_pow(A, n//2)
        return matrix_multiplication(half, half)
    else:
        return matrix_multiplication(A, recursive_pow(A, n-1))

def iterative_pow(A, n):
	# TODO: Your task is required implementing
    # a iterative function
    if n == 0:
        return np.eye(A.shape[0])
    
    result = np.eye(A.shape[0])
    base = A.copy()
    
    while n > 0:
        if n % 2 == 1:
            result = matrix_multiplication(result, base)
        base = matrix_multiplication(base, base)
        n //= 2
    
    return result

def test_pow():
    ## Test cases for the pow function ##
    # test_A = np.array([[2, 1], [1, 3]])
    # print("Recursive:", recursive_pow(test_A, 3))
    # print("Iterative:", iterative_pow(test_A, 3))

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Recursive: A^{} = {}".format(n, recursive_pow(A, n)))

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Iterative: A^{} = {}".format(n, iterative_pow(A, n)))

def get_A():
    # TODO: Find a matrix A
    # You have to return in the format of numpy array
    return np.array([[1, 1], [1, 0]])

def fibo(n):
    # TODO: Calcualte the n'th Fibonacci number
    if n <= 1:
        return 1
    A = get_A()
    result = recursive_pow(A, n-1)
    return result[0,0] + result[0,1]

def f(n, k):
    # TODO: Calcualte the n'th number of the recursive sequence
    if n < k:
        return 1
        
    terms = [1, 1]  # f(0)=1, f(1)=1
    for i in range(2, n+1):
       terms.append(terms[-1] + terms[-2])
       
    # Return sum of previous k terms for nth position
    return sum(terms[n-k:n])

def test_fibonacci():
    ## Test Cases for Fibonacci and Recursive Sequence ##

    a, b = 1, 1
    for i in range(2, 10):
        c = a + b
        assert (fibo(i) == c), "You implementation is incorrect"
        print("[Test Case %d]. Your implementation is correct!. fibo(%d) = %d" % (i - 2, i, fibo(i)))
        a = b
        b = c

    for n in range(5, 11):
        for k in range(2, 5):
            print("f(%d, %d) = %d" % (n, k, f(n, k)))
# ----------------------------------------------------------------------------------------------------
def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n

def DFS(A):
    # A is a mxn matrix
    m, n = A.shape
    if A[0,0] == 0 or A[m-1,n-1] == 0:
        return -1
    
    visited = set()
    path = []
    
    def dfs(x, y):
        if x == m-1 and y == n-1:
            path.append((x,y))
            return True
            
        visited.add((x,y))
        path.append((x,y))
        
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (is_valid(new_x, new_y, m, n) and 
                A[new_x,new_y] == 1 and 
                (new_x,new_y) not in visited):
                if dfs(new_x, new_y):
                    return True
                    
        path.pop()
        return False
    
    if dfs(0, 0):
        print(" → ".join(f"({x},{y})" for x,y in path))
        return path
    return -1

def BFS(A):
    # A is a mxn matrix
    m, n = A.shape
    if A[0,0] == 0 or A[m-1,n-1] == 0:
        return -1
        
    queue = deque([(0,0)])
    visited = {(0,0): None}
    
    while queue:
        x, y = queue.popleft()
        if x == m-1 and y == n-1:
            path = []
            curr = (x,y)
            while curr is not None:
                path.append(curr)
                curr = visited[curr]
            path.reverse()
            print(" → ".join(f"({x},{y})" for x,y in path))
            return path
            
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (is_valid(new_x, new_y, m, n) and 
                A[new_x,new_y] == 1 and 
                (new_x,new_y) not in visited):
                queue.append((new_x,new_y))
                visited[(new_x,new_y)] = (x,y)
    
    return -1

def findMinimum(A):
    # A is a mxn matrix
    m, n = A.shape
    if A[0,0] == 0 or A[m-1,n-1] == 0:
        return -1

    pq = [(A[0,0], 0, 0, [(0,0)])]
    visited = set()
    
    while pq:
        cost, x, y, path = heapq.heappop(pq)
        
        if x == m-1 and y == n-1:
            print(" → ".join(f"({x},{y})" for x,y in path))
            print(f"Total value: {cost}")
            return path
            
        if (x,y) in visited:
            continue
            
        visited.add((x,y))
        
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (is_valid(new_x, new_y, m, n) and 
                A[new_x,new_y] != 0 and 
                (new_x,new_y) not in visited):
                new_cost = cost + A[new_x,new_y]
                new_path = path + [(new_x,new_y)]
                heapq.heappush(pq, (new_cost, new_x, new_y, new_path))
    
    return -1

def findMinimumAdv(A):
    m, n = A.shape
    if A[0,0] == 0 or A[m-1,n-1] == 0:
        return -1

    dist = np.full((m, n), np.inf)
    dist[0,0] = A[0,0]
    
    pq = [(A[0,0], 0, 0)]
    parent = {(0,0): None}
    
    while pq:
        cost, x, y = heapq.heappop(pq)
        
        if cost > dist[x,y]:
            continue
            
        if x == m-1 and y == n-1:
            path = []
            curr = (x,y)
            while curr:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            print(" → ".join(f"({x},{y})" for x,y in path))
            print(f"Total value: {int(dist[m-1,n-1])}")
            return path
            
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y, m, n) and A[new_x,new_y] != 0:
                new_cost = cost + A[new_x,new_y]
                
                if new_cost < dist[new_x,new_y]:
                    dist[new_x,new_y] = new_cost
                    parent[(new_x,new_y)] = (x,y)
                    heapq.heappush(pq, (new_cost, new_x, new_y))
    
    return -1

def test_bfs_dfs_find_minimum():
    ## Test Cases for BFS, DFS, Find Minimum ##
    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])

    BFS(A)

    DFS(A)

    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 2], [1, 1, 0, 2, 1], [1, 1, 0, 2, 1]])

    findMinimum(A)
    findMinimumAdv(A)

## Testing Your Code

test_geomery()
test_matrix_mul()
test_pow()
test_fibonacci()
test_bfs_dfs_find_minimum()
