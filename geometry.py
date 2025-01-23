# class Geometry: 
#     def __init__(self, name = "Shape", points = None): 
#         self.name = name 
#         # name is string that is a name of geometry 
#         self.points = points 
#         # points are a list of tuple points = [(x0, y0), (x1, y1), ...] 
 
#     def calculate_area(self): 
#         pass 
     
#     def get_name(self): 
#         return self.name 
     
#     @classmethod 
#     def count_number_of_geometry(cls): 
#         # TODO: Your task is to implement the class method  
#         # to get the number of instances that have already created



# class Triangle(Geometry): 
#     def __init__(self, a, b, c):  
#         # a, b, c are tuples that represent for 3 vertices of a triangle 
#         # TODO: Your task is to implement the constructor 
#         # super(Triangle, self).__init__(?, ?) 
     
#     def calculate_area(self):
#          #TODO: Your task is to implement an area function


# class Rectangle(Geometry): 
#     def __init__(self, a, b):  
#         # a, b are tuples that represent for top and bottom vertices of a rectangle 
#         # TODO: Your task is to implement the constructor 
#         # super(Rectangle, self).__init__(?, ?) 
     
#     def calculate_area(self): 
#         #TODO: Your task is required implementing an area function 


# class Square(Retangle): 
#     def __init__(self, a, length):  
#         # a is a tuple that represent a top vertex of a square 
#         # length is the side length of a square 
#         # TODO: Your task is to implement the constructor 
#         # super(Square, self).__init__(?, ?) 
     
#     def calculate_area(self): 
#         #TODO: Your task is required implementing an area function


# class Circle(Geometry): 
#     def __init__(self, o, r):  
#         # o is a tuple that represent a center of a circle 
#         # r is the radius of a circle 
#         # TODO: Your task is to implement the constructor 
#         # super(Circle, self).__init__(?, ?)

#     def calculate_area(self): 
#         #TODO: Your task is required implementing an area function


# class Polygon(Geometry): 
#     def __init__(self, points):  
#         # points is a list of tuples that represent vertices of a polygon 
#         # TODO: Your task is to implement the constructor 
#         # super(Polygon, self).__init__(?, ?) 
#     def calculate_area(self): 
#         #TODO: Your task is required implementing an area function

import math
from typing import List, Tuple

class Geometry:
    _instance_count = 0
    
    def __init__(self, name: str = "Shape", points: List[Tuple[float, float]] = None):
        self.name = name
        self.points = points or []
        Geometry._instance_count += 1

    def calculate_area(self) -> float:
        pass
    
    def get_name(self) -> str:
        return self.name
    
    @classmethod
    def count_number_of_geometry(cls) -> int:
        return cls._instance_count

class Triangle(Geometry):
    def __init__(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]):
        super().__init__("Triangle", [a, b, c])

    def calculate_area(self) -> float:
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        x3, y3 = self.points[2]
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)

class Rectangle(Geometry):
    def __init__(self, a: Tuple[float, float], b: Tuple[float, float]):
        width = abs(b[0] - a[0])
        height = abs(b[1] - a[1])
        points = [a, (a[0]+width, a[1]), b, (a[0], a[1]+height)]
        super().__init__("Rectangle", points)

    def calculate_area(self) -> float:
        width = abs(self.points[2][0] - self.points[0][0])
        height = abs(self.points[2][1] - self.points[0][1])
        return width * height

class Square(Rectangle):
    def __init__(self, a: Tuple[float, float], length: float):
        super().__init__(a, (a[0]+length, a[1]+length))
        self.name = "Square"

    def calculate_area(self) -> float:
        return super().calculate_area()

class Circle(Geometry):
    def __init__(self, o: Tuple[float, float], r: float):
        super().__init__("Circle", [o])
        self.radius = r

    def calculate_area(self) -> float:
        return math.pi * self.radius ** 2

class Polygon(Geometry):
    def __init__(self, points: List[Tuple[float, float]]):
        super().__init__("Polygon", points)

    def calculate_area(self) -> float:
        n = len(self.points)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2
    