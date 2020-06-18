import math
class Point:
    def __init__(self, x, y):
        self.x=x
        self.y=y
    def distance_between_points(self, point2):
        a = ((self.x - point2.x)**2 + (self.y - point2.y)**2)**0.5
        return a
    def angle_between_points(self, point2):
        return math.atan2(self.y-point2.y, self.x-point2.x)/math.pi

    def return_position(self):
        return self.x, self.y

    def change_position(self, delta_x=10, delta_y=10):
        self.x = self.x + delta_x
        self.y = self.y + delta_y
        
    def __eq__(self, point2):
        if self.x == point2.x and self.y ==point2.y:
            return True
        else:
            return False