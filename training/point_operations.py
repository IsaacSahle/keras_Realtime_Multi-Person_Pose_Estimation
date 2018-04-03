class Point(object):
    x = None
    y = None

    def __init__(self,x,y):
        self.x = x
        self.y = y

def addPoints(p1,p2):
    return Point(p1.x + p2.x , p1.y + p2.y)
def addScalar(p,scalar):
    return Point(p.x + scalar,p.y + scalar)
def mulScalar(p,scalar):
    return Point(p.x * scalar,p.y * scalar)