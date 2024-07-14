from src.fvdbmElements import *
from src.fvdbmDynamics import *

Element.dynamics = D2Q9()
testElement = Element()
print(testElement)
