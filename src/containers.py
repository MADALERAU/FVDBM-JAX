"""
containers...
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit
from functools import partial
import sys
sys.path.append(".")
from src.dynamics import Dynamics
from utils.utils import *
from src.elements import *

class Container:
    def __init__(self,name:str):
        self.name = name
    def flatten(self):
        pass

class ElementContainer(Container):
    def __init__(self,name:str, elements: list[Element]):
        super().__init__(name)
        self.elements = elements

    def flatten(self):
        assert all(self.elements[0].__class__ == element.__class__ for element in self.elements), "ElementContainer: Element types differ"
        assert len(self.elements) > 0, "ElementContainer: Empty List"
        
        dynamic,static = zip(*[element.flatten() for element in self.elements])

        return pad_stack_tree(dynamic),pad_stack_tree(static)
    
class MultiElementContainer(Container):
    def __init__(self,name:str,*args:Container):
        super().__init__(name)
        self.containers = [arg for arg in args]

    @classmethod
    def create(cls,name:str,*args:Container):

        temp = cls.__init__(name,[arg for arg in args])
        return temp

    def flatten(self):
        dynamic = {}
        static = {}
        for container in self.containers:
            dynamic[container.name],static[container.name] = container.flatten()
        return dynamic,static
    
class NodeContainer(ElementContainer):
    # Class Var
    type_key = {0: "Internal",
                1: "Velocity",
                2: "Pressure"}
    def __init__(self,name:str,elements: list[Element]):
        super().__init__(name,elements)
    
    def flatten(self):
        dynamic, static = super().flatten()
        static, dynamic = splice_split_2dict(static,dynamic,'type',self.type_key)
        return dynamic,static