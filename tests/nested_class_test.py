import jax

class A:
    def __init__(self,x,y,b):
        self.x = x
        self.y = y
        self.b = b

    def tree_flatten(self):
        children = (self.x,self.y,self.b)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)

    @jax.jit
    def sum_x(self):
        return self.x
    
class B:
    def __init__(self,x,y,a):
        self.x = x
        self.y = y
        self.a = a

    def tree_flatten(self):
        children = (self.x,self.y,self.a)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)


b = B(10,20,None)
a = A(5,10,b)
b.a = a

print(a.sum_x())