"""Small procedural mesh utilities."""
import numpy as np

def closed(p):
    p=np.asarray(p,float)
    return np.vstack([p,p[0]])


class Mesh:
    def __init__(self):self.faces=[];self.materials=[];self.edges=[];self.edge_materials=[]
    def face(self,p,material):
        self.faces.append(np.asarray(p,float));self.materials.append(material)
    def line(self,p,material='accent'):
        self.edges.append(np.asarray(p,float));self.edge_materials.append(material)
    def prism(self,xy,z0,z1,material,edge=None):
        bottom=np.column_stack([xy,np.full(len(xy),z0)])
        top=np.column_stack([xy,np.full(len(xy),z1)])
        self.face(bottom[::-1],material);self.face(top,material)
        for i in range(len(xy)):
            j=(i+1)%len(xy);self.face([bottom[i],bottom[j],top[j],top[i]],material)
        if edge:self.line(closed(top),edge)
    def cylinder(self,center,radius,length,material,n=18,axis='z',edge=None):
        t=np.linspace(0,2*np.pi,n,endpoint=False)
        radial=np.column_stack([radius*np.cos(t),radius*np.sin(t)])
        if axis=='z':
            ring=np.column_stack([radial,np.zeros(n)]);d=np.array([0,0,length])
        else:
            ring=np.column_stack([np.zeros(n),radial]);d=np.array([length,0,0])
        a=ring+center;b=a+d
        self.face(a[::-1],material);self.face(b,material)
        for i in range(n):
            j=(i+1)%n;self.face([a[i],a[j],b[j],b[i]],material)
        if edge:self.line(closed(b),edge)
