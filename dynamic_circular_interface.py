# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:18:16 2020

@author: alexi
"""

import mymesh
import numpy as np
from matplotlib import pyplot as plt


if __name__=='__main__':
    
    
    mesh = mymesh.read('meshes/circle.vtk')
    mesh.fixed_vertices=np.array([])
    mesh.boundary_vertices=np.append(mesh.boundary_vertices,0)


    plt.draw()
    i=0
    plt.savefig('meshes/animations/circular_interface/circle{:02}'.format(i))
    
    for j in range(50):
         i+=1
         
         
         plt.clf()

         if i>=25:
             for vertex in mesh.boundary_vertices:
                 if mesh.points[vertex][0]>0:
                     mesh.points[vertex][0]-=0.03
                 else:
                     mesh.points[vertex][0]+=0.03
         else:        
             for vertex in mesh.boundary_vertices:
                 if mesh.points[vertex][0]>0:
                     mesh.points[vertex][0]+=0.03
                 else:
                     mesh.points[vertex][0]-=0.03
        
                 



       
       
         # mesh.refine()
         # mesh.coarsen()
         mesh.reconnect()
         mesh.smooth()
     
            
         mesh.plot_quality(True)
    
            
         plt.savefig('meshes/animations/circular_interface/circle{:02}'.format(i))