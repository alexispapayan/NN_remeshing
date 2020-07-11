## -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:18:16 2020

@author: alexi
"""

import mymesh_001 as mymesh
import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy.integrate import quad,quadrature
from math import sin,cos,pi



def integrand_x(t,x,y,T):
     return -(np.sin(np.pi*x)**2)*np.sin(2*np.pi*y)*np.cos(np.pi*t/T)


def integrand_y(t,x,y,T):
     return  np.sin(2*np.pi*x)*(np.sin(np.pi*y)**2)*np.cos(np.pi*t/T)


def integrand_example(x,a):
   return a

if __name__=='__main__':


    mesh = mymesh.read('meshes/circle2.vtk')
    mesh.fixed_vertices=np.array([], dtype=np.int64)
    mesh.boundary_vertices=np.append(mesh.boundary_vertices,0)

    plt.ion()
    mesh.plot_quality(True)
    plt.axis([0,1,0,1])
    plt.draw()

    i=0
    # plt.savefig('meshes/animations/circular_interface/circle{:02}'.format(i))
    T=3 # Parameter for maximum deformation the bigger the more deformed the circle
    t=0.0
    dt=0.02 # timestep

    initial_points=copy.deepcopy(mesh.points)


    x0,y0=initial_points[:,0],initial_points[:,1]

    while(t<T): # and input() == '':
         i+=1

         plt.clf()




         for vertex in mesh.boundary_vertices:
             x=mesh.points[vertex][0]
             y=mesh.points[vertex][1]
             intergral_x=quad(integrand_x,t,t+dt,args=(x,y,T))[0]
             intergral_y=quad(integrand_y,t, t+dt,args=(x,y,T))[0]
             x1=intergral_x+x
             y1=intergral_y+y
             integral_x1=quad(integrand_x,t,t+dt,args=(x1,y1,T))[0]
             integral_y1=quad(integrand_y,t, t+dt,args=(x1,y1,T))[0]
             x2=integral_x1*0.5+x
             y2=integral_x1*0.5+y
             integral_x2=quad(integrand_x,t,t+dt,args=(x2,y2,T))[0]
             integral_y2=quad(integrand_y,t, t+dt,args=(x2,y2,T))[0]
             x3=integral_x2+x
             y3=integral_x2+y
             integral_x3=quad(integrand_x,t,t+dt,args=(x3,y3,T))[0]
             integral_y3=quad(integrand_y,t, t+dt,args=(x3,y3,T))[0]
             mesh.points[vertex][0]=(1/6)*(intergral_x + 2*integral_x1+ 2*integral_x2 + integral_x3)+x
             mesh.points[vertex][1]=(1/6)*(intergral_y + 2*integral_y1+ 2*integral_y2 + integral_y3)+y

         for vertex in mesh.interior_vertices:
             x=mesh.points[vertex][0]
             y=mesh.points[vertex][1]
             intergral_x=quad(integrand_x,t,t+dt,args=(x,y,T))[0]
             intergral_y=quad(integrand_y,t, t+dt,args=(x,y,T))[0]
             x1=intergral_x+x
             y1=intergral_y+y
             integral_x1=quad(integrand_x,t,t+dt,args=(x1,y1,T))[0]
             integral_y1=quad(integrand_y,t, t+dt,args=(x1,y1,T))[0]
             x2=integral_x1*0.5+x
             y2=integral_x1*0.5+y
             integral_x2=quad(integrand_x,t,t+dt,args=(x2,y2,T))[0]
             integral_y2=quad(integrand_y,t, t+dt,args=(x2,y2,T))[0]
             x3=integral_x2+x
             y3=integral_x2+y
             integral_x3=quad(integrand_x,t,t+dt,args=(x3,y3,T))[0]
             integral_y3=quad(integrand_y,t, t+dt,args=(x3,y3,T))[0]
             mesh.points[vertex][0]=(1/6)*(intergral_x + 2*integral_x1+ 2*integral_x2 + integral_x3)+x
             mesh.points[vertex][1]=(1/6)*(intergral_y + 2*integral_y1+ 2*integral_y2 + integral_y3)+y






         mesh.refine()
         mesh.coarsen()
         # if t<T/2:
         mesh.refine_boundary()
         # else:
         mesh.coarsen_boundary()


         mesh.reconnect()


         mesh.smooth_boundary()


         mesh.smooth()


         mesh.plot_quality(True)  # change to plt.axis([0,1,0,1]) to see full effect
         plt.axis([0,1,0,1])
         # plt.plot(mesh.points[mesh.boundary_vertices][:,0],mesh.points[mesh.boundary_vertices][:,1])
         t+=dt
         # plt.savefig('meshes/animations/circular_interface/circle{:02}'.format(i))
