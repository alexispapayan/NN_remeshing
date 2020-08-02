# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:08:47 2020

@author: alexi
"""

import sys
import mymesh_001 as mymesh
import numpy as np
from matplotlib import pyplot as plt
import copy
from itertools import combinations


def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt,np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt

if __name__=='__main__':


    mesh = mymesh.read('meshes/zalesak_disc.vtk')
    # mesh.fixed_vertices=np.array([0,1,2,3])
    # for element in mesh.get_triangles():
    #     if mesh.triangle_area(element)<0:
    #         print("Found negative")
    #         element[0],element[1]=element[1],element[0]
    #
    #
    mesh.interface_vertices = np.append(mesh.interface_vertices, [4, 5, 6, 7, 8])
    mesh.fixed_vertices = np.array([0,1,2,3], dtype=np.int)
    # mymesh.write('meshes/zalesak_disc.vtk',mesh)

    # plt.ion()
    mesh.plot_quality(True)
    # plt.draw()
    # if input() != '':
    #     sys.exit()
    i=0

    plt.savefig('animations/zalesak_disc/zalesak_disc{:02}'.format(i))


    origin=np.array([2,2])
    angular_speed=0.05
    angular_degree=0
    center=np.array([2.0,2.75])
    # vertices_around_interface=[]
    # for vertex in mesh.interior_vertices:
    #     if np.linalg.norm(mesh.points[vertex][:2]-center)<0.6:
    #         vertices_around_interface.append(vertex)
    # vertices_around_interface=np.array(vertices_around_interface)


    while(angular_degree<2*np.pi):
         i+=1

         plt.clf()

         vertices_around_interface=[]
         for vertex in mesh.interior_vertices:
             if np.linalg.norm(mesh.points[vertex][:2]-center)<0.6:
                 vertices_around_interface.append(vertex)
         vertices_around_interface=np.array(vertices_around_interface)

         for vertex in mesh.interface_vertices:
             mesh.points[vertex,:2]=Rotate2D(mesh.points[vertex,:2], origin, angular_speed)
         for vertex in vertices_around_interface:
             mesh.points[vertex,:2]=Rotate2D(mesh.points[vertex,:2], origin,angular_speed)

         center = Rotate2D(center, origin, angular_speed)

         # print(len(np.intersect1d(mesh.interface_vertices, mesh.interior_vertices)))
         mesh.refine()

         # plt.clf()
         # mesh.plot_quality(True)
         # plt.show()

         mesh.coarsen()

         mesh.reconnect()
         mesh.smooth_boundary()
         # mesh.smooth()

         mesh.smooth_interface()

         plt.clf()
         mesh.plot_quality(True)
         # plt.draw()
         # if input() != '':
         #     break
         # plt.axis([0,1,0,1])
         # plt.plot(mesh.points[mesh.boundary_vertices][:,0],mesh.points[mesh.boundary_vertices][:,1])
         angular_degree+=0.05
         plt.savefig('animations/zalesak_disc/zalesak_disc{:02}'.format(i))

         # mesh.write('test.vtk')
