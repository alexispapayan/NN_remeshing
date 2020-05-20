# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:52:58 2020

@author: alexi
"""

import math 
import mymesh
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from scipy.interpolate import interp1d
import copy

#  leave functions maybe need them later to find moving direction
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
	

		denom = (x1-x2) * (x1-x3) * (x2-x3);
		A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
		B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
		C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

		return A,B,C
  
def parabola(a, b, c): 
    print ("Vertex: (" , (-b / (2 * a)) , ", "
        ,(((4 * a * c) - (b * b)) / (4 * a)) , ")" ) 
          
    print ("Focus: (" , (-b / (2 * a)) , ", "
        , (((4 * a * c) - (b * b) + 1) / (4 * a)) , ")" ) 
      
    print ("Directrix: y="
            , (int)(c - ((b * b) + 1) * 4 * a ))     
      
      

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy , oz= origin
    px, py , pz= point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy, pz])

# def translate_to_normal_direction(vertices,tol):
#     new_points=[]
#     for index,vertex in enumerate(vertices[1:-1]):
#         x_normal= mesh.points[vertices[index]][0] +1*tol*(mesh.points[vertices[index+1]][1] -mesh.points[vertices[index]][1]) /np.sqrt(((mesh.points[vertices[index+1]][0] -mesh.points[vertices[index]][0] )**2+(mesh.points[vertices[index+1]][1] -mesh.points[vertices[index]][1] )**2));
#         y_normal = mesh.points[vertices[index]][1] - tol*(mesh.points[vertices[index+1]][1]-mesh.points[vertices[index]][1])/np.sqrt(((mesh.points[vertices[index+1]][0]-mesh.points[vertices[index]][0])**2+(mesh.points[vertices[index+1]][1]-mesh.points[vertices[index]][0])**2));
#         new_point=np.array([x_normal,y_normal,0])
#         mesh.points[vertex]=new_point

def get_edges_close_to_interface(interface_distance):
        vertices_close_to_interface=[]
        for vertex in mesh.interior_vertices:
            for interface_vertex in mesh.interface_vertices:
                vertex_coords=mesh.points[vertex]
                interface_coords=mesh.points[interface_vertex]
                distance=np.linalg.norm(vertex_coords-interface_coords,2)
                if distance<interface_distance and vertex not in vertices_close_to_interface:
                    vertices_close_to_interface.append(vertex)
        interface_edges=[]
        for k in vertices_close_to_interface:
              objects=mesh.get_neighbourhood(k)
              neighbor_elements=mesh.get_elements()[objects]
              
              neighbor_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in neighbor_elements]
              for edge in neighbor_edges:
                  interface_edges.append(edge)
        interface_edges=np.array(interface_edges) 
        interface_edges = np.unique(interface_edges, axis=0)
        mesh.interface_edges=interface_edges
        length = np.linalg.norm(mesh.points[interface_edges[:,0]] - mesh.points[interface_edges[:,1]], axis=1)
        target_edgelength_interface = np.mean(length)
        print("TARGET EDGE LENGTH INTERFACE:",target_edgelength_interface)
        
                
        elements = mesh.get_triangles()
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        edges=np.unique(all_edges, axis=0)
        edges=np.array([j for j in edges if j not in mesh.interface_edges])
        length = np.linalg.norm(mesh.points[edges[:,0]] - mesh.points[edges[:,1]], axis=1)
        target_edgelength = np.mean(length)
        mesh.target_edgelength=target_edgelength
        print("TARGET EDGE LENGTH ",target_edgelength)
        
        return interface_edges,target_edgelength,target_edgelength_interface
                    
    
    
if __name__=="__main__":
    
    

    # code starts here, previous information is to direct the interface if needed later
    mesh = mymesh.read('meshes/moving_interface.vtk')
    
    mesh.cells[0] = mesh.cells[0]._replace(data=np.array([[0],[1],[2],[3]], dtype=np.int))
    mesh.write('meshes/moving_interface_redefined.vtk')
    mesh = mymesh.read('meshes/moving_interface_redefined.vtk')
    mesh.interface_vertices=np.insert(mesh.interface_vertices,0,7)
    mesh.interface_vertices=np.append(mesh.interface_vertices,8)
    
    
    mesh.plot_quality(plot_vertices=True)
    X,Y=mesh.points[mesh.interface_vertices][:,0],mesh.points[mesh.interface_vertices][:,1]
    

    plt.ion()
        

    
    
    plt.draw()
    i=0
    plt.savefig('meshes/animations/shockwave/shockwave{:02}'.format(i))
    
    for j in range(154):
        i+=1
        plt.clf()

        
     
        for vertex in mesh.interface_vertices:
            old_point=mesh.points[vertex]
            x_new=old_point[0]+np.sin(np.pi/4)*1e-2
            y_new=old_point[1]+np.cos(np.pi/4)*1e-2
            new_point=np.array([x_new,y_new,0])
            mesh.points[vertex]=new_point
        elements=mesh.get_elements()
        
        # vertices 7 and 8 area part of the interface and the boundary as well. After moving them for the first time,
        # I delete them from boundary vertices list and then add the elements that are formed by connecting them to their
        # adajacnet edges        
        if i==1:

            point1=copy.deepcopy(mesh.points[7])
            point2=copy.deepcopy(mesh.points[8])
            neighbor_objects=mesh.get_neighbourhood([7,8])
            neighbor_elements=mesh.get_elements()[neighbor_objects]
                                    
            new_points=[point1,point2]
            mesh.points=np.append(mesh.points,new_points,axis=0)
            new_elements=[[29,7,28],[8,10,11]]
            mesh.boundary_vertices = np.delete(mesh.boundary_vertices, np.argwhere(mesh.boundary_vertices == 7))
            mesh.boundary_vertices = np.delete(mesh.boundary_vertices, np.argwhere(mesh.boundary_vertices == 8))
                      
            element_list=elements.tolist()
            elements=np.array(element_list)
            elements=np.append(elements,new_elements,axis=0)
            mesh.set_triangles(elements)
    
        # Get all edges that are close up to a precribed distance from the interface
        mesh.interface_edges,mesh.target_edgelength_inteface,mesh.target_edgelength=get_edges_close_to_interface(3e-1)
                    
        mesh.refine_interface(mesh.target_edgelength_inteface)
        
        # Update to get the list if edges close to interface after refinement
        mesh.interface_edges,_,_=get_edges_close_to_interface(3e-1)
        
        
        # mesh.coarsen_interface(mesh.target_edgelength_inteface)

        #mesh.refine()
        # mesh.coarsen()
        mesh.reconnect()
        mesh.smooth()
        mesh.smooth_boundary()
        mesh.smooth_interface()
 


        
       
     
    
        
        mesh.plot_quality(True)

        
        plt.savefig('meshes/animations/shockwave/shockwave{:02}'.format(i))

    