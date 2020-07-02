# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:25:44 2020

@author: alexi
"""

import mymesh_001 as mymesh
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations


def distance_from_boundary_box(vertex):
    distances=np.array([
                        [np.linalg.norm(mesh.points[vertex][:2]-np.array([mesh.points[vertex][0],-1]),2)],
                        [np.linalg.norm(mesh.points[vertex][:2]-np.array([mesh.points[vertex][0],1]),2)],
                        [np.linalg.norm(mesh.points[vertex][:2]-np.array([1,mesh.points[vertex][1]]),2)],
                        [np.linalg.norm(mesh.points[vertex][:2]-np.array([-1,mesh.points[vertex][1]]),2)]
                        ])
    minimum_distance_from_boundary=np.min(distances)
    return minimum_distance_from_boundary

def distance_from_interface(mesh, vertex):
    interface_vertices_coords=mesh.points[mesh.interface_vertices][:,:2]
    vertex_coords=mesh.points[vertex][:2]


    distances=np.array([np.linalg.norm(i-vertex_coords,2) for i in interface_vertices_coords])
    return np.min(distances)

# def get_vertices_with_interface_length(mesh):
#     interface_vertices=[]
#     for vertex in mesh.interior_vertices:
#         distance_from_boundary=distance_from_boundary_box(vertex)
#         distance_from_inter=distance_from_interface(mesh, vertex)
#         if distance_from_inter<distance_from_boundary:
#             interface_vertices.append(vertex)
#     interface_vertices=np.array(interface_vertices)
#     return interface_vertices
#
# def get_interface_target_edge_length_edges(interface_target_edge_length_vertices):
#     interface_edges=[]
#     for vertex in interface_target_edge_length_vertices:
#         objects=mesh.get_neighbourhood(vertex)
#         neighbor_elements=mesh.get_elements()[objects]
#
#         neighbor_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in neighbor_elements]
#         neighbor_edges=[edge for edge in neighbor_edges if edge[0] in interface_target_edge_length_vertices or edge[0] in mesh.interface_vertices  \
#                     and edge[1] in interface_target_edge_length_vertices or edge[1] in mesh.interface_vertices]
#         for edge in neighbor_edges:
#             interface_edges.append(edge)
#
#     interface_edges=np.array(interface_edges)
#     interface_edges = np.unique(interface_edges, axis=0)
#     return interface_edges

def get_interface_target_edgelength(mesh):
    objects = mymesh.objects_boundary_includes_some(mesh.get_triangles(), 1, mesh.interface_vertices)
    elements = mesh.get_triangles()[objects]
    all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
    edges = np.unique(all_edges, axis=0)
    length = np.linalg.norm(mesh.points[edges[:,0]] - mesh.points[edges[:,1]], axis=1)
    return np.mean(length)

mesh = mymesh.read('meshes/circle_inside_square.vtk')
# mesh.cells[0] = mesh.cells[0]._replace(data=np.array([[0],[1],[2],[3]], dtype=np.int))
# mesh.write('meshes/circle_inside_square.vtk')


plt.ion()
mesh.plot_quality(True)








plt.draw()
i=0
# plt.savefig('meshes/animations/collapsing_circle/collapsing_circle{:02}'.format(i))
radius=0.5
# for j in range(20):
while input() == '':
    i += 1
    plt.clf()
    center=np.array([0.0,0.0,0])

    interior_interface_vertices=[]
    for vertex in mesh.interior_vertices:
        distance=np.linalg.norm(mesh.points[vertex]-center)
        if distance<radius:
            interior_interface_vertices.append(vertex)
    interior_interface_vertices=np.array(interior_interface_vertices)

    # translatable_vertices=np.append(mesh.interface_vertices,interior_interface_vertices)




    # for vertex in translatable_vertices:
    #       mesh.translate_vertex_towards_center(vertex,center,epsilon=5e-3,check_injectivity=False)



    for vertex in mesh.interface_vertices:
        mesh.translate_vertex(vertex, mesh.points[vertex]*-0.05, False)




    # interface_edges=[]
    # for k in translatable_vertices:
    #       objects=mesh.get_neighbourhood(k)
    #       neighbor_elements=mesh.get_elements()[objects]

    #       neighbor_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in neighbor_elements]
    #       for edge in neighbor_edges:
    #           interface_edges.append(edge)
    # interface_edges=np.array(interface_edges)
    # interface_edges = np.unique(interface_edges, axis=0)


    # interface_edge_length_vertices=get_vertices_with_interface_length(mesh)
    # interface_edges=get_interface_target_edge_length_edges(interface_edge_length_vertices)
    #
    # length = np.linalg.norm(mesh.points[interface_edges[:,0]] - mesh.points[interface_edges[:,1]], axis=1)
    # target_edgelength_interface = np.mean(length)
    target_edgelength_interface = get_interface_target_edgelength(mesh)
    print("TARGET EDGE LENGTH INTERFACE:",target_edgelength_interface)
    # mesh.interface_edges=interface_edges
    mesh.target_edgelength_inteface=target_edgelength_interface

    elements = mesh.get_triangles()
    all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
    edges=np.unique(all_edges, axis=0)
    # edges=np.array([j for j in edges if j not in mesh.interface_edges])
    is_interface = mymesh.objects_boundary_includes_some(edges, 2, *mesh.interface_vertices)
    edges = edges[~is_interface]
    length = np.linalg.norm(mesh.points[edges[:,0]] - mesh.points[edges[:,1]], axis=1)
    target_edgelength = np.mean(length)
    mesh.target_edgelength=target_edgelength
    print("TARGET EDGE LENGTH ",target_edgelength)

    mesh.coarsen_interface()
    # mesh.refine_interface()
    mesh.coarsen()
    mesh.refine()
    # print(mesh.interface_vertices)

    interior_interface_vertices=[]
    for vertex in mesh.interior_vertices:
        distance=np.linalg.norm(mesh.points[vertex]-center)
        if distance<radius:
            interior_interface_vertices.append(vertex)
    interior_interface_vertices=np.array(interior_interface_vertices)

    # translatable_vertices=np.append(mesh.interface_vertices,interior_interface_vertices)

    # interface_edges=[]
    # for k in translatable_vertices:
    #     objects=mesh.get_neighbourhood(k)
    #     neighbor_elements=mesh.get_elements()[objects]

    #     neighbor_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in neighbor_elements]
    #     for edge in neighbor_edges:
    #         interface_edges.append(edge)

    # interface_edges=np.array(interface_edges)
    # interface_edges = np.unique(interface_edges, axis=0)

    # interface_edge_length_vertices=get_vertices_with_interface_length(mesh)
    # interface_edges=get_interface_target_edge_length_edges(interface_edge_length_vertices)

    # mesh.interface_edges=interface_edges



    # mesh.refine_interface(target_edgelength_interface)


    # interface_edges=[]
    # for k in translatable_vertices:
    #     objects=mesh.get_neighbourhood(k)
    #     neighbor_elements=mesh.get_elements()[objects]

    #     neighbor_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in neighbor_elements]
    #     for edge in neighbor_edges:
    #         interface_edges.append(edge)

    # interface_edges=np.array(interface_edges)
    # interface_edges = np.unique(interface_edges, axis=0)

    # interface_edge_length_vertices=get_vertices_with_interface_length(mesh)
    # interface_edges=get_interface_target_edge_length_edges(interface_edge_length_vertices)

    # mesh.interface_edges=interface_edges
    #
    #
    #
    # mesh.interface_edges=interface_edges



    # mesh.coarsen_interface(target_edgelength_interface)
    mesh.reconnect()
    mesh.smooth_boundary()
    # mesh.smooth_interface()

    mesh.smooth()
    radius=np.linalg.norm(mesh.points[4]-mesh.points[12],2)/2
    #radius=0.5
    # plt.clf()
    mesh.plot_quality(True)
    plt.draw()
    # plt.savefig('meshes/animations/collapsing_circle/collapsing_circle{:02}'.format(i))
