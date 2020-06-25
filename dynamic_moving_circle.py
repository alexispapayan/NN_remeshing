import mymesh
import numpy as np
from matplotlib import pyplot as plt





def get_edges_close_to_interface(radius,center):
        vertices_close_to_interface=[]
        for vertex in mesh.interior_vertices:
            vertex_coords=mesh.points[vertex]
            distance=np.linalg.norm(vertex_coords-center,2)
            if distance<2*radius:
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
    
    
move = [0.02, 0, 0]

mesh = mymesh.read('meshes/circle_in_tube2.vtk')
mesh.cells[0] = mesh.cells[0]._replace(data=np.array([[0],[1],[2],[3]], dtype=np.int))
mesh.write('meshes/circle_in_tube2.vtk')

i = 0

plt.ion()
mesh.plot_quality(True)
plt.draw()
plt.savefig('meshes/animations/circle_in_tube/circle_in_tube_{:02}'.format(i))
radius=0.25
iterations=50
center=np.array([0.5,0.5,0])

mesh.interface_edges,mesh.target_edgelength_inteface,mesh.target_edgelength=get_edges_close_to_interface(radius,center)


for j in range(iterations):
    i += 1
    plt.clf()

    mesh.interface_edges,_,_=get_edges_close_to_interface(radius,center)

    interior_interface_vertices=[]
    for vertex in mesh.interior_vertices:
        if np.linalg.norm(mesh.points[vertex]-center,2)<radius:
            interior_interface_vertices.append(vertex)
    interior_interface_vertices=np.array(interior_interface_vertices)       
    
    translatable_vertices=np.append(mesh.interface_vertices,interior_interface_vertices)

    for vertex in translatable_vertices:
        mesh.translate_vertex(vertex, move, False)
    

    
    mesh.refine()
    mesh.coarsen()
    mesh.reconnect()
    
    mesh.interface_edges,_,_=get_edges_close_to_interface(radius,center)
    mesh.refine_interface(mesh.target_edgelength_inteface)
   
    
   # mesh.interface_edges,_,_=get_edges_close_to_interface(radius,center)
    # mesh.coarsen_interface(mesh.target_edgelength_inteface)

    
    mesh.smooth_boundary()
    mesh.smooth()

    mesh.smooth_interface()

    center=np.array([0.5+i*0.02,0.5,0])
    
    mesh.plot_quality(True)
    plt.draw()
    plt.savefig('meshes/animations/circle_in_tube/circle_in_tube_{:02}'.format(i))
