
import numpy as np
import matplotlib
import sys
import os
import copy
from collections import OrderedDict
from itertools import permutations
from matplotlib import pyplot as plt
from numpy import pi,sin,cos,sqrt

import triangle as tri
import triangle.plot as plot
import Triangulation


from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn  import manifold
# from numba import jit
from math import acos

# sys.path.insert(0, '../network_datasets/')
from Neural_network import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from functools import lru_cache

import pdb
'''
  ======================================
  Main idea of finding the connectivity:
  ======================================

1) Build set of elements and edges
2) Sort edges according to maximum quality
3) for each pair of the edge put into set of edges  of the maximum quality
4) Put also to the set of elements the element that is formed
5) Proceed to the next edge and check if the edge formed from each pair already exist
6) If a pair of edges already exists proceed to next edge
7) If yes proceed to the next element

(*) Add function computing the quality of the mesh given that every point of a contour is connected
    to a point of the mesh -> normalize the qualities for each point -> Add as parameter to neural network                                             Issues:

(1) Avoiding the consideration of triangles that are invalid ( by setting quality to 0):

     1st Approach (Failed):
     See if formed triangle contains points of the polygon

     2nd Approach :
     Calculate the angles of the polygon. Once that is done, any edge departing from a point in the polygon
     can have a greater angle from the departed edge greater than the edge has with the pre-existing edges


(2) Avoiding interseactions when creating elements:

    Here the main idea is whenever a new element is created to check if it includes a forabideen vertex. If it does
    the new element can't be formed and we porceed to check the connection with the second smallest quality.

   ==========================================================================
      How to check if the vertex is locked ( So no new connections with it.)
   ==========================================================================


    So the idea is that given  a vertex we check the edges and the elements that include those edges.
    If start from an edge of the contour and end up to an edge of the contour again then the vertex


    + For every vertex of the element that is to be created  and

        for vtx in element:

            if closed_ring(vtx,adj_vertices,elements):

                  don't created element
                  proceed to connection with second most great quality
                  if doesn't exist:
                  proceeed  to next edge

    + for the vtx and the adjacent v1 look for the element
        (vtx,v1,v2) ->  (vtx, v2) -> (vtx,v3,v2) -> (vtx,v3)-> ... -> (vtx,vn)
        Check if end is edge of contour if yes then vtx is forbidden -> insert to forbidden vertices

      '''



"""
Idea about triangulation with point:


The key issue is that even if we can recognize the inner points with as open vertices we have to assign them either to subpolygons
or elements to allow for reiteration of the algorithm.
So what we can do is connect vertices with the inner point and see if they intersect inner edges.

"""

@torch.no_grad()
def retriangulate_with_interior(contour, *args):
    try:
        net = get_connectivity_network(contour.shape[0], len(args))
    except FileNotFoundError:
        if len(args) == 1:
            # print('Using fallback naive retriangulation')
            return simple_retriangulate(contour, *args)
        else:
            raise ValueError('Network for {} edges and {} interior points not trained'.format(contour.shape[0], len(args)))

    procrustes_transform,_,_ = get_procrustes_transform(contour)
    procrustes = procrustes_transform(contour)
    inner = procrustes_transform(np.array(args))
    input = torch.tensor(np.concatenate([np.asarray(np.append(procrustes, procrustes[0][None,:], axis=0), dtype=np.float32), np.asarray(inner, dtype=np.float32)], axis=0)[None,None,:,:])
    # with open('contour.pkl', 'wb') as file:
    #     pickle.dump(input, file)
    table = net(input)

    table = table[0].numpy().reshape([contour.shape[0], contour.shape[0]+len(args)])

    # table,_=quality_matrix(procrustes,inner)

    ordered_matrix = order_quality_matrix(table, procrustes, np.concatenate([procrustes, inner], axis=0))
    new_elements, sub_elements = triangulate(procrustes, inner, ordered_matrix, recursive=True)
    # print(new_elements, sub_elements)

    return np.array(list(new_elements) + sub_elements, dtype=np.int)

def simple_retriangulate(contour, *args):
    if len(args) == 0:
        raise ValueError('No interior points given')
    elif len(args) > 1:
        raise NotImplementedError('Refinement with multiple points not supported')
    else:
        n = len(contour)
        triangles = np.array([[i, (i+1) % n, n] for i in range(n)])
        return triangles

# In[3]:

def BCE_accuracy(model,variable,labels):
    net.eval()
    predictions=model(variable).data.numpy()
    predictions[np.where(predictions>0.5)]=1
    predictions[np.where(predictions<=0.5)]=0
    diff=labels-predictions
    correct_prediction=0
    for i in diff:
        if (not i.any()):
            correct_prediction+=1
    net.train()
    return  100*correct_prediction/variable.size()[0],diff

# Check if two lines intersect
def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# A bit more sophisticated method to see if lines do intersect (https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/)

def on_segment(p, q, r):
    '''Given three colinear points p, q, r, the function checks if
    point q lies on line segment "pr"
    '''
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) -
            (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1   # clockwise
    else:
        return 2  # counter-clockwise

def do_intersect(p1, q1, p2, q2):
    '''Main function to check whether the closed line segments p1 - q1 and p2
       - q2 intersect'''
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4) and not (o1 == 0 and on_segment(p1, p2, q1)) and not (o2 == 0 and on_segment(p1, q2, q1)) and not (o3 == 0 and on_segment(p2, p1, q2)) and not (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False # Doesn't fall in any of the above cases




def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees
def angle_counterclockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0:
        return 360-inner
    else:
        return inner




def extract_barycenter(points,nb_of_points):

    barycenters=[]

    polygons=points.reshape(len(points),nb_of_points,2)
    for polygon in polygons:
        barycenter=np.array([polygon[:,0].sum()/nb_of_points,polygon[:,1].sum()/nb_of_points])
        barycenter_triangles=[]
        barycenters.append(barycenter)
        for i in range(nb_of_points):
            barycenter_triangles.append(np.array([barycenter,polygon[i],polygon[(i+1)%nb_of_points]]))
    barycenters=np.array(barycenters).reshape(len(points),1,2)

    return np.array(barycenter)



def sort_points(point_coordinates,nb_of_points):
    polygon=point_coordinates.reshape(len(point_coordinates),nb_of_points,2)
    barycenters=extract_barycenter(point_coordinates,nb_of_points)
    angles=[]
    polygons=point_coordinates.reshape(len(point_coordinates),nb_of_points,2)
    vectors=polygons-barycenters

    for  barycenter_vectors in vectors:
        for vector in barycenter_vectors:
            angles.append(angle_counterclockwise(np.array([1,0]),vector))

    angles=np.array(angles).reshape(len(vectors),nb_of_points,1)
    point_coordinates_with_angles=np.dstack([polygons,angles])
    point_coordinates_sorted=[]
    for points in point_coordinates_with_angles:
        points_sorted=np.array(sorted(points,key=lambda x: x[2]))
        points_sorted=points_sorted[:,0:2]
        point_coordinates_sorted.append(points_sorted.reshape(1,nb_of_points,2))
    return np.array(point_coordinates_sorted)






def connectivity_information(triangulation,print_info=False):

    segments= tuple(triangulation['segments'].tolist())
    triangles=tuple(triangulation['triangles'].tolist())
    vertices=triangulation['vertices']

    connect_info={str(r):[0 for i in range(len(vertices))] for r in tuple(triangulation['segments'].tolist())}
    for segment in segments:
        for triangle in triangles:
            if set(segment).issubset(set(triangle)):
                connection=set(triangle)-set(segment)
                # if print_info: print("segment:",segment,"is connected to:",connection,"to form triangle:",triangle)
                connect_info[str(segment)][tuple(connection)[0]]=1
    return connect_info



def get_labels(triangulation,connect_info):
    indices=[]
    vertices=list(range(triangulation['vertices'].shape[0]))
    for i in triangulation['segments']:
           indices.append(set(vertices)-set(i))
    labels=[]
    list_values=list(connect_info.values())
    for i in range(len(list_values)):
        for j in indices[i]:
            labels.append(list_values[i][j])
    return  labels



def rot(theta):
    return np.array([[cos(theta),-sin(theta)],
                     [sin(theta),cos(theta)]])




def get_reference_polygon(nb_of_points,plot=False):
    angles=np.empty(nb_of_points)
    points=np.empty([nb_of_points,2])
    plot_coords=np.empty([nb_of_points,2])
    indices=[]
    angle_division=2*pi/nb_of_points

    for i in range(nb_of_points):
        angle=i*angle_division
        angles[i]=angle
        point=np.array([1,0]) #pick edge length of 1
        points[i]=np.dot(rot(angle),point.T)  #rotate it according to the  chosen angle
        indices.append(i)

    if plot==True:
        plot_coords=np.vstack([points,points[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))

    return points



def generate_contour(nb_of_points,plot=False):


    angles=np.empty(nb_of_points)
    points=np.empty([nb_of_points,2])
    plot_coords=np.empty([nb_of_points,2])
    indices=[]
    angle_division=2*pi/nb_of_points

    for i in range(nb_of_points):
        angle=((i+1)*angle_division-i*angle_division)*np.random.random_sample()+i*angle_division
        angles[i]=angle
        point=np.array([np.random.uniform(0.3,1),0]) #pick random point at (1,0)
       #point=np.array([1,0]) #pick edge length of 1

        points[i]=np.dot(rot(angle),point.T)  #rotate it according to the  chosen angle
        indices.append(i)

    if plot==True:
        plot_coords=np.vstack([points,points[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))

    return points


def plot_contour(contour):
    plot_coords=np.vstack([contour,contour[0]])
    (s,t)=zip(*plot_coords)
    plt.plot(s,t)
    indices=[i for i in range(contour.shape[0])]
    for index,i in enumerate(indices):
        plt.annotate(str(i),(s[index],t[index]))



def get_procrustes_transform(polygon):
    centralised_ref_polygon, norm_ss_ref_polygon = get_reference_data(polygon.shape[0])

    mu_polygon = polygon.mean(0)
    centralised_polygon = polygon - mu_polygon
    ss_polygon = (centralised_polygon**2).sum()
    norm_ss_polygon = np.sqrt(ss_polygon)
    centralised_polygon /= norm_ss_polygon

    A = np.dot(centralised_ref_polygon.T, centralised_polygon)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V= Vt.T
    R = np.dot(V, U.T)
    traceTA = s.sum()
    Rinv = R.T

    def procrustes_transform(polygon):
        return norm_ss_ref_polygon * traceTA * np.dot((polygon - mu_polygon) / norm_ss_polygon, R)

    def inverse_transform(polygon):
        return np.dot(polygon, Rinv) / (norm_ss_ref_polygon * traceTA) * norm_ss_polygon + mu_polygon

    def tangent_transform(tangents):
        return np.dot(tangents, R)

    return procrustes_transform, inverse_transform, tangent_transform

@lru_cache(maxsize=8)
def get_reference_data(edges):
    ref_polygon = get_reference_polygon(edges)
    norm_ss_ref_polygon = np.sqrt((ref_polygon**2).sum())
    return ref_polygon / norm_ss_ref_polygon, norm_ss_ref_polygon


def apply_procrustes(polygon_points,plot=False):

    # Get reference polygona and adjust any random poygon to that
    ref_polygon=get_reference_polygon(polygon_points.shape[0])


    #Mean of each coordinate
    mu_polygon = polygon_points.mean(0)
    mu_ref_polygon = ref_polygon.mean(0)

    #Centralize data to the mean
    centralised_ref_polygon_points = ref_polygon-mu_ref_polygon
    centralised_polygon_points = polygon_points-mu_polygon

    #Squared sum of X-mean(X)
    ss_ref_polygon_points = (centralised_ref_polygon_points**2.).sum()
    ss_polygon_points = (centralised_polygon_points**2.).sum()


    #Frobenius norm of X
    norm_ss_ref_polygon_points = np.sqrt(ss_ref_polygon_points)
    norm_ss_polygon_points = np.sqrt(ss_polygon_points)


    # scale to equal (unit) norm
    centralised_ref_polygon_points /=norm_ss_ref_polygon_points
    centralised_polygon_points /=norm_ss_polygon_points


    #Finding best rotation to superimpose on regular triangle
    #Applying SVD to the  matrix
    A = np.dot(centralised_ref_polygon_points.T, centralised_polygon_points)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V=Vt.T
    R = np.dot(V,U.T)


    traceTA = s.sum()
    indices=[i for i in range(polygon_points.shape[0])]



    polygon_transformed =norm_ss_ref_polygon_points*traceTA*np.dot(centralised_polygon_points,R)+mu_ref_polygon

    if plot==True:
        plot_coords=np.vstack([polygon_transformed,polygon_transformed[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))

    return polygon_transformed



def rotation_projection(nb_of_points):

    contours=[]
    ref_polygon=get_reference_polygon(nb_of_points,True)
    random_contour=generate_contour(nb_of_points,True)
    contours.append(ref_polygon)

    contours.append(random_contour)

    for i in range(1,nb_of_points):
        rotated_contour=np.dot(rot(pi/i),random_contour.T).T
        plot_contour(rotated_contour)
        contours.append(rotated_contour)

    contours=np.array(contours)

    # Projecting via Isomap

    contours=contours.reshape(contours.shape[0],2*nb_of_points)
    isomap=manifold.Isomap(n_neighbors=2,n_components=2)
    isomap.fit(contours)
    Polygons_manifold_2D=isomap.transform(contours)
    Polygons_manifold_2D=isomap.transform(contours)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(Polygons_manifold_2D[:,0],Polygons_manifold_2D[:,1],color=['red' if i == 0 else 'blue'
                                                                      for i,_ in enumerate(Polygons_manifold_2D)])
    ax.set_title('Isomap projection without procrustes')
    ax.set_xlabel('1st Component')
    ax.set_ylabel('2nd Component')

    # Applying procrustes
    procrustes_points=np.empty([contours.shape[0],nb_of_points,2])
    for i,contour in enumerate(contours):
        procrustes_points[i]=apply_procrustes(contour,True)
    ref_polygon=get_reference_polygon(nb_of_points,True)

def scale_projection(polygon):
    pass









# In[25]:


def contains_points(triangle,polygon):
    try:
        hull=ConvexHull(triangle)
    except:
        # print("Invalid Convex hull")
        return True
    hull_path=Path(triangle[hull.vertices])
    set_polygon=set(tuple(i) for i in polygon)
    set_triangle=set(tuple(i) for i in triangle)
    #print(set_polygon,set_triangle)
    difference=set_polygon-set_triangle

    if len(difference)==0:
        return False

    for i in difference:
        if hull_path.contains_point(i):
            return True
            break
    return False

# @jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
def is_counterclockwise(polygon):
    area = 0
    counterclokwise=False
    for index,_ in enumerate(polygon):
        second_index=(index+1)%len(polygon)
        area+=polygon[index][0]*polygon[second_index][1]
        area-=polygon[second_index][0]*polygon[index][1]
    if area/2<0:
        counterclokwise=False
    else:
        counterclokwise=True
    return counterclokwise



def compute_edge_lengths(pt1,pt2):
    return np.linalg.norm(pt1-pt2)

def compute_edge_lengths2(triangle):
    edgelengths2=np.empty([2,3])
    for i in range(2):
        for j in range(i+1,3):
            eij=triangle[j]-triangle[i]
            edgelengths2[i][j]=np.dot(eij,eij)
    return edgelengths2



def compute_triangle_normals(triangle):

    e01=triangle[1]-triangle[0]
    e02=triangle[2]-triangle[0]

    e01_cross_e02=np.cross(e01,e02)

    return e01_cross_e02


def compute_triangle_area(triangle):

    e01=triangle[1]-triangle[0]
    e02=triangle[2]-triangle[0]

    e01_cross_e02=np.cross(e01,e02)

    # Omit triangles that are inverted (out of the domain)
    if e01_cross_e02<0:
        return 0


    e01_cross_e02_norm=np.linalg.norm(e01_cross_e02)


    return e01_cross_e02_norm/2

def compute_triangle_quality(triangle,polygon=None):

    if polygon is None:
        polygon=triangle


    factor=4/sqrt(3)
    area=compute_triangle_area(triangle)


    sum_edge_lengths=0
    edge_length2=compute_edge_lengths2(triangle)
    for i in range(2):
        for j in range(i+1,3):
            sum_edge_lengths+=edge_length2[i][j]



    lrms=sqrt(sum_edge_lengths/3)
    lrms2=lrms**2
    quality=area/lrms2

    return quality*factor

def compute_minimum_quality_triangle(triangle,polygon,polygon_with_inner_points):
    if polygon is None:
        polygon=triangle


    #barycenter=polygon.sum(0)/polygon.shape[0]
    #polygon_with_point=np.vstack([polygon,inner_point])
    # The incoming triangle has an edge [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)

    indices=[]


    for point in triangle:
        for index,point_in_polygon in enumerate(polygon_with_inner_points):
              if np.allclose(point,point_in_polygon):
                  indices.append(index)



    p0,p1,p2=indices[0],indices[1],indices[2]

    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    # print(indices)
 #   if p2< polygon_with_inner_points.shape[0]:
        # Checking if edges of connected poiints form an angle bigger than the polygon angles
 #       if (polygon_angles[p0]<calculate_angle(polygon[p0],polygon[p1],polygon[p2])
  #          or polygon_angles[p1]<calculate_angle(polygon[p1],polygon[p0],polygon[p2])):
   #         print("Spotted inverted triangle: {}".format([p0,p1,p2]))
   #         return 0


    #    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p0])
     #       or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p1])
      #      ):
       #     print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        #    return 0

   #     if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p0])
    #        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p1])):
     #       print("Spotted inverted triangle: {}".format([p0,p1,p2]))
      #      return 0

    area=compute_triangle_area(triangle)

    if area==0:
        return 0


    Invalid_triangulation=False

    try:
        contains_points(triangle,polygon_with_inner_points)
    except:
        # print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True

    if Invalid_triangulation:
         return 0

    if contains_points(triangle,polygon_with_inner_points):
        return 0

    triangles_in_mesh=[]
    triangles_in_mesh.append(triangle)
    contour_connectivity=get_contour_edges(polygon)
    contour_connectivity=np.vstack([contour_connectivity,[p0,p2],[p1,p2]])
    hole=np.array([(triangle.sum(0))/3])
    shape=dict(holes=hole,vertices=polygon_with_inner_points,segments=contour_connectivity)
    t = tri.triangulate(shape, 'pq0')


    try:
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon_with_inner_points[np.asarray([triangle_index])])
    except :
        # print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True

    triangle_qualities=[]
    for triangle in triangles_in_mesh:
        triangle.resize(3,2)
        triangle_quality=compute_triangle_quality(triangle)
        triangle_qualities.append(triangle_quality)

    if Invalid_triangulation:
        mean_quality,minimum_quality=0,0
    else:
        triangle_qualities=np.array(triangle_qualities)
        mean_quality=triangle_qualities.mean()
        minimum_quality=triangle_qualities.min()

    return minimum_quality


def compute_mean_quality_triangle(triangle,polygon,polygon_with_inner_points):
    if polygon is None:
        polygon=triangle


    #barycenter=polygon.sum(0)/polygon.shape[0]
    #polygon_with_point=np.vstack([polygon,inner_point])
    # The incoming triangle has an edge [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)

    indices=[]


    for point in triangle:
        for index,point_in_polygon in enumerate(polygon_with_inner_points):
              if np.allclose(point,point_in_polygon):
                  indices.append(index)



    p0,p1,p2=indices[0],indices[1],indices[2]

    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    # print(indices)
 #   if p2< polygon_with_inner_points.shape[0]:
        # Checking if edges of connected poiints form an angle bigger than the polygon angles
 #       if (polygon_angles[p0]<calculate_angle(polygon[p0],polygon[p1],polygon[p2])
  #          or polygon_angles[p1]<calculate_angle(polygon[p1],polygon[p0],polygon[p2])):
   #         print("Spotted inverted triangle: {}".format([p0,p1,p2]))
   #         return 0


    #    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p0])
     #       or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p1])
      #      ):
       #     print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        #    return 0

   #     if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p0])
    #        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p1])):
     #       print("Spotted inverted triangle: {}".format([p0,p1,p2]))
      #      return 0

    area=compute_triangle_area(triangle)

    if area==0:
        return 0

    if contains_points(triangle,polygon_with_inner_points):
        return 0

    triangles_in_mesh=[]
    triangles_in_mesh.append(triangle)
    contour_connectivity=get_contour_edges(polygon)
    contour_connectivity=np.vstack([contour_connectivity,[p0,p2],[p1,p2]])
    hole=np.array([(triangle.sum(0))/3])
    shape=dict(holes=hole,vertices=polygon_with_inner_points,segments=contour_connectivity)
    t = tri.triangulate(shape, 'pq0')

    Invalid_triangulation=False

    try:
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon_with_inner_points[np.asarray([triangle_index])])
    except :
        # print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True

    triangle_qualities=[]
    for triangle in triangles_in_mesh:
        triangle.resize(3,2)
        triangle_quality=compute_triangle_quality(triangle)
        triangle_qualities.append(triangle_quality)

    if Invalid_triangulation:
        mean_quality,minimum_quality=0,0
    else:
        triangle_qualities=np.array(triangle_qualities)
        mean_quality=triangle_qualities.mean()
        minimum_quality=triangle_qualities.min()

    return mean_quality






def compute_qualities_triangle(triangle,polygon,polygon_with_inner_points):
    if polygon is None:
        polygon=triangle


    #barycenter=polygon.sum(0)/polygon.shape[0]
    #polygon_with_point=np.vstack([polygon,inner_point])
    # The incoming triangle has an edge [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)

    indices=[]


    for point in triangle:
        for index,point_in_polygon in enumerate(polygon_with_inner_points):
              if np.allclose(point,point_in_polygon):
                  indices.append(index)



    p0,p1,p2=indices[0],indices[1],indices[2]

    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    # print(indices)


    area=compute_triangle_area(triangle)

    if area==0:
        return [0,0]

    if contains_points(triangle,polygon_with_inner_points):
        return [0,0]

    triangles_in_mesh=[]
    triangles_in_mesh.append(triangle)
    contour_connectivity=get_contour_edges(polygon)
    contour_connectivity=np.vstack([contour_connectivity,[p0,p2],[p1,p2]])
    hole=np.array([(triangle.sum(0))/3])
    shape=dict(holes=hole,vertices=polygon_with_inner_points,segments=contour_connectivity)
    t = tri.triangulate(shape, 'pq0')

    Invalid_triangulation=False

    try:
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon_with_inner_points[np.asarray([triangle_index])])
    except :
        # print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True

    triangle_qualities=[]
    for triangle in triangles_in_mesh:
        triangle.resize(3,2)
        triangle_quality=compute_triangle_quality(triangle)
        triangle_qualities.append(triangle_quality)

    if Invalid_triangulation:
        mean_quality,minimum_quality=0,0
    else:
        triangle_qualities=np.array(triangle_qualities)
        mean_quality=triangle_qualities.mean()
        minimum_quality=triangle_qualities.min()



    return [minimum_quality,mean_quality]


def compute_delaunay_minimum_quality_with_points(polygon,points):
        triangles_in_mesh=[]
        contour_connectivity=get_contour_edges(polygon)
        polygon_with_points=np.vstack([polygon,points])
        shape=dict(vertices=polygon_with_points,segments=contour_connectivity)

        t = tri.triangulate(shape, 'pq0')

        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon_with_points[np.asarray([triangle_index])])

        triangle_qualities=[]
        for triangle in triangles_in_mesh:
            triangle.resize(3,2)
            triangle_quality=compute_triangle_quality(triangle)
            triangle_qualities.append(triangle_quality)

        triangle_qualities=np.array(triangle_qualities)

        minimum_quality=triangle_qualities.min()

        return minimum_quality


def compute_delaunay_mean_quality_with_points(polygon,points):
        triangles_in_mesh=[]
        contour_connectivity=get_contour_edges(polygon)
        polygon_with_points=np.vstack([polygon,points])
        shape=dict(vertices=polygon_with_points,segments=contour_connectivity)

        t = tri.triangulate(shape, 'pq0')

        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon_with_points[np.asarray([triangle_index])])

        triangle_qualities=[]
        for triangle in triangles_in_mesh:
            triangle.resize(3,2)
            triangle_quality=compute_triangle_quality(triangle)
            triangle_qualities.append(triangle_quality)

        triangle_qualities=np.array(triangle_qualities)

        mean_quality=triangle_qualities.mean()

        return mean_quality



# Quality of elements formed by connecting each edge with one of the other points of the contour
def quality_matrix(polygon,inner_points=None,compute_minimum=True ,normalize=False):
    #polygon=apply_procrustes(polygon,False)

   # if inner_points is None:
    #    inner_points=polygon.sum(0)/polygon.shape[0]

    contour_connectivity=np.array(list(tuple(i) for i in get_contour_edges(polygon)))

    for point in inner_points:
        polygon_with_inner_points=np.vstack([polygon,inner_points])
    #print(polygon_with_point)

    quality_matrix=np.zeros([contour_connectivity.shape[0],polygon_with_inner_points.shape[0]])
    #area_matrix=np.zeros([contour_connectivity.shape[0],polygon.shape[0]])
    normals_matrix=np.zeros([contour_connectivity.shape[0],polygon_with_inner_points.shape[0]])

    list_of_triangles=[]

    for index,edge in enumerate(contour_connectivity):
        # Not omitting non triangles because either way their quality is zero
        triangles_to_edge_indices=[[*edge,i] for i in range(polygon_with_inner_points.shape[0]) ]



        #print(triangles_to_edge_indices)
        triangles_to_edge_indices=np.asarray(triangles_to_edge_indices)

        triangles=polygon_with_inner_points[triangles_to_edge_indices]
        list_of_triangles.append(triangles)


    list_of_triangles=np.array(list_of_triangles)

    # print(list_of_triangles)
    if compute_minimum:
        for i,triangles in enumerate(list_of_triangles):
            for j,triangle in enumerate(triangles):
                quality_matrix[i,j]=compute_minimum_quality_triangle(triangle,polygon,polygon_with_inner_points)
    else:
         for i,triangles in enumerate(list_of_triangles):
            for j,triangle in enumerate(triangles):
                quality_matrix[i,j]=compute_triangle_quality(triangle,polygon)

            #area_matrix[i,j]=compute_triangle_area(triangle)
            #normals_matrix[i,j]=compute_triangle_normals(triangle)



    sum_of_qualities=quality_matrix.sum(1)

    if normalize is True:
        for i,_ in enumerate(quality_matrix):
            quality_matrix[i]/=sum_of_qualities[i]

    return quality_matrix,normals_matrix


def quality_matrices(polygon,inner_points=None,compute_minimum=True ,normalize=False):
    #polygon=apply_procrustes(polygon,False)

   # if inner_points is None:
    #    inner_points=polygon.sum(0)/polygon.shape[0]

    contour_connectivity=np.array(list(tuple(i) for i in get_contour_edges(polygon)))

    for point in inner_points:
        polygon_with_inner_points=np.vstack([polygon,inner_points])
    #print(polygon_with_point)


    min_quality_matrix=np.zeros([contour_connectivity.shape[0],polygon_with_inner_points.shape[0]])
    mean_quality_matrix=np.zeros([contour_connectivity.shape[0],polygon_with_inner_points.shape[0]])


    list_of_triangles=[]

    for index,edge in enumerate(contour_connectivity):
        # Not omitting non triangles because either way their quality is zero
        triangles_to_edge_indices=[[*edge,i] for i in range(polygon_with_inner_points.shape[0]) ]



        #print(triangles_to_edge_indices)
        triangles_to_edge_indices=np.asarray(triangles_to_edge_indices)

        triangles=polygon_with_inner_points[triangles_to_edge_indices]
        list_of_triangles.append(triangles)


    list_of_triangles=np.array(list_of_triangles)

    # print(list_of_triangles)
    if compute_minimum:
        for i,triangles in enumerate(list_of_triangles):
            for j,triangle in enumerate(triangles):
                qualities=compute_qualities_triangle(triangle,polygon,polygon_with_inner_points)
                min_quality_matrix[int(i),int(j)]=qualities[0]
                mean_quality_matrix[int(i),int(j)]=qualities[1]


            #area_matrix[i,j]=compute_triangle_area(triangle)
            #normals_matrix[i,j]=compute_triangle_normals(triangle)



    return min_quality_matrix,mean_quality_matrix



def check_edge_validity(edge,polygon,set_edges,interior_edges):
    # Check if new edges are already in the set
    found_in_set=False
    found_in_interior_set=False
    for index in range(len(polygon)+1):
        occuring_index=index

        edge1,edge2=tuple(permutations((edge[0],index))),tuple(permutations((edge[1],index)))
        condition1= edge1[0] in set_edges or edge1[1] in set_edges
        condition2= edge2[0] in set_edges or edge2[1] in set_edges
        condition3= edge1[0] in interior_edges or edge1[1] in interior_edges
        condition4= edge2[0] in interior_edges or edge2[1] in interior_edges


            # both edges are found in the list of set of edges (Invalid)
        if (condition1 and condition2):
            found_in_set=True
            occuring_index=index


        # both edges are found in the list of interior edges created
        if (condition3 and condition4):
            found_in_interior_set=True
            occuring_index=index

        if found_in_interior_set or found_in_set:
            break
    return found_in_interior_set,found_in_set,occuring_index

# def triangulate(polygon,points,ordered_quality_matrix,recursive=True,plot_mesh=False):
#     set_edges=set(tuple(i) for i in get_contour_edges(polygon))
#     interior_edges=set()
#     set_elements=set()
#     set_locked_vertices=set()
#     set_forbidden_intersections=set()
#     set_interior_edge_with_inner_point=set()
#     set_orphan_vertices=set()
#     # print("initial set edges:", set_edges)
#
#
#
#     polygon_with_points=np.vstack([polygon,points])
#
#
#     # print("meshing polygon: " , polygon," with inner points :", points)
#
#
#     for edge in ordered_quality_matrix.keys():
#
#         found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)
#
#         for qualities_with_edges in ordered_quality_matrix[edge][0]:
#
#             element_created=False
#
#             target_vtx=qualities_with_edges[1]
#
#             if target_vtx==edge[0] or target_vtx==edge[1]:
#                 continue
#
#             # print("Edge:",edge,"targeting:",target_vtx)
#
#             if found_in_interior_set: #and target_vtx!=polygon.shape[0]:
#                 element=(edge[0],edge[1],index)
#                 set_elements.add(element)
#                 # print("Element inserted:",element)
#                 continue
#
#             ############# Could be a Triangle with inner points inside we want to mesh this as well ################
#             if polygon.shape[0]>3:
#                 if found_in_set and not found_in_interior_set: #and target_vtx!=polygon.shape[0]:
# #                    if(index != target_vtx) and index<=polygon.shape[0] :
# #                        print('found',(edge[0],index),(edge[1],index),"Canceling creation")
# #                        continue
#                     if(index != target_vtx) and index<=polygon.shape[0] :
#                         # print('found',(edge[0],index),(edge[1],index),"Canceling creation")
#                         continue
#
#
#
#
#
#             # Passed edges checking
#             # Proceed to check vertices
#             temp_element=(edge[0],edge[1],target_vtx)
#             # print(temp_element)
#             existing_element=False
#             for element in set_elements:
#                 if set(temp_element)== set(element):
#                     # print("Element {} already in set".format(element))
#                     existing_element=True
#                     break
#             if existing_element:
#                 break
#
#
#
#             if target_vtx in set_locked_vertices:
#                 # print(" Target vertex {} is locked".format(target_vtx))
#                 continue
#             set_elements.add(temp_element)
#
#             triangle_indices=np.asarray(temp_element)
#             triangle=polygon_with_points[triangle_indices]
#             # print(triangle)
#
#
#
#             ################# posteriori checks for NN ####################################
#             # Checking for invalid connections outside of the contour
#
#             if  compute_triangle_area(triangle)==0:
#                 # print("found zero area triangle", temp_element)
#                 set_elements.remove(temp_element)
#                 continue
#
#             contains_inner_points=False
#             for index in temp_element:
#                 if index>=polygon.shape[0]:
#                     contains_inner_points=True
#
#
# #           # Checking if element includes interior point, if it does then check if there is an element in set element formed by
#              # the inner point and the edges of temp element
#             is_inside=False
#             contains_element=False
# #            bad_quality_after=False
#             for point_index,point in enumerate(points):
#                 is_inside=ray_tracing(point[0],point[1],triangle)
#
# #                if is_inside:
# #                    barycenter=np.array([triangle[:,0].sum()/3,triangle[:,1].sum()/3])
# #                    if np.linalg.norm(barycenter-point)>0.05:
# #                        bad_quality_after=True
# #                        break
# #                if is_inside and not contains_inner_points:
# #                    print("Found interior point inside ",temp_element)
# #                    set_elements.remove(temp_element)
# #                    break
#                 if is_inside and not contains_inner_points:
#                      possible_element1=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[1])
#                      possible_element2=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[2])
#                      possible_element3=(polygon.shape[0]+point_index,triangle_indices[1],triangle_indices[2])
#                      for element in set_elements:
#                          if set(element)==set(possible_element1) or set(element)==set(possible_element2) or set(element)==set(possible_element3):
#                              contains_element=True
#                              if temp_element in set_elements:
#                                  set_elements.remove(temp_element)
#                              break
# #            if bad_quality_after:
# #                continue
#             if contains_element:
#                 continue
#
#
#             # Checking if element contains points of contour ( other that the ones of the element)
#             contour_points_indices=set(np.array(range(0,len(polygon))))
#             element_indices=set(temp_element)
#             points_to_check_indices=np.asarray(list(contour_points_indices-element_indices))
#             check_points=polygon_with_points[points_to_check_indices]
#             is_inside=False
#
#
#             for point in check_points:
#                 is_inside=ray_tracing(point[0],point[1],triangle)
#                 if is_inside:
#                     # print("Found contour point inside ",temp_element)
#                     set_elements.remove(temp_element)
#                     break
#             if is_inside:
#                 continue
#             ################################################################################
#
#
#
#
#
#             # Check if a locked vertex was created after the creation of the element
#             # If so, add it to the list
#             #Tracer()()
#             Found_locked_vertex=False
#             for vertex in temp_element:
#                    if vertex<polygon.shape[0]:
#                        _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
#                        if isclosed and vertex not in set_locked_vertices:
#                            # print("Vertex locked:",vertex)
#                            Found_locked_vertex=True
#                            set_locked_vertices.add(vertex)
#             set_elements.remove(temp_element)
#
#
#
#             # Locking the vertices and checking if the connection is with a locked vertex has been checked/
#             # Proceeding to check if both internal edges intersect with other internal edges
#             if target_vtx<polygon.shape[0] :
#
#                 internal_edge1=(edge[0],target_vtx)
#                 internal_edge2=(edge[1],target_vtx)
#
#                 set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])
#
#                 internal_condition1= (internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections) and internal_edge1  not in set_interior_edge_with_inner_point
#
#                 internal_condition2=(internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections) and internal_edge2  not in set_interior_edge_with_inner_point
#
#
#
#                 internal_intersection=False
#                 if internal_condition1 or  internal_condition2:
#                     # print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
#                     # print("Abandoning creation of element",temp_element)
#                     internal_intersection=True
#
#
#                 if internal_intersection:
#                     for vtx in temp_element:
#                         if Found_locked_vertex and vtx in set_locked_vertices:
#                             # print("Unlocking vertex",vtx)
#                             set_locked_vertices.remove(vtx)
#                     continue
#
#
#
#             # Create the element
#             element=temp_element
#
#
#
#             # Add to set of edges all the forbidden intersections after the creation of the element
#             if target_vtx<polygon.shape[0]:
#
#                 for i in set_a:
#                     for j in set_b:
#                         set_forbidden_intersections.add((i,j))
#             # print("set of forbidden inter section edges updated:",set_forbidden_intersections)
#
#
#
#
#
#
#
#             # New edges after creation of the element
#
#             new_edge1=(edge[0],target_vtx)
#             new_edge2=(edge[1],target_vtx)
#
#            ### Add to check if the edges are intersecting with edge linked with interior point#####
#             found_line_intersection=False
#
#             if len(set_interior_edge_with_inner_point)!=0:
#                 for edge_connected_to_inner_point in set_interior_edge_with_inner_point:
#                     if found_line_intersection:
#                         break
#                     for new_edge in [new_edge1,new_edge2]:
#                         if new_edge==edge_connected_to_inner_point:
#                             continue
#                         new_edge_indices=np.asarray(list(new_edge))
#                         new_edge_coord=polygon_with_points[new_edge_indices]
#                         inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
#                         inner_edge_coord=polygon_with_points[inner_edge_indices]
#                         found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
#                         if found_line_intersection:
#                             break
#
#             if found_line_intersection:
#                 # print("found line intersection", new_edge, "with ",inner_edge_indices)
#                 for vtx in temp_element:
#                     if vtx  in set_locked_vertices:
#                         set_locked_vertices.remove(vtx)
#                 continue
#
#
#
#             ###Add to check if the edges are intersecting with edges###
#             found_line_intersection=False
#
#
#             for edge_connected_to_inner_point in interior_edges:
#                     if found_line_intersection:
#                         break
#                     for new_edge in [new_edge1,new_edge2]:
#                         if new_edge==edge_connected_to_inner_point:
#                             continue
#                         new_edge_indices=np.asarray(list(new_edge))
#                         new_edge_coord=polygon_with_points[new_edge_indices]
#                         inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
#                         inner_edge_coord=polygon_with_points[inner_edge_indices]
#                         found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
#                         if found_line_intersection:
#                             break
#
#             if found_line_intersection:
#
#                 # print("found line intersection", new_edge, "with ",inner_edge_indices)
#                 for vtx in temp_element:
#                     if vtx  in set_locked_vertices:
#                         set_locked_vertices.remove(vtx)
#                 continue
#
#             ###Add to check if the edges are intersecting with contour edges###
#
#             for contour_edge in get_contour_edges(polygon):
#                     if found_line_intersection:
#                         break
#                     for new_edge in [new_edge1,new_edge2]:
#                         if new_edge==tuple(contour_edge):
#                             continue
#                         new_edge_indices=np.asarray(list(new_edge))
#                         new_edge_coord=polygon_with_points[new_edge_indices]
#                         inner_edge_indices=np.asarray(list(contour_edge))
#                         inner_edge_coord=polygon_with_points[inner_edge_indices]
#                         found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
#                         if found_line_intersection:
#                             break
#
#             if found_line_intersection:
#                 # print("found line intersection", new_edge, "with ",inner_edge_indices)
#                 for vtx in temp_element:
#                     if vtx  in set_locked_vertices:
#                         set_locked_vertices.remove(vtx)
#                 continue
#
#
#
#
#
#             if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
#                 set_edges.add(new_edge1)
#                 interior_edges.add(new_edge1)
#                 # print("edges inserted:",new_edge1)
#                 # print("set of interior edges updated:",interior_edges)
#                 # print("set of edges updated:",set_edges)
#             if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:
#                 set_edges.add(new_edge2)
#                 interior_edges.add(new_edge2)
#                 # print("edges inserted:",new_edge2)
#                 # print("set of interior edges updated:",interior_edges)
#                 # print("set of edges updated:",set_edges)
#
#
#
#
#             # Checking list of elements to see whether the were created or were already there
#
#
#             set_elements.add(element)
#             if target_vtx>=polygon.shape[0]:
#                 set_interior_edge_with_inner_point.add(new_edge1)
#                 set_interior_edge_with_inner_point.add(new_edge2)
#
#             # print("element inserted:",element)
#             # print("Spotted edges linked with point: ",new_edge1," ",new_edge2)
#             element_created=True
#             #pdb.set_trace()
#             if element_created:
#                 break
#
#
#
#
#     if plot_mesh:
#         triangulated={'segment_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
#                       'vertex_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'vertices':np.vstack([ polygon,points])}
#         plot.plot(plt.axes(), **triangulated)
#     # print("Final edges:",set_edges)
#     # print("Elements created:",set_elements)
#     # print("Set of locked vertices:", set_locked_vertices)
#
#
#     # find open vertices
#     for element in set_elements:
#         for vertex in  element:
#                     if vertex>=polygon.shape[0]:
#                         continue
#
#                     _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
#                     if isclosed and vertex not in set_locked_vertices:
#                         # print("Vertex locked:",vertex)
#                         Found_locked_vertex=True
#                         set_locked_vertices.add(vertex)
#     set_open_vertices=set(range(len(polygon)))-set_locked_vertices
#     #unless the interior point is connected with all remaining vertices then it is condidered open
#     for i in range(len(polygon),len(polygon_with_points)):
#         set_open_vertices.add(i)
#
#
#
#     # Check for vertices that are not connecting to any point
#     set_interior_edge_with_inner_point_reformed=np.array(list(set_interior_edge_with_inner_point)).flatten()
#     for vertex in range(len(polygon),len(polygon_with_points)):
#         if vertex not in set_interior_edge_with_inner_point_reformed:
#             set_orphan_vertices.add(vertex)
#
#     # All interior points are treated as open vertices to forcesseking sub polygons uncluding the interior points
#     #for i in range(len(polygon),len(polygon_with_points)):
#      #   if i not in set_orphan_vertices:
#       #      set_open_vertices.add(i)
#     #set_open_vertices=set_open_vertices-set_orphan_vertex
#
#     # print("set of orphan vertex:",set_orphan_vertices)
#     # print("Set of open vertices:", set_open_vertices)
#     set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
#     sub_element_list=[]
#     if recursive:
#
#         sub_polygon_list=check_for_sub_polygon(set_orphan_vertices,set_open_vertices,interior_edges,set_elements,polygon,points)
#
#         ################## There could an orphan point inside an element making the element a subpolygon #############################
#
#         if len(set_orphan_vertices)!=0:
#             for element in set_elements:
#                 triangle_indices=np.asarray(element)
#                 triangle=polygon_with_points[triangle_indices]
#                 for vtx in set_orphan_vertices:
#                     point=polygon_with_points[vtx]
#                     is_inside=ray_tracing(point[0],point[1],triangle)
#                     if is_inside:
#                         sub_polygon_list.append(list(element))
#
#
#
#
#
#
#
#
#         if len(sub_polygon_list)==0:
#             return set_elements,[]
#
#         for sub_polygon_indices in sub_polygon_list:
#             if len(set_orphan_vertices)==0:
#                 if len(sub_polygon_indices)>=3:
#                     # print("remeshing subpolygon",sub_polygon_indices)
#                     polygon_copy=np.vstack([polygon,points])
#                     sub_polygon=np.array(polygon_copy[sub_polygon_indices])
#
#                     if not is_counterclockwise(sub_polygon):
#                         sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])
#
#                     sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
#                     sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)
#
#                     # print(sub_quality,sub_order_matrix)
#                     sub_elements,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
#                     if len(sub_elements)!=0:
#                         for element in sub_elements:
#                             indices=np.asarray(element)
#                             # print(element)
#                             triangle=sub_polygon[indices]
#                             polygon_indices=get_indices(triangle,polygon_with_points)
#                             sub_element_list.append(polygon_indices)
#             else:
#                 if len(sub_polygon_indices)>=3:
#
#
#                     sub_polygon_inner_points=[]
#                     inner_points_indices=np.asarray(list(set_orphan_vertices)).flatten()
#                     #inner_points_indices=np.sort(inner_points_indices)
#                     # print("remeshing subpolygon",sub_polygon_indices)
#                     polygon_copy=np.vstack([polygon,points])
#                     sub_polygon=np.array(polygon_copy[sub_polygon_indices])
#
#
#                     if not is_counterclockwise(sub_polygon):
#                             sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])
#
#                     inner_points=np.array(polygon_copy[inner_points_indices])
#                     inner_points=sort_points(inner_points.reshape(1,len(inner_points),2),len(inner_points)).reshape(len(inner_points),2)
#
#                     for point in inner_points:
#                         is_inside=ray_tracing(point[0],point[1],sub_polygon)
#                         if is_inside:
#                             sub_polygon_inner_points.append(point)
#                             # print("Point ",point," is inside ", sub_polygon_indices)
#
#
#                     if len(sub_polygon_inner_points)!=0:
#
#                         sub_polygon_inner_points=np.array(sub_polygon_inner_points)
#
#                        # if not is_counterclockwise(sub_polygon_inner _points):
#                         #    sub_polygon_inner_points=sub_polygon_inner_points[::-1]
#
#                         sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
#                         sub_quality,_=quality_matrix(sub_polygon,sub_polygon_inner_points,compute_minimum=True,normalize=False)
#                         sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=True)
#
#                         # print(sub_quality,sub_order_matrix)
#                         # print(sub_polygon)
#                         sub_elements,_=triangulate(sub_polygon,sub_polygon_inner_points,sub_order_matrix,recursive=True)
#                         if len(sub_elements)!=0:
#                             for element in sub_elements:
#                                 indices=np.asarray(element)
#                                 # print(element)
#                                 triangle=sub_polygon_with_points[indices]
#                                 polygon_indices=get_indices(triangle,polygon_with_points)
#                                 sub_element_list.append(polygon_indices)
#                         #print("sub_polygon: " , sub_polygon)
#                         #print("sub_polygon_poiny: ",sub_polygon_inner_points)
#
#                     else:
#                         # polygon_copy=np.vstack([polygon,points])
#                          #sub_polygon=np.array(polygon_copy[sub_polygon_indices])
#                          #if not is_counterclockwise(sub_polygon):
#                           #   sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])
#
#     #                    sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
#                          sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
#                          sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)
#
#                          # print(sub_quality,sub_order_matrix)
#                          sub_elements,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
#                          if len(sub_elements)!=0:
#                              for element in sub_elements:
#                                  indices=np.asarray(element)
#                                  # print(element)
#                                  triangle=sub_polygon[indices]
#                                  polygon_indices=get_indices(triangle,polygon_with_points)
#                                  sub_element_list.append(polygon_indices)
#
#     return set_elements,sub_element_list

def triangulate(polygon,points,ordered_quality_matrix,recursive=True,plot_mesh=False):

    set_edges=set(tuple(i) for i in get_contour_edges(polygon))
    interior_edges=set()
    set_elements=set()
    set_locked_vertices=set()
    set_forbidden_intersections=set()
    set_interior_edge_with_inner_point=set()
    set_orphan_vertices=set()
    # print("initial set edges:", set_edges)



    polygon_with_points=np.vstack([polygon,points])


    # print("meshing polygon: " , polygon," with inner points :", points)


    for edge in ordered_quality_matrix.keys():

        found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)

        for qualities_with_edges in ordered_quality_matrix[edge][0]:

            element_created=False

            target_vtx=qualities_with_edges[1]

            if target_vtx==edge[0] or target_vtx==edge[1]:
                continue

            # print("Edge:",edge,"targeting:",target_vtx)

            if found_in_interior_set: #and target_vtx!=polygon.shape[0]:
                element=(edge[0],edge[1],index)
                set_elements.add(element)
                # print("Element inserted:",element)
                continue

            ############# Could be a Triangle with inner points inside we want to mesh this as well ################
            if polygon.shape[0]>3:
                if found_in_set and not found_in_interior_set: #and target_vtx!=polygon.shape[0]:
#                    if(index != target_vtx) and index<=polygon.shape[0] :
#                        print('found',(edge[0],index),(edge[1],index),"Canceling creation")
#                        continue
                    if(index != target_vtx) and index<=polygon.shape[0] :
                        # print('found',(edge[0],index),(edge[1],index),"Canceling creation")
                        continue





            # Passed edges checking
            # Proceed to check vertices
            temp_element=(edge[0],edge[1],target_vtx)
            # print(temp_element)
            existing_element=False
            for element in set_elements:
                if set(temp_element)== set(element):
                    # print("Element {} already in set".format(element))
                    existing_element=True
                    break
            if existing_element:
                break



            if target_vtx in set_locked_vertices:
                # print(" Target vertex {} is locked".format(target_vtx))
                continue
            set_elements.add(temp_element)

            triangle_indices=np.asarray(temp_element)
            triangle=polygon_with_points[triangle_indices]
            # print(triangle)



            ################# posteriori checks for NN ####################################
            # Checking for invalid connections outside of the contour

            if  compute_triangle_area(triangle)==0:
                # print("found zero area triangle", temp_element)
                set_elements.remove(temp_element)
                continue

            contains_inner_points=False
            for index in temp_element:
                if index>=polygon.shape[0]:
                    contains_inner_points=True


#           # Checking if element includes interior point, if it does then check if there is an element in set element formed by
             # the inner point and the edges of temp element
            is_inside=False
            contains_element=False
#            bad_quality_after=False
            for point_index,point in enumerate(points):
                is_inside=ray_tracing(point[0],point[1],triangle)

#                if is_inside:
#                    barycenter=np.array([triangle[:,0].sum()/3,triangle[:,1].sum()/3])
#                    if np.linalg.norm(barycenter-point)>0.05:
#                        bad_quality_after=True
#                        break
#                if is_inside and not contains_inner_points:
#                    print("Found interior point inside ",temp_element)
#                    set_elements.remove(temp_element)
#                    break
                if is_inside and not contains_inner_points:
                     possible_element1=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[1])
                     possible_element2=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[2])
                     possible_element3=(polygon.shape[0]+point_index,triangle_indices[1],triangle_indices[2])
                     for element in set_elements:
                         if set(element)==set(possible_element1) or set(element)==set(possible_element2) or set(element)==set(possible_element3):
                             contains_element=True
                             if temp_element in set_elements:
                                 set_elements.remove(temp_element)
                             break
#            if bad_quality_after:
#                continue
            if contains_element:
                continue


            # Checking if element contains points of contour ( other that the ones of the element)
            contour_points_indices=set(np.array(range(0,len(polygon))))
            element_indices=set(temp_element)
            points_to_check_indices=np.asarray(list(contour_points_indices-element_indices))
            check_points=polygon_with_points[points_to_check_indices]
            is_inside=False


            for point in check_points:
                is_inside=ray_tracing(point[0],point[1],triangle)
                if is_inside:
                    # print("Found contour point inside ",temp_element)
                    set_elements.remove(temp_element)
                    break
            if is_inside:
                continue
            ################################################################################





            # Check if a locked vertex was created after the creation of the element
            # If so, add it to the list
            #Tracer()()
            Found_locked_vertex=False
            for vertex in temp_element:
                   if vertex<polygon.shape[0]:
                       _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                       if isclosed and vertex not in set_locked_vertices:
                           # print("Vertex locked:",vertex)
                           Found_locked_vertex=True
                           set_locked_vertices.add(vertex)
            set_elements.remove(temp_element)



            # Locking the vertices and checking if the connection is with a locked vertex has been checked/
            # Proceeding to check if both internal edges intersect with other internal edges
            if target_vtx<polygon.shape[0] :

                internal_edge1=(edge[0],target_vtx)
                internal_edge2=(edge[1],target_vtx)

                set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])

                internal_condition1= (internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections) and internal_edge1  not in set_interior_edge_with_inner_point

                internal_condition2=(internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections) and internal_edge2  not in set_interior_edge_with_inner_point



                internal_intersection=False
                if internal_condition1 or  internal_condition2:
                    # print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
                    # print("Abandoning creation of element",temp_element)
                    internal_intersection=True


                if internal_intersection:
                    for vtx in temp_element:
                        if Found_locked_vertex and vtx in set_locked_vertices:
                            # print("Unlocking vertex",vtx)
                            set_locked_vertices.remove(vtx)
                    continue



            # Create the element
            element=temp_element



            # Add to set of edges all the forbidden intersections after the creation of the element
            if target_vtx<polygon.shape[0]:

                for i in set_a:
                    for j in set_b:
                        set_forbidden_intersections.add((i,j))
            # print("set of forbidden inter section edges updated:",set_forbidden_intersections)







            # New edges after creation of the element

            new_edge1=(edge[0],target_vtx)
            new_edge2=(edge[1],target_vtx)

           ### Add to check if the edges are intersecting with edge linked with interior point#####
            found_line_intersection=False

            if len(set_interior_edge_with_inner_point)!=0:
                for edge_connected_to_inner_point in set_interior_edge_with_inner_point:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue



            ###Add to check if the edges are intersecting with edges###
            found_line_intersection=False


            for edge_connected_to_inner_point in interior_edges:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:

                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue

            ###Add to check if the edges are intersecting with contour edges###

            for contour_edge in get_contour_edges(polygon):
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==tuple(contour_edge):
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(contour_edge))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue





            if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
                set_edges.add(new_edge1)
                interior_edges.add(new_edge1)
                # print("edges inserted:",new_edge1)
                # print("set of interior edges updated:",interior_edges)
                # print("set of edges updated:",set_edges)
            if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:
                set_edges.add(new_edge2)
                interior_edges.add(new_edge2)
                # print("edges inserted:",new_edge2)
                # print("set of interior edges updated:",interior_edges)
                # print("set of edges updated:",set_edges)




            # Checking list of elements to see whether the were created or were already there


            set_elements.add(element)
            if target_vtx>=polygon.shape[0]:
                set_interior_edge_with_inner_point.add(new_edge1)
                set_interior_edge_with_inner_point.add(new_edge2)

            # print("element inserted:",element)
            # print("Spotted edges linked with point: ",new_edge1," ",new_edge2)
            element_created=True
            #pdb.set_trace()
            if element_created:
                break




    if plot_mesh:
        triangulated={'segment_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
                      'vertex_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'vertices':np.vstack([ polygon,points])}
        plot(plt.axes(), **triangulated)
    # print("Final edges:",set_edges)
    # print("Elements created:",set_elements)
    # print("Set of locked vertices:", set_locked_vertices)


    # find open vertices
    for element in set_elements:
        for vertex in  element:
                    if vertex>=polygon.shape[0]:
                        continue

                    _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                    if isclosed and vertex not in set_locked_vertices:
                        # print("Vertex locked:",vertex)
                        Found_locked_vertex=True
                        set_locked_vertices.add(vertex)
    set_open_vertices=set(range(len(polygon)))-set_locked_vertices
    #unless the interior point is connected with all remaining vertices then it is condidered open
    for i in range(len(polygon),len(polygon_with_points)):
        set_open_vertices.add(i)



    # Check for vertices that are not connecting to any point
    set_interior_edge_with_inner_point_reformed=np.array(list(set_interior_edge_with_inner_point)).flatten()
    for vertex in range(len(polygon),len(polygon_with_points)):
        if vertex not in set_interior_edge_with_inner_point_reformed:
            set_orphan_vertices.add(vertex)

    # All interior points are treated as open vertices to forcesseking sub polygons uncluding the interior points
    #for i in range(len(polygon),len(polygon_with_points)):
     #   if i not in set_orphan_vertices:
      #      set_open_vertices.add(i)
    #set_open_vertices=set_open_vertices-set_orphan_vertex

    # print("set of orphan vertex:",set_orphan_vertices)
    # print("Set of open vertices:", set_open_vertices)
    set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
    sub_element_list=[]
    if recursive:

        sub_polygon_list=check_for_sub_polygon(set_orphan_vertices,set_open_vertices,interior_edges,set_elements,polygon,points)

        ################## There could an orphan point inside an element making the element a subpolygon #############################

        if len(set_orphan_vertices)!=0:
            for element in set_elements:
                triangle_indices=np.asarray(element)
                triangle=polygon_with_points[triangle_indices]
                for vtx in set_orphan_vertices:
                    point=polygon_with_points[vtx]
                    is_inside=ray_tracing(point[0],point[1],triangle)
                    if is_inside:
                        sub_polygon_list.append(list(element))








        if len(sub_polygon_list)==0:
            return set_elements,[]

        for sub_polygon_indices in sub_polygon_list:
            if len(set_orphan_vertices)==0:
                if len(sub_polygon_indices)>=3:
                    # print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])

                    if not is_counterclockwise(sub_polygon):
                        sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                    sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                    sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)

                    # print(sub_quality,sub_order_matrix)
                    sub_elements,_,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
                    if len(sub_elements)!=0:
                        for element in sub_elements:
                            indices=np.asarray(element)
                            # print(element)
                            triangle=sub_polygon[indices]
                            polygon_indices=get_indices(triangle,polygon_with_points)
                            sub_element_list.append(polygon_indices)
            else:
                if len(sub_polygon_indices)>=3:


                    sub_polygon_inner_points=[]
                    inner_points_indices=np.asarray(list(set_orphan_vertices)).flatten()
                    #inner_points_indices=np.sort(inner_points_indices)
                    # print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])


                    if not is_counterclockwise(sub_polygon):
                            sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                    inner_points=np.array(polygon_copy[inner_points_indices])
                    inner_points=sort_points(inner_points.reshape(1,len(inner_points),2),len(inner_points)).reshape(len(inner_points),2)

                    for point in inner_points:
                        is_inside=ray_tracing(point[0],point[1],sub_polygon)
                        if is_inside:
                            sub_polygon_inner_points.append(point)
                            # print("Point ",point," is inside ", sub_polygon_indices)


                    if len(sub_polygon_inner_points)!=0:

                        sub_polygon_inner_points=np.array(sub_polygon_inner_points)

                       # if not is_counterclockwise(sub_polygon_inner _points):
                        #    sub_polygon_inner_points=sub_polygon_inner_points[::-1]

                        sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
                        sub_quality,_=quality_matrix(sub_polygon,sub_polygon_inner_points,compute_minimum=True,normalize=False)
                        sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=True)

                        # print(sub_quality,sub_order_matrix)
                        # print(sub_polygon)
                        sub_elements,_=triangulate(sub_polygon,sub_polygon_inner_points,sub_order_matrix,recursive=True)
                        if len(sub_elements)!=0:
                            for element in sub_elements:
                                indices=np.asarray(element)
                                # print(element)
                                triangle=sub_polygon_with_points[indices]
                                polygon_indices=get_indices(triangle,polygon_with_points)
                                sub_element_list.append(polygon_indices)
                        #print("sub_polygon: " , sub_polygon)
                        #print("sub_polygon_poiny: ",sub_polygon_inner_points)

                    else:
                        # polygon_copy=np.vstack([polygon,points])
                         #sub_polygon=np.array(polygon_copy[sub_polygon_indices])
                         #if not is_counterclockwise(sub_polygon):
                          #   sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

    #                    sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
                         sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                         sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)

                         # print(sub_quality,sub_order_matrix)
                         sub_elements,_,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
                         if len(sub_elements)!=0:
                             for element in sub_elements:
                                 indices=np.asarray(element)
                                 # print(element)
                                 triangle=sub_polygon[indices]
                                 polygon_indices=get_indices(triangle,polygon_with_points)
                                 sub_element_list.append(polygon_indices)

    return set_elements,sub_element_list

def triangulate_NN(polygon,points,ordered_quality_matrix,recursive=True,plot_mesh=True):
    set_edges=set(tuple(i) for i in get_contour_edges(polygon))
    interior_edges=set()
    set_elements=set()
    set_locked_vertices=set()
    set_forbidden_intersections=set()
    set_interior_edge_with_inner_point=set()
    set_orphan_vertices=set()
    # print("initial set edges:", set_edges)



    polygon_with_points=np.vstack([polygon,points])


    # print("meshing polygon: " , polygon," with inner points :", points)


    for edge in ordered_quality_matrix.keys():

        found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)

        for qualities_with_edges in ordered_quality_matrix[edge][0]:

            element_created=False

            target_vtx=qualities_with_edges[1]

            if target_vtx==edge[0] or target_vtx==edge[1]:
                continue

            # print("Edge:",edge,"targeting:",target_vtx)

            if found_in_interior_set: #and target_vtx!=polygon.shape[0]:
                element=(edge[0],edge[1],index)
                set_elements.add(element)
                # print("Element inserted:",element)
                continue

            ############# Could be a Triangle with inner points inside we want to mesh this as well ################
            if polygon.shape[0]>3:
                if found_in_set and not found_in_interior_set: #and target_vtx!=polygon.shape[0]:
#                    if(index != target_vtx) and index<=polygon.shape[0] :
#                        print('found',(edge[0],index),(edge[1],index),"Canceling creation")
#                        continue
                    if(index != target_vtx) and index<=polygon.shape[0] :
                        # print('found',(edge[0],index),(edge[1],index),"Canceling creation")
                        continue





            # Passed edges checking
            # Proceed to check vertices
            temp_element=(edge[0],edge[1],target_vtx)
            # print(temp_element)
            existing_element=False
            for element in set_elements:
                if set(temp_element)== set(element):
                    # print("Element {} already in set".format(element))
                    existing_element=True
                    break
            if existing_element:
                break



            if target_vtx in set_locked_vertices:
                # print(" Target vertex {} is locked".format(target_vtx))
                continue
            set_elements.add(temp_element)

            triangle_indices=np.asarray(temp_element)
            triangle=polygon_with_points[triangle_indices]
            # print(triangle)



            ################# posteriori checks for NN ####################################
            # Checking for invalid connections outside of the contour

            if  compute_triangle_area(triangle)==0:
                # print("found zero area triangle", temp_element)
                set_elements.remove(temp_element)
                continue

            contains_inner_points=False
            for index in temp_element:
                if index>=polygon.shape[0]:
                    contains_inner_points=True


#           # Checking if element includes interior point, if it does then check if there is an element in set element formed by
             # the inner point and the edges of temp element
            is_inside=False
            contains_element=False
#            bad_quality_after=False
            for point_index,point in enumerate(points):
                is_inside=ray_tracing(point[0],point[1],triangle)

#                if is_inside:
#                    barycenter=np.array([triangle[:,0].sum()/3,triangle[:,1].sum()/3])
#                    if np.linalg.norm(barycenter-point)>0.05:
#                        bad_quality_after=True
#                        break
#                if is_inside and not contains_inner_points:
#                    print("Found interior point inside ",temp_element)
#                    set_elements.remove(temp_element)
#                    break
                if is_inside and not contains_inner_points:
                     possible_element1=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[1])
                     possible_element2=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[2])
                     possible_element3=(polygon.shape[0]+point_index,triangle_indices[1],triangle_indices[2])
                     for element in set_elements:
                         if set(element)==set(possible_element1) or set(element)==set(possible_element2) or set(element)==set(possible_element3):
                             contains_element=True
                             if temp_element in set_elements:
                                 set_elements.remove(temp_element)
                             break
#            if bad_quality_after:
#                continue
            if contains_element:
                continue


            # Checking if element contains points of contour ( other that the ones of the element)
            contour_points_indices=set(np.array(range(0,len(polygon))))
            element_indices=set(temp_element)
            points_to_check_indices=np.asarray(list(contour_points_indices-element_indices))
            check_points=polygon_with_points[points_to_check_indices]
            is_inside=False


            for point in check_points:
                is_inside=ray_tracing(point[0],point[1],triangle)
                if is_inside:
                    # print("Found contour point inside ",temp_element)
                    set_elements.remove(temp_element)
                    break
            if is_inside:
                continue
            ################################################################################





            # Check if a locked vertex was created after the creation of the element
            # If so, add it to the list
            #Tracer()()
            Found_locked_vertex=False
            for vertex in temp_element:
                   if vertex<polygon.shape[0]:
                       _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                       if isclosed and vertex not in set_locked_vertices:
                           # print("Vertex locked:",vertex)
                           Found_locked_vertex=True
                           set_locked_vertices.add(vertex)
            set_elements.remove(temp_element)



            # Locking the vertices and checking if the connection is with a locked vertex has been checked/
            # Proceeding to check if both internal edges intersect with other internal edges
            if target_vtx<polygon.shape[0] :

                internal_edge1=(edge[0],target_vtx)
                internal_edge2=(edge[1],target_vtx)

                set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])

                internal_condition1= (internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections) and internal_edge1  not in set_interior_edge_with_inner_point

                internal_condition2=(internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections) and internal_edge2  not in set_interior_edge_with_inner_point



                internal_intersection=False
                if internal_condition1 or  internal_condition2:
                    # print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
                    # print("Abandoning creation of element",temp_element)
                    internal_intersection=True


                if internal_intersection:
                    for vtx in temp_element:
                        if Found_locked_vertex and vtx in set_locked_vertices:
                            # print("Unlocking vertex",vtx)
                            set_locked_vertices.remove(vtx)
                    continue



            # Create the element
            element=temp_element



            # Add to set of edges all the forbidden intersections after the creation of the element
            if target_vtx<polygon.shape[0]:

                for i in set_a:
                    for j in set_b:
                        set_forbidden_intersections.add((i,j))
            # print("set of forbidden inter section edges updated:",set_forbidden_intersections)







            # New edges after creation of the element

            new_edge1=(edge[0],target_vtx)
            new_edge2=(edge[1],target_vtx)

           ### Add to check if the edges are intersecting with edge linked with interior point#####
            found_line_intersection=False

            if len(set_interior_edge_with_inner_point)!=0:
                for edge_connected_to_inner_point in set_interior_edge_with_inner_point:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue



            ###Add to check if the edges are intersecting with edges###
            found_line_intersection=False


            for edge_connected_to_inner_point in interior_edges:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:

                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue

            ###Add to check if the edges are intersecting with contour edges###

            for contour_edge in get_contour_edges(polygon):
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==tuple(contour_edge):
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(contour_edge))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
                # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue





            if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
                set_edges.add(new_edge1)
                interior_edges.add(new_edge1)
                # print("edges inserted:",new_edge1)
                # print("set of interior edges updated:",interior_edges)
                # print("set of edges updated:",set_edges)
            if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:
                set_edges.add(new_edge2)
                interior_edges.add(new_edge2)
                # print("edges inserted:",new_edge2)
                # print("set of interior edges updated:",interior_edges)
                # print("set of edges updated:",set_edges)




            # Checking list of elements to see whether the were created or were already there


            set_elements.add(element)
            if target_vtx>=polygon.shape[0]:
                set_interior_edge_with_inner_point.add(new_edge1)
                set_interior_edge_with_inner_point.add(new_edge2)

            # print("element inserted:",element)
            # print("Spotted edges linked with point: ",new_edge1," ",new_edge2)
            element_created=True
            #pdb.set_trace()
            if element_created:
                break




    if plot_mesh:
        triangulated={'segment_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
                      'vertex_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'vertices':np.vstack([ polygon,points])}
        plot.plot(plt.axes(), **triangulated)
    # print("Final edges:",set_edges)
    # print("Elements created:",set_elements)
    # print("Set of locked vertices:", set_locked_vertices)


    # find open vertices
    for element in set_elements:
        for vertex in  element:
                    if vertex>=polygon.shape[0]:
                        continue

                    _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                    if isclosed and vertex not in set_locked_vertices:
                        # print("Vertex locked:",vertex)
                        Found_locked_vertex=True
                        set_locked_vertices.add(vertex)
    set_open_vertices=set(range(len(polygon)))-set_locked_vertices
    #unless the interior point is connected with all remaining vertices then it is condidered open
    for i in range(len(polygon),len(polygon_with_points)):
        set_open_vertices.add(i)



    # Check for vertices that are not connecting to any point
    set_interior_edge_with_inner_point_reformed=np.array(list(set_interior_edge_with_inner_point)).flatten()
    for vertex in range(len(polygon),len(polygon_with_points)):
        if vertex not in set_interior_edge_with_inner_point_reformed:
            set_orphan_vertices.add(vertex)

    # All interior points are treated as open vertices to forcesseking sub polygons uncluding the interior points
    #for i in range(len(polygon),len(polygon_with_points)):
     #   if i not in set_orphan_vertices:
      #      set_open_vertices.add(i)
    #set_open_vertices=set_open_vertices-set_orphan_vertex

    # print("set of orphan vertex:",set_orphan_vertices)
    # print("Set of open vertices:", set_open_vertices)
    set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
    sub_element_list=[]
    if recursive:

        sub_polygon_list=check_for_sub_polygon(set_orphan_vertices,set_open_vertices,interior_edges,set_elements,polygon,points)

        ################## There could an orphan point inside an element making the element a subpolygon #############################

        if len(set_orphan_vertices)!=0:
            for element in set_elements:
                triangle_indices=np.asarray(element)
                triangle=polygon_with_points[triangle_indices]
                for vtx in set_orphan_vertices:
                    point=polygon_with_points[vtx]
                    is_inside=ray_tracing(point[0],point[1],triangle)
                    if is_inside:
                        sub_polygon_list.append(list(element))








        if len(sub_polygon_list)==0:
            return set_elements,[]

        for sub_polygon_indices in sub_polygon_list:
            if len(set_orphan_vertices)==0:
                if len(sub_polygon_indices)>=3:
                    # print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])

                    if not is_counterclockwise(sub_polygon):
                        sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                    sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                    sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)

                    # print(sub_quality,sub_order_matrix)
                    sub_elements,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
                    if len(sub_elements)!=0:
                        for element in sub_elements:
                            indices=np.asarray(element)
                            # print(element)
                            triangle=sub_polygon[indices]
                            polygon_indices=get_indices(triangle,polygon_with_points)
                            sub_element_list.append(polygon_indices)
            else:
                if len(sub_polygon_indices)>=3:


                    sub_polygon_inner_points=[]
                    inner_points_indices=np.asarray(list(set_orphan_vertices)).flatten()
                    #inner_points_indices=np.sort(inner_points_indices)
                    # print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])


                    if not is_counterclockwise(sub_polygon):
                            sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                    inner_points=np.array(polygon_copy[inner_points_indices])
                    inner_points=sort_points(inner_points.reshape(1,len(inner_points),2),len(inner_points)).reshape(len(inner_points),2)

                    for point in inner_points:
                        is_inside=ray_tracing(point[0],point[1],sub_polygon)
                        if is_inside:
                            sub_polygon_inner_points.append(point)
                            # print("Point ",point," is inside ", sub_polygon_indices)


                    if len(sub_polygon_inner_points)!=0:

                        sub_polygon_inner_points=np.array(sub_polygon_inner_points)

                       # if not is_counterclockwise(sub_polygon_inner _points):
                        #    sub_polygon_inner_points=sub_polygon_inner_points[::-1]
#
                        sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
#                        sub_quality,_=quality_matrix(sub_polygon,sub_polygon_inner_points,compute_minimum=True,normalize=False)
#                        sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=True)

                        nb_of_edges=len(sub_polygon)
                        nb_of_points=len(sub_polygon_inner_points)
                        with open('../network_datasets/connectivity_NN/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_NN_qualities_with_extra_grid_points.pkl','rb') as f:
                            connection_network=pickle.load(f)

                        contour_with_point_variable=Variable(torch.from_numpy(sub_polygon_with_points.reshape(-1)).type(torch.FloatTensor)).expand(1,len(sub_polygon_with_points.reshape(-1)))
                        connection_network=connection_network.cpu().eval()
                        prediction=connection_network(contour_with_point_variable).data[0].numpy()
                        sub_quality=prediction.reshape(nb_of_edges,nb_of_edges+nb_of_points)
                        sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=False)

                        # print(sub_quality,sub_order_matrix)
                        # print(sub_polygon)
                        sub_elements,_=triangulate(sub_polygon,sub_polygon_inner_points,sub_order_matrix,recursive=True)
                        if len(sub_elements)!=0:
                            for element in sub_elements:
                                indices=np.asarray(element)
                                # print(element)
                                triangle=sub_polygon_with_points[indices]
                                polygon_indices=get_indices(triangle,polygon_with_points)
                                sub_element_list.append(polygon_indices)
                        #print("sub_polygon: " , sub_polygon)
                        #print("sub_polygon_poiny: ",sub_polygon_inner_points)

                    else:
                        # polygon_copy=np.vstack([polygon,points])
                         #sub_polygon=np.array(polygon_copy[sub_polygon_indices])
                         #if not is_counterclockwise(sub_polygon):
                          #   sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

    #                    sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
                         sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                         sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)

                         # print(sub_quality,sub_order_matrix)
                         sub_elements,_=Triangulation.triangulate(sub_polygon,sub_order_matrix,recursive=True)
                         if len(sub_elements)!=0:
                             for element in sub_elements:
                                 indices=np.asarray(element)
                                 # print(element)
                                 triangle=sub_polygon[indices]
                                 polygon_indices=get_indices(triangle,polygon_with_points)
                                 sub_element_list.append(polygon_indices)

    return set_elements,sub_element_list


def triangulate_NN_pure(polygon,points,ordered_quality_matrix,recursive=True,plot_mesh=True):
    set_edges=set(tuple(i) for i in get_contour_edges(polygon))
    interior_edges=set()
    set_elements=set()
    set_locked_vertices=set()
    set_forbidden_intersections=set()
    set_interior_edge_with_inner_point=set()
    set_orphan_vertices=set()
   #print("initial set edges:", set_edges)



    polygon_with_points=np.vstack([polygon,points])


    #print("meshing polygon: " , polygon," with inner points :", points)


    for edge in ordered_quality_matrix.keys():

        found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)

        for qualities_with_edges in ordered_quality_matrix[edge][0]:

            element_created=False

            target_vtx=qualities_with_edges[1]

            if target_vtx==edge[0] or target_vtx==edge[1]:
                continue

     #       print("Edge:",edge,"targeting:",target_vtx)

            if found_in_interior_set: #and target_vtx!=polygon.shape[0]:
                element=(edge[0],edge[1],index)
                set_elements.add(element)
      #          print("Element inserted:",element)
                continue

            ############# Could be a Triangle with inner points inside we want to mesh this as well ################
            if polygon.shape[0]>3:
                if found_in_set and not found_in_interior_set: #and target_vtx!=polygon.shape[0]:
#                    if(index != target_vtx) and index<=polygon.shape[0] :
#                        print('found',(edge[0],index),(edge[1],index),"Canceling creation")
#                        continue
                    if(index != target_vtx) and index<=polygon.shape[0] :
       #                 print('found',(edge[0],index),(edge[1],index),"Canceling creation")
                        continue





            # Passed edges checking
            # Proceed to check vertices
            temp_element=(edge[0],edge[1],target_vtx)
        #    print(temp_element)
            existing_element=False
            for element in set_elements:
                if set(temp_element)== set(element):
         #           print("Element {} already in set".format(element))
                    existing_element=True
                    break
            if existing_element:
                break



            if target_vtx in set_locked_vertices:
          #      print(" Target vertex {} is locked".format(target_vtx))
                continue
            set_elements.add(temp_element)

            triangle_indices=np.asarray(temp_element)
            triangle=polygon_with_points[triangle_indices]
           # print(triangle)



            ################# posteriori checks for NN ####################################
            # Checking for invalid connections outside of the contour

            if  compute_triangle_area(triangle)==0:
            #    print("found zero area triangle", temp_element)
                set_elements.remove(temp_element)
                continue

            contains_inner_points=False
            for index in temp_element:
                if index>=polygon.shape[0]:
                    contains_inner_points=True


#           # Checking if element includes interior point, if it does then check if there is an element in set element formed by
             # the inner point and the edges of temp element
            is_inside=False
            contains_element=False
#            bad_quality_after=False
            for point_index,point in enumerate(points):
                is_inside=ray_tracing(point[0],point[1],triangle)

#                if is_inside:
#                    barycenter=np.array([triangle[:,0].sum()/3,triangle[:,1].sum()/3])
#                    if np.linalg.norm(barycenter-point)>0.05:
#                        bad_quality_after=True
#                        break
#                if is_inside and not contains_inner_points:
#                    print("Found interior point inside ",temp_element)
#                    set_elements.remove(temp_element)
#                    break
                if is_inside and not contains_inner_points:
                     possible_element1=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[1])
                     possible_element2=(polygon.shape[0]+point_index,triangle_indices[0],triangle_indices[2])
                     possible_element3=(polygon.shape[0]+point_index,triangle_indices[1],triangle_indices[2])
                     for element in set_elements:
                         if set(element)==set(possible_element1) or set(element)==set(possible_element2) or set(element)==set(possible_element3):
                             contains_element=True
                             if temp_element in set_elements:
                                 set_elements.remove(temp_element)
                             break
#            if bad_quality_after:
#                continue
            if contains_element:
                continue


            # Checking if element contains points of contour ( other that the ones of the element)
            contour_points_indices=set(np.array(range(0,len(polygon))))
            element_indices=set(temp_element)
            points_to_check_indices=np.asarray(list(contour_points_indices-element_indices))
            check_points=polygon_with_points[points_to_check_indices]
            is_inside=False


            for point in check_points:
                is_inside=ray_tracing(point[0],point[1],triangle)
                if is_inside:
            #        print("Found contour point inside ",temp_element)
                    set_elements.remove(temp_element)
                    break
            if is_inside:
                continue
            ################################################################################





            # Check if a locked vertex was created after the creation of the element
            # If so, add it to the list
            #Tracer()()
            Found_locked_vertex=False
            for vertex in temp_element:
                   if vertex<polygon.shape[0]:
                       _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                       if isclosed and vertex not in set_locked_vertices:
             #              print("Vertex locked:",vertex)
                           Found_locked_vertex=True
                           set_locked_vertices.add(vertex)
            set_elements.remove(temp_element)



            # Locking the vertices and checking if the connection is with a locked vertex has been checked/
            # Proceeding to check if both internal edges intersect with other internal edges
            if target_vtx<polygon.shape[0] :

                internal_edge1=(edge[0],target_vtx)
                internal_edge2=(edge[1],target_vtx)

                set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])

                internal_condition1= (internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections) and internal_edge1  not in set_interior_edge_with_inner_point

                internal_condition2=(internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections) and internal_edge2  not in set_interior_edge_with_inner_point



                internal_intersection=False
                if internal_condition1 or  internal_condition2:
              #      print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
               #     print("Abandoning creation of element",temp_element)
                    internal_intersection=True


                if internal_intersection:
                    for vtx in temp_element:
                        if Found_locked_vertex and vtx in set_locked_vertices:
                #            print("Unlocking vertex",vtx)
                            set_locked_vertices.remove(vtx)
                    continue



            # Create the element
            element=temp_element



            # Add to set of edges all the forbidden intersections after the creation of the element
            if target_vtx<polygon.shape[0]:

                for i in set_a:
                    for j in set_b:
                        set_forbidden_intersections.add((i,j))
            #print("set of forbidden inter section edges updated:",set_forbidden_intersections)







            # New edges after creation of the element

            new_edge1=(edge[0],target_vtx)
            new_edge2=(edge[1],target_vtx)

           ### Add to check if the edges are intersecting with edge linked with interior point#####
            found_line_intersection=False

            if len(set_interior_edge_with_inner_point)!=0:
                for edge_connected_to_inner_point in set_interior_edge_with_inner_point:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
             #   print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue



            ###Add to check if the edges are intersecting with edges###
            found_line_intersection=False


            for edge_connected_to_inner_point in interior_edges:
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==edge_connected_to_inner_point:
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(edge_connected_to_inner_point))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:

              #  print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue

            ###Add to check if the edges are intersecting with contour edges###

            for contour_edge in get_contour_edges(polygon):
                    if found_line_intersection:
                        break
                    for new_edge in [new_edge1,new_edge2]:
                        if new_edge==tuple(contour_edge):
                            continue
                        new_edge_indices=np.asarray(list(new_edge))
                        new_edge_coord=polygon_with_points[new_edge_indices]
                        inner_edge_indices=np.asarray(list(contour_edge))
                        inner_edge_coord=polygon_with_points[inner_edge_indices]
                        found_line_intersection=do_intersect(new_edge_coord[0],new_edge_coord[1],inner_edge_coord[0],inner_edge_coord[1])
                        if found_line_intersection:
                            break

            if found_line_intersection:
               # print("found line intersection", new_edge, "with ",inner_edge_indices)
                for vtx in temp_element:
                    if vtx  in set_locked_vertices:
                        set_locked_vertices.remove(vtx)
                continue





            if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
                set_edges.add(new_edge1)
                interior_edges.add(new_edge1)
                #print("edges inserted:",new_edge1)
                #print("set of interior edges updated:",interior_edges)
                #print("set of edges updated:",set_edges)
            if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:
                set_edges.add(new_edge2)
                interior_edges.add(new_edge2)
                #print("edges inserted:",new_edge2)
                #print("set of interior edges updated:",interior_edges)
                #print("set of edges updated:",set_edges)




            # Checking list of elements to see whether the were created or were already there


            set_elements.add(element)
            if target_vtx>=polygon.shape[0]:
                set_interior_edge_with_inner_point.add(new_edge1)
                set_interior_edge_with_inner_point.add(new_edge2)

            #print("element inserted:",element)
            #print("Spotted edges linked with point: ",new_edge1," ",new_edge2)
            element_created=True
            #pdb.set_trace()
            if element_created:
                break




#    if plot_mesh:
#        triangulated={'segment_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
#                      'vertex_markers': np.ones([polygon.shape[0]+points.shape[0]]), 'vertices':np.vstack([ polygon,points])}
#        plot.plot(plt.axes(), **triangulated)
   # print("Final edges:",set_edges)
   # print("Elements created:",set_elements)
   # print("Set of locked vertices:", set_locked_vertices)


    # find open vertices
    for element in set_elements:
        for vertex in  element:
                    if vertex>=polygon.shape[0]:
                        continue

                    _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                    if isclosed and vertex not in set_locked_vertices:
                    #    print("Vertex locked:",vertex)
                        Found_locked_vertex=True
                        set_locked_vertices.add(vertex)
    set_open_vertices=set(range(len(polygon)))-set_locked_vertices
    #unless the interior point is connected with all remaining vertices then it is condidered open
    for i in range(len(polygon),len(polygon_with_points)):
        set_open_vertices.add(i)



    # Check for vertices that are not connecting to any point
    set_interior_edge_with_inner_point_reformed=np.array(list(set_interior_edge_with_inner_point)).flatten()
    for vertex in range(len(polygon),len(polygon_with_points)):
        if vertex not in set_interior_edge_with_inner_point_reformed:
            set_orphan_vertices.add(vertex)

    # All interior points are treated as open vertices to forcesseking sub polygons uncluding the interior points
    #for i in range(len(polygon),len(polygon_with_points)):
     #   if i not in set_orphan_vertices:
      #      set_open_vertices.add(i)
    #set_open_vertices=set_open_vertices-set_orphan_vertex

    #print("set of orphan vertex:",set_orphan_vertices)
    #print("Set of open vertices:", set_open_vertices)
    set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
    sub_element_list=[]
    if recursive:

        sub_polygon_list=check_for_sub_polygon_pure(set_orphan_vertices,set_open_vertices,interior_edges,set_elements,polygon,points)

        ################## There could an orphan point inside an element making the element a subpolygon #############################

        if len(set_orphan_vertices)!=0:
            for element in set_elements:
                triangle_indices=np.asarray(element)
                triangle=polygon_with_points[triangle_indices]
                for vtx in set_orphan_vertices:
                    point=polygon_with_points[vtx]
                    is_inside=ray_tracing(point[0],point[1],triangle)
                    if is_inside:
                        sub_polygon_list.append(list(element))








        if len(sub_polygon_list)==0:
            return set_elements,[]

        for sub_polygon_indices in sub_polygon_list:
            if len(set_orphan_vertices)==0:
                if len(sub_polygon_indices)>=3:
     #               print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])

                    if not is_counterclockwise(sub_polygon):
                        sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])
                    nb_of_edges=len(sub_polygon)
                    with open('../network_datasets/connectivity_NN/'+str(nb_of_edges)+'_NN_qualities.pkl','rb') as f:
                        connection_network=pickle.load(f)

                    contour_variable=Variable(torch.from_numpy(sub_polygon.reshape(-1)).type(torch.FloatTensor)).expand(1,len(sub_polygon.reshape(-1)))
                    connection_network=connection_network.cpu().eval()
                    prediction=connection_network(contour_variable).data[0].numpy()
                    sub_quality=prediction.reshape(nb_of_edges,nb_of_edges)
                    sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=False)
       #             print(sub_quality,sub_order_matrix)
                    sub_elements,_=Triangulation.pure_triangulate(sub_polygon,sub_order_matrix,recursive=True)
                    if len(sub_elements)!=0:
                        for element in sub_elements:
                            indices=np.asarray(element)
        #                    print(element)
                            triangle=sub_polygon[indices]
                            polygon_indices=get_indices(triangle,polygon_with_points)
                            sub_element_list.append(polygon_indices)
            else:
                if len(sub_polygon_indices)>=3:


                    sub_polygon_inner_points=[]
                    inner_points_indices=np.asarray(list(set_orphan_vertices)).flatten()
                    #inner_points_indices=np.sort(inner_points_indices)
         #           print("remeshing subpolygon",sub_polygon_indices)
                    polygon_copy=np.vstack([polygon,points])
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices])


                    if not is_counterclockwise(sub_polygon):
                            sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                    inner_points=np.array(polygon_copy[inner_points_indices])
                    inner_points=sort_points(inner_points.reshape(1,len(inner_points),2),len(inner_points)).reshape(len(inner_points),2)

                    for point in inner_points:
                        is_inside=ray_tracing(point[0],point[1],sub_polygon)
                        if is_inside:
                            sub_polygon_inner_points.append(point)
          #                  print("Point ",point," is inside ", sub_polygon_indices)


                    if len(sub_polygon_inner_points)!=0:

                        sub_polygon_inner_points=np.array(sub_polygon_inner_points)

                       # if not is_counterclockwise(sub_polygon_inner _points):
                        #    sub_polygon_inner_points=sub_polygon_inner_points[::-1]
#
                        sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
#                        sub_quality,_=quality_matrix(sub_polygon,sub_polygon_inner_points,compute_minimum=True,normalize=False)
#                        sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=True)

                        nb_of_edges=len(sub_polygon)
                        nb_of_points=len(sub_polygon_inner_points)
                        with open('../network_datasets/connectivity_NN/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_NN_qualities_with_extra_grid_points.pkl','rb') as f:
                            connection_network=pickle.load(f)

                        contour_with_point_variable=Variable(torch.from_numpy(sub_polygon_with_points.reshape(-1)).type(torch.FloatTensor)).expand(1,len(sub_polygon_with_points.reshape(-1)))
                        connection_network=connection_network.cpu().eval()
                        prediction=connection_network(contour_with_point_variable).data[0].numpy()
                        sub_quality=prediction.reshape(nb_of_edges,nb_of_edges+nb_of_points)
                        sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,sub_polygon_with_points,check_for_equal=False)

           #             print(sub_quality,sub_order_matrix)
            #            print(sub_polygon)
                        sub_elements,_=triangulate_NN_pure(sub_polygon,sub_polygon_inner_points,sub_order_matrix,recursive=True)
                        if len(sub_elements)!=0:
                            for element in sub_elements:
                                indices=np.asarray(element)
                               # print(element)
                                triangle=sub_polygon_with_points[indices]
                                polygon_indices=get_indices(triangle,polygon_with_points)
                                sub_element_list.append(polygon_indices)
                        #print("sub_polygon: " , sub_polygon)
                        #print("sub_polygon_poiny: ",sub_polygon_inner_points)

                    else:
                        # polygon_copy=np.vstack([polygon,points])
                         #sub_polygon=np.array(polygon_copy[sub_polygon_indices])
                         #if not is_counterclockwise(sub_polygon):
                          #   sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

    #                    sub_polygon_with_points=np.vstack([sub_polygon,sub_polygon_inner_points])
                         sub_quality,_=Triangulation.quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                         sub_order_matrix=Triangulation.order_quality_matrix(sub_quality,sub_polygon,check_for_equal=False)

             #            print(sub_quality,sub_order_matrix)
                         sub_elements,_=Triangulation.pure_triangulate(sub_polygon,sub_order_matrix,recursive=True)
                         if len(sub_elements)!=0:
                             for element in sub_elements:
                                 indices=np.asarray(element)
              #                   print(element)
                                 triangle=sub_polygon[indices]
                                 polygon_indices=get_indices(triangle,polygon_with_points)
                                 sub_element_list.append(polygon_indices)

    return set_elements,sub_element_list



def get_indices(triangle,polygon):

    indices=[]

    for point in triangle:
        for index,point_in_polygon in enumerate(polygon):
              if np.allclose(point,point_in_polygon):
                    indices.append(index)

    return indices


def order_quality_matrix(_quality_matrix,_polygon,_polygon_with_inner_points, check_for_equal=True):

    #  Create the quality matrix in accordance with the edges
    quality_board=[(q,index)  for qualities in _quality_matrix for index,q in enumerate(qualities)]
    quality_board=np.array(quality_board)

    #print("Quality board not resized:",quality_board)

    quality_board.resize(len(get_contour_edges(_polygon)),len(_polygon_with_inner_points),2)


    quality_board=dict(zip(list(tuple(i) for i in get_contour_edges(_polygon)),quality_board))
    #sorted_quality_board={i[0]:i[1] for i in sorted(board.items(),key=lambda x: max(x[1]),reverse=True)}
    #print("Quality board")
    #for keys,items in quality_board.items():
    #    print(keys,items)
    edge_quality=quality_board[(0,1)]
    edge_quality=edge_quality[np.lexsort(np.fliplr(edge_quality).T)]




    for i in quality_board.keys():
        quality_board[i]=quality_board[i][np.lexsort(np.fliplr(quality_board[i]).T)]
        quality_board[i]=quality_board[i][::-1]
        quality_board[i][:,1]=quality_board[i][:,1].astype(int)

    listing=[]
    for keys,values in quality_board.items():
        listing.append([keys,max(values[:,0])])

    listing=np.array(listing)
    listing=listing[np.lexsort(np.transpose(listing)[::-3]).T]
    listing=listing[::-1]
    ordered_indices=listing[:,0]

    ordered_quality_matrix={}

    for i in ordered_indices:
        ordered_quality_matrix[i]=[tuple(zip(quality_board[i][:,0],quality_board[i][:,1].astype(int)))]

    if check_for_equal:
        ordered_quality_matrix=check_ordered_matrix(ordered_quality_matrix,_polygon,_polygon_with_inner_points)
    return ordered_quality_matrix





def check_ordered_matrix(_order_matrix,polygon,polygon_with_inner_points):

   # polygon_with_point=np.vstack([polygon,polygon.sum(0)/polygon.shape[0]])
    checked_matrix=copy.deepcopy(_order_matrix)
    listing=np.empty([len(checked_matrix),len(polygon_with_inner_points)],dtype=np.float32)
    for i,keys in enumerate(checked_matrix):

        for qualities_with_indices in  checked_matrix[keys]:
            for j,(qualities,indices) in enumerate(qualities_with_indices):

                #print(qualities,indices,'\n')
                listing[i,j]=qualities
    # print(listing)
    edge_list=list(checked_matrix.keys())
   # listing=listing[::-1]
    for ind,i in enumerate(listing):
       # print("checking",edge_list[ind])
        non_zero_list=i[np.where(i!=0)]
        #non_zero_list=non_zero_list[::-1]
        unique_non_zero_list,count=np.unique(non_zero_list,return_counts=True)
        unique_non_zero_list=unique_non_zero_list[::-1]
        count=count[::-1]
        value_with_counts=list(zip(unique_non_zero_list,count))
        # print(value_with_counts)
        for j in value_with_counts:
            lst=list(checked_matrix[edge_list[ind]][0])

            if j[1]>1:
#                pdb.set_trace()
                indices=[]

                connection_vertex_with_mean_qualities=[]
                tag=j[0]
                for index,j in enumerate(non_zero_list):
                    if tag==j:
                        # print(index)
                        indices.append(index)
                        connection_vertex=int(checked_matrix[edge_list[ind]][0][index][1])
                        triangle_indices=np.asarray([edge_list[ind][0],edge_list[ind][1],connection_vertex])
                        # print("triangle",triangle_indices)
                        triangle=polygon_with_inner_points[triangle_indices]
                        mean_quality=compute_mean_quality_triangle(triangle,polygon,polygon_with_inner_points)
                        # print(mean_quality)
                        connection_vertex_with_mean_qualities.append(tuple((mean_quality,connection_vertex)))
                connection_vertex_with_mean_qualities=np.array(connection_vertex_with_mean_qualities,dtype='float32,uint16')
               # connection_vertex_with_mean_qualities[:,1]= connection_vertex_with_mean_qualities[:,1].astype(int)
                # print(connection_vertex_with_mean_qualities)
                sorted_array=np.sort(connection_vertex_with_mean_qualities,axis=0)
                sorted_array=sorted_array[::-1]

                sorted_array=[tuple(i) for i in sorted_array]
                # print(sorted_array)
                for index,k in enumerate(indices):
                    lst[k]=sorted_array[index]
            #print("replacing {} \n with {}:".format(checked_matrix[edge_list[ind]][0],lst))
            checked_matrix[edge_list[ind]][0]=tuple(lst)
        # print("checked",edge_list[ind])





    return checked_matrix







# Function to get the list of edges of a polygon
def get_contour_edges(polygon):
    contour_connectivity=np.array([[i,(i+1)%polygon.shape[0]] for i in range(polygon.shape[0])])
    return contour_connectivity



# Function to return indices that are connected to a vertex
def connection_indices(vertex,edges):
    indices=[]
    for edge in edges:
        if vertex in edge:

            if edge[0] == vertex:
                indices.append(edge[1])
            else:
                indices.append(edge[0])

    return indices

# Function to calculate and angle:
def calculate_angle(p0,p1,p2):
    v0 = p1 - p0
    v1 = p2 - p0


    #normal=compute_triangle_normals([p0,p1,p2])
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    angle=abs(angle)
    #unit_v0=v0 / np.linalg.norm(v0)
    #unit_v1=v1 / np.linalg.norm(v1)
    #angle=np.arccos(np.clip(np.dot(unit_v0, unit_v1), -1.0, 1.0))

    return np.degrees(angle)



# Function to calculate the angles of a polygon
def get_polygon_angles(polygon):
    angles=[]
    for index,point in enumerate(polygon):
        p0=point
        neighbor_points=connection_indices(index,get_contour_edges(polygon))
        #print("neighbor points",neighbor_points)
        indices=np.asarray(neighbor_points)
        p1,p2=polygon[indices]
        angle=calculate_angle(p0,p1,p2)
        if index !=0:
            triangle_normal=compute_triangle_normals([p0,p1,p2])
        else:
            triangle_normal=compute_triangle_normals([p1,p0,p2])


        if triangle_normal>0:
            angle=360-angle

        angles.append(angle)
    return angles




def is_closed_ring(vtx,set_of_elements,*adj_vtx):
    contour_edge1=(vtx,adj_vtx[0])
    contour_edge2=(vtx,adj_vtx[1])
    visited_elements=set_of_elements.copy()

    target_edge=contour_edge1

    edges_found=[]
    edges_found.append(contour_edge1)

    proceed=True

    while proceed:

        if not visited_elements:
            break

        remaining_edge,found_element=edge2elem(target_edge,visited_elements)

        if found_element is None:
            #print("stopped")
            proceed=False
            break

        visited_elements.remove(found_element)
        edges_found.append(remaining_edge)
        target_edge=remaining_edge



    #print(set(edges_found))
    found_contour_edge1,found_contour_edge2=False,False
    found_contour_edges=False

    # Checking if both contour edges area contained in the set of edges acquired

    for edge in edges_found:
        condition1= contour_edge1[0] in set(edge) and contour_edge1[1] in set((edge))
        condition2= contour_edge2[0] in set(edge) and contour_edge2[1] in set((edge))
        if condition1:
            #print("found ",contour_edge1)
            found_contour_edge1=True
        if condition2:
            #print("found",contour_edge2)
            found_contour_edge2=True

    if found_contour_edge1 and found_contour_edge2:
        found_contour_edges=True
        #print("found both of contour edges in set")

    visited_elements.clear()
    return edges_found,found_contour_edges

# See if we can get from one edge to another element wise
def is_traversable(vtx,set_of_elements,vertex1,vertex2):
    edge1=(vtx,vertex1)
    edge2=(vtx,vertex2)
    visited_elements=set_of_elements.copy()

    target_edge=edge1

    edges_found=[]
    edges_found.append(edge2)

    proceed=True

    while proceed:

        if not visited_elements:
            break

        remaining_edge,found_element=edge2elem(target_edge,visited_elements)

        if found_element is None:
            #print("stopped")
            proceed=False
            break

        visited_elements.remove(found_element)
        edges_found.append(remaining_edge)
        target_edge=remaining_edge



    #print(set(edges_found))
    found_edge1,found_edge2=False,False
    found_edges=False

    # Checking if both contour edges area contained in the set of edges acquired

    for edge in edges_found:
        condition1= edge1[0] in set(edge) and edge1[1] in set((edge))
        condition2= edge2[0] in set(edge) and edge2[1] in set((edge))
        if condition1:
            #print("found ",contour_edge1)
            found_edge1=True
        if condition2:
            #print("found",contour_edge2)
            found_edge2=True

    if found_edge1 and found_edge2:
        found_edges=True
        #print("found both of contour edges in set")

    visited_elements.clear()
    return edges_found,found_edges

# Finds element containing the edge and exits (does not give the full list of elements)
# Serve is_one_ring function
def edge2elem(edge,set_of_elements):
    Found_element=()
    Remaining_edge=()

    for element in set_of_elements.copy():

        if edge[0] in  set(element) and edge[1] in element:
            #print("Edge {} is part of element {}".format(edge,element))
            Found_element=element
            Remaining_index=set(element)-set(edge)
            Remaining_index=list(Remaining_index)
            Remaining_edge=(edge[0],Remaining_index[0])
            #print(" Remaining edge is {}".format(Remaining_edge))
            break
        else:
            Found_element=None
            Remaining_edge=None
    return  Remaining_edge,Found_element




# Find elements that are connected to a specific vertex
def vert2elem(vtx,set_of_elements):
    found_elements=set()
    for element in set_of_elements:
        if vtx in  set(element):
            found_elements.add(element)
    return found_elements

def edge2vert(vtx,polygon,set_interior_edges):
    found_edges=set()
    if vtx<polygon.shape[0]:
        found_edges.add((vtx,(vtx+1)%polygon.shape[0]))
        found_edges.add((vtx,(vtx-1)%polygon.shape[0]))
    for edge in set_interior_edges:
        if vtx in set(edge):
            if edge not in found_edges or edge[::-1] not in found_edges:
                found_edges.add(edge)
    return found_edges



# sort edges around point counterclock wise #

def sort_edges_around_vertex(vertex,edges_around_vert,polygon,points):
    polygon_with_points=np.vstack([polygon,points])
    edges_coordinates=[]
    edges_indices=[]
    for edges in edges_around_vert:
        edge_indices=np.asarray(edges if edges[0]==vertex else edges[::-1])
        edges_indices.append(edges if edges[0]==vertex else edges[::-1])
        edges_coordinates.append(polygon_with_points[edge_indices])

    edge_list={edge:edge_coordinate for edge,edge_coordinate in zip(edges_indices,edges_coordinates)}
    vertex_list={edge:edge_coordinates[1]-edge_coordinates[0] for edge,edge_coordinates in zip(edge_list.keys(),edges_coordinates)}
    vertex_coordinates=list(vertex_list.values())
    angle_list=[]
    for vertices in vertex_coordinates:
        angle=angle_counterclockwise(np.array([0,1]),vertices)
        angle_list.append(angle)


    angle_list={edge:angle for edge,angle in zip(edges_indices,angle_list)}
    sorted_edges=dict(OrderedDict(sorted(angle_list.items(),key=lambda x:x[1])))
    # print(edge_list)
    # print(sorted_edges)
    return [*sorted_edges]

def found_element_with_edges(edge1,edge2,set_elements):
    found_element=False
    edge1=set(edge1)
    edge2=set(edge2)
    possible_element=edge1.union(edge2)
    for element in set_elements:
         if set(element)==set(possible_element):
             found_element=True
    return found_element



# Departing from a target vertex connected with and edge get all intermediate  indices from one side and other
def get_intermediate_indices(target_vtx,polygon,*edge):

    set_1=set()
    set_2=set()


    contour_edges=get_contour_edges(polygon)


    # Depart from target vertex and get neighbor indices
    neighbors=connection_indices(target_vtx,contour_edges)
    found_vertex1,found_vertex2=neighbors[0],neighbors[1]
    #print("found vertices:",found_vertex1,found_vertex2)


    # Include them into seperate lists
    set_1.add(found_vertex1)
    set_2.add(found_vertex2)

    visited_vertex=target_vtx


    while found_vertex1!=edge[0] and found_vertex1!=edge[1]:
        visiting_vertex=found_vertex1
        neighbors=connection_indices(visiting_vertex,contour_edges)
        for index in neighbors:
            if index !=  visited_vertex:
                set_1.add(index)
                found_vertex1=index
                #print("Found vertex:",found_vertex1)
        visited_vertex=visiting_vertex

    #print("Start  looking the other way")

    # Resetting to go the other way
    visited_vertex=target_vtx

    while found_vertex2!=edge[0] and found_vertex2!=edge[1]:
        visiting_vertex=found_vertex2
        neighbors=connection_indices(visiting_vertex,contour_edges)
        for index in neighbors:
            if index !=  visited_vertex:
                set_2.add(index)
                found_vertex2=index
                #print("Found vertex:",found_vertex2)
        visited_vertex=visiting_vertex





    return set_1,set_2







# In[8]:

################# Adding a function to see if the points of a contour edges are linked with an inner point ##########


def linked_via_inner_point(vtx1,vtx2,edges_to_visit,set_of_open_vertices):

    vtx_set=set([vtx for edges in edges_to_visit  for vtx in edges])

    if vtx1 not in vtx_set and vtx2 not in vtx_set:
        return True
#    if vtx1 in vtx_set and vtx2 not in vtx_set:
#        return True
#    if vtx2 in vtx_set and vtx1 not in vtx_set:
#        return True
#    if vtx2 not in set_of_open_vertices:
        return True
    for edges in edges_to_visit:
        for index,vtx in enumerate(edges):
            if edges[index]==vtx1:
                adjacent_point=edges[(index+1)%2]
                for edges in edges_to_visit:
                    for index,vertices in enumerate(edges):
                        if edges[index]==adjacent_point and edges[(index+1)%2]==vtx2:
                            return True
    return False









def polygon_2_vtx(starting_vertex,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon):
    from  more_itertools import unique_everseen

    if not edges_to_visit:
        return
    added_edges=set()
    # print(" Initial edges to visit" , edges_to_visit)
#    if len(set_of_open_vertices-set_orphan_vertices)==3:
#        return

#    vertex_list=[vtx for edge in edges_to_visit for vtx in edge]
#    # The edges to visit are interior edges but it could be that  there are open vertices in the contour
#    # with edges to traverse around the contour
    if len(set_of_common_vertices)==0:
#        for vtx in set_of_open_vertices:
#            if vtx<polygon.shape[0]:
#                found_in_edges_to_visit=False
#                found_neighbor_point=False
#                for edges in edges_to_visit.copy():
#                    for index,vertices in enumerate(edges):
#                        if edges[index]==vtx:
#                            adjacent_point=edges[(index+1)%2]
#                            for edges in edges_to_visit.copy():
#                                for index,vertices in enumerate(edges):
#                                    if edges[index]==adjacent_point and edges[(index+1)%2]==(vtx+1)%polygon.shape[0] or  edges[index]==adjacent_point and edges[(index+1)%2]==(vtx-1)%polygon.shape[0]:
#                                        found_neighbor_point=True
#                                        break
##                                    for edges in edges_to_visit:
##                                        if edges==(vtx,adjacent_point) or edges==(adjacent_point,vtx):
##                                                found_in_edges_to_visit=True
##                                                break
##                                    if found_in_edges_to_visit:
##                                            break
#
#
#
#
#                if found_neighbor_point:
#                    continue
##                if found_neighbor_point or found_in_edges_to_visit :
##                    continue
##
#
#
#                if not found_in_edges_to_visit:
#                    adjacent_edge1=(vtx,(vtx+1)%polygon.shape[0])
#                    adjacent_edge2=(vtx,(vtx-1)%polygon.shape[0])
#                    if adjacent_edge1 in edges_to_visit.copy() or adjacent_edge1[::-1] in  edges_to_visit.copy():
#                        continue
#                    if adjacent_edge2 in  edges_to_visit.copy() or adjacent_edge2[::-1] in  edges_to_visit.copy():
#                        continue
#                    if adjacent_edge1[0] in set_of_open_vertices and adjacent_edge1[1]  in set_of_open_vertices:
#                        print(" Adding edge",adjacent_edge1 )
#                        edges_to_visit.add(adjacent_edge1)
#                    if adjacent_edge2[0] in set_of_open_vertices and adjacent_edge2[1]  in set_of_open_vertices:
#                        print(" Adding edge",adjacent_edge2 )
#                        edges_to_visit.add(adjacent_edge2)



        for vtx in set_of_open_vertices:
            if vtx<polygon.shape[0]:
                candidate_point1=(vtx+1)%polygon.shape[0]
                if not linked_via_inner_point(vtx,candidate_point1,edges_to_visit,set_of_open_vertices):
                    is_ok=True
                    candidate_edge1=(vtx,candidate_point1)
                    for element in set_of_elements:
                        if set(candidate_edge1).issubset( set(element)):
                            # print("Candidate edge", candidate_edge1, "found in element", element)
                            is_ok=False
                    if candidate_edge1 in edges_to_visit.copy() or candidate_edge1[::-1] in  edges_to_visit.copy():
                        is_ok=False
                    if is_ok and candidate_edge1[0] in set_of_open_vertices and candidate_edge1[1]  in set_of_open_vertices:
                        edges_to_visit.add(candidate_edge1)
                        # print("Added edge", candidate_edge1 )
                        added_edges.add(candidate_edge1)
                candidate_point2=(vtx-1)%polygon.shape[0]
                if not linked_via_inner_point(vtx,candidate_point2,edges_to_visit,set_of_open_vertices):
                    is_ok=True
                    candidate_edge2=(vtx,candidate_point2)
                    for element in set_of_elements:
                        if set(candidate_edge2).issubset( set(element)):
                            # print("Candidate edge", candidate_edge2, "found in element", element)
                            is_ok=False
                    if candidate_edge2 in edges_to_visit.copy() or candidate_edge2[::-1] in  edges_to_visit.copy():
                        is_ok=False
                    if is_ok and candidate_edge2[0] in set_of_open_vertices and candidate_edge2[1]  in set_of_open_vertices:
                        edges_to_visit.add(candidate_edge2)
                        # print("Added edge", candidate_edge2 )
                        added_edges.add(candidate_edge2)


    else:

                vertex_list=set([vtx for edges in pair_of_adjacent_edges for edge in edges for vtx in edge])-set_of_common_vertices
#                # The edges to visit are interior edges but it could be that  there are open vertices in the contour
#                # with edges to traverse around the contour
#
#                for vtx in vertex_list:
#                            found_in_edges_to_visit=False
#                            found_neighbor_point=False
#                            for edges in edges_to_visit.copy():
#                                for index,vertices in enumerate(edges):
#                                    if edges[index]==vtx:
#                                        adjacent_point=edges[(index+1)%2]
#                                        for edges in edges_to_visit.copy():
#                                            for index,vertices in enumerate(edges):
#                                                if edges[index]==adjacent_point and edges[(index+1)%2]==(vtx+1)%polygon.shape[0] or  edges[index]==adjacent_point and edges[(index+1)%2]==(vtx-1)%polygon.shape[0]:
#                                                    found_neighbor_point=True
#                                                    break
#            #                                    for edges in edges_to_visit:
#            #                                        if edges==(vtx,adjacent_point) or edges==(adjacent_point,vtx):
#            #                                                found_in_edges_to_visit=True
#            #                                                break
#            #                                    if found_in_edges_to_visit:
#            #                                            break
#
#
#
#
#                            if found_neighbor_point:
#                                continue
#            #                if found_neighbor_point or found_in_edges_to_visit :
#            #                    continue
#            #
#
#
#                            if not found_in_edges_to_visit:
#                                adjacent_edge1=(vtx,(vtx+1)%polygon.shape[0])
#                                adjacent_edge2=(vtx,(vtx-1)%polygon.shape[0])
#                                if adjacent_edge1 in edges_to_visit.copy() or adjacent_edge1[::-1] in  edges_to_visit.copy():
#                                    continue
#                                if adjacent_edge2 in  edges_to_visit.copy() or adjacent_edge2[::-1] in  edges_to_visit.copy():
#                                    continue
#                                if adjacent_edge1[0] in set_of_open_vertices and adjacent_edge1[1]  in set_of_open_vertices:
#                                    print(" Adding edge",adjacent_edge1 )
#                                    edges_to_visit.add(adjacent_edge1)
#                                if adjacent_edge2[0] in set_of_open_vertices and adjacent_edge2[1]  in set_of_open_vertices:
#                                    print(" Adding edge",adjacent_edge2 )
#                                    edges_to_visit.add(adjacent_edge2)
                for vtx in vertex_list:
                    candidate_point1=(vtx+1)%polygon.shape[0]
                    is_ok=True
                    if not linked_via_inner_point(vtx,candidate_point1,initial_edges_to_visit,set_of_open_vertices):
                        if (vtx,candidate_point1) in edges_to_visit.copy() or(vtx,candidate_point1) in  edges_to_visit.copy():
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if (vtx,candidate_point1) in edges_in_same_polygon or (candidate_point1,vtx) in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok:
                                edges_to_visit.add((vtx,candidate_point1))
                                # print("Added edge", (vtx,candidate_point1) )
                                added_edges.add((vtx,candidate_point1))


                    candidate_point2=(vtx-1)%polygon.shape[0]
                    if not linked_via_inner_point(vtx,candidate_point2,initial_edges_to_visit,set_of_open_vertices):
                        if (vtx,candidate_point2) in edges_to_visit.copy() or(vtx,candidate_point2) in  edges_to_visit.copy():
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if (vtx,candidate_point2) in edges_in_same_polygon or (candidate_point2,vtx) in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok:
                                edges_to_visit.add((vtx,candidate_point2))
                                # print("Added edge", (vtx,candidate_point2) )
                                added_edges.add((vtx,candidate_point2))

                for vtx in set_of_open_vertices:
                    if vtx<polygon.shape[0]:
                        candidate_point1=(vtx+1)%polygon.shape[0]
                        if not linked_via_inner_point(vtx,candidate_point1,edges_to_visit,set_of_open_vertices):
                            is_ok=True
                            candidate_edge1=(vtx,candidate_point1)
                            for element in set_of_elements:
                                if set(candidate_edge1).issubset( set(element)):
                                    # print("Candidate edge", candidate_edge1, "found in element", element)
                                    is_ok=False
                            if candidate_edge1 in edges_to_visit.copy() or candidate_edge1[::-1] in  edges_to_visit.copy():
                                is_ok=False
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if candidate_edge1 in edges_in_same_polygon or candidate_edge1[::-1] in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok and candidate_edge1[0] in set_of_open_vertices and candidate_edge1[1]  in set_of_open_vertices:
                                edges_to_visit.add(candidate_edge1)
                                # print("Added edge", candidate_edge1)
                                added_edges.add(candidate_edge1)



                        candidate_point2=(vtx-1)%polygon.shape[0]
                        if not linked_via_inner_point(vtx,candidate_point2,edges_to_visit,set_of_open_vertices):
                            is_ok=True
                            candidate_edge2=(vtx,candidate_point2)
                            for element in set_of_elements:
                                if set(candidate_edge2).issubset( set(element)):
                                    # print("Candidate edge", candidate_edge2, "found in element", element)
                                    is_ok=False
                            if candidate_edge2 in edges_to_visit.copy() or candidate_edge2[::-1] in  edges_to_visit.copy():
                                is_ok=False
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if candidate_edge2 in edges_in_same_polygon or candidate_edge2[::-1] in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok and candidate_edge2[0] in set_of_open_vertices and candidate_edge2[1]  in set_of_open_vertices:
                                edges_to_visit.add(candidate_edge2)
                                # print("Added edge", candidate_edge2)
                                added_edges.add(candidate_edge2)




     #### Handles subpolygons that are triangles ###
    if len(edges_to_visit)==2:
        # if they share a common vertex
        edges=[edge for edge in edges_to_visit]
        if edges[0][0]==edges[1][0] :
            edges_to_visit.add((edges[0][1],edges[1][1]))
        if edges[0][0]==edges[1][1] :
             edges_to_visit.add((edges[0][1],edges[1][0]))
        if edges[0][1]==edges[1][0] :
            edges_to_visit.add((edges[0][0],edges[1][1]))
        if edges[0][1]==edges[1][1] :
            edges_to_visit.add((edges[0][1],edges[1][0]))



    closed=False

    #pdb.set_trace()

    # print("Edges to visit:",edges_to_visit)
    subpolygon=[]

    set_of_points=set([j for i in edges_to_visit for j in i])

    if starting_vertex not in set_of_points:
        return

    found_vertex=starting_vertex
    target_edge=[]
    visited_added_edge=False
    deleted_edges=set()
    count=0
    while not closed:
        count+=1
        for index,edge in enumerate(edges_to_visit.copy()):
            visiting_vertex=found_vertex

            if target_edge:
 #               pdb.set_trace()
                if edge!= target_edge[0] and  edge!= tuple(reversed(target_edge[0])) :
                    continue
                else:
                    target_edge.pop()
            #if visiting_vertex not in set(edge) and index==len(edges_to_visit.copy()):
               # Tracer()()
                #print("Not found in list of edges")
                #closed=True
                #break
            adjacent_edge1=(visiting_vertex,(visiting_vertex+1)%polygon.shape[0])
            adjacent_edge2=(visiting_vertex,(visiting_vertex-1)%polygon.shape[0])



            if visiting_vertex not in set(edge):
                if index==int(len(edges_to_visit))-1 and visited_added_edge:
                    # print(" Reached end found no matching vertex after visiting added edge ")
                    for edge in deleted_edges:
                        edges_to_visit.add(edge)
                    return 0
                if len(edges_to_visit)==0 and visited_added_edge:
                    # print(" Reached end found no matching vertex after visiting added edge ")
                    for edge in deleted_edges:
                        edges_to_visit.add(edge)
                    return 0
                continue
            subpolygon.append(visiting_vertex)


            # print("Visiting vertex",visiting_vertex)

        #    found_starting_vtx=False
            subpolygon.append(found_vertex)


            # print(visiting_vertex," in ", edge)
            #  Starting from a visiting vertex may not be a good idea because we don't know if it will be included to close a polygon
            if (edge in added_edges or edge[::-1] in added_edges) and count==1:
                continue




            for index in set(edge):
                if visiting_vertex!= index:
                    found_vertex=index
                    # print("Found vertex:",found_vertex)
                    subpolygon.append(found_vertex)
            found_crossroad=False
            found_in_set=False
            # Check if edge is part of a crossroad (check if found vertex is point of multiple polygons)
            if found_vertex in set_of_common_vertices:
                found_crossroad=True

            # If yes then the next visiting edge should be the one is the pair of adjacent edges
            duplicate_edge=False



            if found_crossroad:




                for  edges_in_same_polygon in pair_of_adjacent_edges.copy():
                    if edge in set(edges_in_same_polygon) or tuple(reversed(edge)) in set(edges_in_same_polygon):
                        for edges in edges_in_same_polygon:
                            if edges!=edge and edges!=tuple(reversed(edge)):
                                target_edge.append(edges)
                                found_in_set=True
                                # print("edge {} should be followed by {}".format(edge,edges))
                                count=0
                                for edges_in_same_polygon in pair_of_adjacent_edges:
                                    for edges in edges_in_same_polygon:
                                        if edge==edges or edge[::-1]==edges:
                                            count+=1
                                if count>1:
                                    # print("found duplicate edge ",edge)
                                    duplicate_edge=True



                                pair_of_adjacent_edges.discard(edges_in_same_polygon)



                                break
                    if found_in_set:
                        break

            if not duplicate_edge :
                # print("Removing edge",edge)
                if edge in added_edges:
                    visited_added_edge=True
                if edge  not in added_edges:
                    deleted_edges.add(edge)
                edges_to_visit.discard(edge)
            # print(edges_to_visit)
            if found_vertex==starting_vertex:
                subpolygon=list(unique_everseen(subpolygon))
                # print("Back to starting vertex")
                closed=True
                break

    if  len(subpolygon)<3:
        return
    else:
        return subpolygon




def polygon_2_vtx_pure(starting_vertex,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon):
    from  more_itertools import unique_everseen

    if not edges_to_visit:
        return
    added_edges=set()
   # print(" Initial edges to visit" , edges_to_visit)
#    if len(set_of_open_vertices-set_orphan_vertices)==3:
#        return

#    vertex_list=[vtx for edge in edges_to_visit for vtx in edge]
#    # The edges to visit are interior edges but it could be that  there are open vertices in the contour
#    # with edges to traverse around the contour
    if len(set_of_common_vertices)==0:
#        for vtx in set_of_open_vertices:
#            if vtx<polygon.shape[0]:
#                found_in_edges_to_visit=False
#                found_neighbor_point=False
#                for edges in edges_to_visit.copy():
#                    for index,vertices in enumerate(edges):
#                        if edges[index]==vtx:
#                            adjacent_point=edges[(index+1)%2]
#                            for edges in edges_to_visit.copy():
#                                for index,vertices in enumerate(edges):
#                                    if edges[index]==adjacent_point and edges[(index+1)%2]==(vtx+1)%polygon.shape[0] or  edges[index]==adjacent_point and edges[(index+1)%2]==(vtx-1)%polygon.shape[0]:
#                                        found_neighbor_point=True
#                                        break
##                                    for edges in edges_to_visit:
##                                        if edges==(vtx,adjacent_point) or edges==(adjacent_point,vtx):
##                                                found_in_edges_to_visit=True
##                                                break
##                                    if found_in_edges_to_visit:
##                                            break
#
#
#
#
#                if found_neighbor_point:
#                    continue
##                if found_neighbor_point or found_in_edges_to_visit :
##                    continue
##
#
#
#                if not found_in_edges_to_visit:
#                    adjacent_edge1=(vtx,(vtx+1)%polygon.shape[0])
#                    adjacent_edge2=(vtx,(vtx-1)%polygon.shape[0])
#                    if adjacent_edge1 in edges_to_visit.copy() or adjacent_edge1[::-1] in  edges_to_visit.copy():
#                        continue
#                    if adjacent_edge2 in  edges_to_visit.copy() or adjacent_edge2[::-1] in  edges_to_visit.copy():
#                        continue
#                    if adjacent_edge1[0] in set_of_open_vertices and adjacent_edge1[1]  in set_of_open_vertices:
#                        print(" Adding edge",adjacent_edge1 )
#                        edges_to_visit.add(adjacent_edge1)
#                    if adjacent_edge2[0] in set_of_open_vertices and adjacent_edge2[1]  in set_of_open_vertices:
#                        print(" Adding edge",adjacent_edge2 )
#                        edges_to_visit.add(adjacent_edge2)



        for vtx in set_of_open_vertices:
            if vtx<polygon.shape[0]:
                candidate_point1=(vtx+1)%polygon.shape[0]
                if not linked_via_inner_point(vtx,candidate_point1,edges_to_visit,set_of_open_vertices):
                    is_ok=True
                    candidate_edge1=(vtx,candidate_point1)
                    for element in set_of_elements:
                        if set(candidate_edge1).issubset( set(element)):
                     #       print("Candidate edge", candidate_edge1, "found in element", element)
                            is_ok=False
                    if candidate_edge1 in edges_to_visit.copy() or candidate_edge1[::-1] in  edges_to_visit.copy():
                        is_ok=False
                    if is_ok and candidate_edge1[0] in set_of_open_vertices and candidate_edge1[1]  in set_of_open_vertices:
                        edges_to_visit.add(candidate_edge1)
                      #  print("Added edge", candidate_edge1 )
                        added_edges.add(candidate_edge1)
                candidate_point2=(vtx-1)%polygon.shape[0]
                if not linked_via_inner_point(vtx,candidate_point2,edges_to_visit,set_of_open_vertices):
                    is_ok=True
                    candidate_edge2=(vtx,candidate_point2)
                    for element in set_of_elements:
                        if set(candidate_edge2).issubset( set(element)):
                       #     print("Candidate edge", candidate_edge2, "found in element", element)
                            is_ok=False
                    if candidate_edge2 in edges_to_visit.copy() or candidate_edge2[::-1] in  edges_to_visit.copy():
                        is_ok=False
                    if is_ok and candidate_edge2[0] in set_of_open_vertices and candidate_edge2[1]  in set_of_open_vertices:
                        edges_to_visit.add(candidate_edge2)
                        #print("Added edge", candidate_edge2 )
                        added_edges.add(candidate_edge2)


    else:

                vertex_list=set([vtx for edges in pair_of_adjacent_edges for edge in edges for vtx in edge])-set_of_common_vertices
#                # The edges to visit are interior edges but it could be that  there are open vertices in the contour
#                # with edges to traverse around the contour
#
#                for vtx in vertex_list:
#                            found_in_edges_to_visit=False
#                            found_neighbor_point=False
#                            for edges in edges_to_visit.copy():
#                                for index,vertices in enumerate(edges):
#                                    if edges[index]==vtx:
#                                        adjacent_point=edges[(index+1)%2]
#                                        for edges in edges_to_visit.copy():
#                                            for index,vertices in enumerate(edges):
#                                                if edges[index]==adjacent_point and edges[(index+1)%2]==(vtx+1)%polygon.shape[0] or  edges[index]==adjacent_point and edges[(index+1)%2]==(vtx-1)%polygon.shape[0]:
#                                                    found_neighbor_point=True
#                                                    break
#            #                                    for edges in edges_to_visit:
#            #                                        if edges==(vtx,adjacent_point) or edges==(adjacent_point,vtx):
#            #                                                found_in_edges_to_visit=True
#            #                                                break
#            #                                    if found_in_edges_to_visit:
#            #                                            break
#
#
#
#
#                            if found_neighbor_point:
#                                continue
#            #                if found_neighbor_point or found_in_edges_to_visit :
#            #                    continue
#            #
#
#
#                            if not found_in_edges_to_visit:
#                                adjacent_edge1=(vtx,(vtx+1)%polygon.shape[0])
#                                adjacent_edge2=(vtx,(vtx-1)%polygon.shape[0])
#                                if adjacent_edge1 in edges_to_visit.copy() or adjacent_edge1[::-1] in  edges_to_visit.copy():
#                                    continue
#                                if adjacent_edge2 in  edges_to_visit.copy() or adjacent_edge2[::-1] in  edges_to_visit.copy():
#                                    continue
#                                if adjacent_edge1[0] in set_of_open_vertices and adjacent_edge1[1]  in set_of_open_vertices:
#                                    print(" Adding edge",adjacent_edge1 )
#                                    edges_to_visit.add(adjacent_edge1)
#                                if adjacent_edge2[0] in set_of_open_vertices and adjacent_edge2[1]  in set_of_open_vertices:
#                                    print(" Adding edge",adjacent_edge2 )
#                                    edges_to_visit.add(adjacent_edge2)
                for vtx in vertex_list:
                    candidate_point1=(vtx+1)%polygon.shape[0]
                    is_ok=True
                    if not linked_via_inner_point(vtx,candidate_point1,initial_edges_to_visit,set_of_open_vertices):
                        if (vtx,candidate_point1) in edges_to_visit.copy() or(vtx,candidate_point1) in  edges_to_visit.copy():
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if (vtx,candidate_point1) in edges_in_same_polygon or (candidate_point1,vtx) in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok:
                                edges_to_visit.add((vtx,candidate_point1))
                         #       print("Added edge", (vtx,candidate_point1) )
                                added_edges.add((vtx,candidate_point1))


                    candidate_point2=(vtx-1)%polygon.shape[0]
                    if not linked_via_inner_point(vtx,candidate_point2,initial_edges_to_visit,set_of_open_vertices):
                        if (vtx,candidate_point2) in edges_to_visit.copy() or(vtx,candidate_point2) in  edges_to_visit.copy():
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if (vtx,candidate_point2) in edges_in_same_polygon or (candidate_point2,vtx) in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok:
                                edges_to_visit.add((vtx,candidate_point2))
                          #      print("Added edge", (vtx,candidate_point2) )
                                added_edges.add((vtx,candidate_point2))

                for vtx in set_of_open_vertices:
                    if vtx<polygon.shape[0]:
                        candidate_point1=(vtx+1)%polygon.shape[0]
                        if not linked_via_inner_point(vtx,candidate_point1,edges_to_visit,set_of_open_vertices):
                            is_ok=True
                            candidate_edge1=(vtx,candidate_point1)
                            for element in set_of_elements:
                                if set(candidate_edge1).issubset( set(element)):
                         #           print("Candidate edge", candidate_edge1, "found in element", element)
                                    is_ok=False
                            if candidate_edge1 in edges_to_visit.copy() or candidate_edge1[::-1] in  edges_to_visit.copy():
                                is_ok=False
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if candidate_edge1 in edges_in_same_polygon or candidate_edge1[::-1] in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok and candidate_edge1[0] in set_of_open_vertices and candidate_edge1[1]  in set_of_open_vertices:
                                edges_to_visit.add(candidate_edge1)
                          #      print("Added edge", candidate_edge1)
                                added_edges.add(candidate_edge1)



                        candidate_point2=(vtx-1)%polygon.shape[0]
                        if not linked_via_inner_point(vtx,candidate_point2,edges_to_visit,set_of_open_vertices):
                            is_ok=True
                            candidate_edge2=(vtx,candidate_point2)
                            for element in set_of_elements:
                                if set(candidate_edge2).issubset( set(element)):
                           #         print("Candidate edge", candidate_edge2, "found in element", element)
                                    is_ok=False
                            if candidate_edge2 in edges_to_visit.copy() or candidate_edge2[::-1] in  edges_to_visit.copy():
                                is_ok=False
                            for edges_in_same_polygon in initial_pair_of_adjacent_edges:
                                if candidate_edge2 in edges_in_same_polygon or candidate_edge2[::-1] in edges_in_same_polygon:
                                    is_ok=False
                            if is_ok and candidate_edge2[0] in set_of_open_vertices and candidate_edge2[1]  in set_of_open_vertices:
                                edges_to_visit.add(candidate_edge2)
                            #    print("Added edge", candidate_edge2)
                                added_edges.add(candidate_edge2)




     #### Handles subpolygons that are triangles ###
    if len(edges_to_visit)==2:
        # if they share a common vertex
        edges=[edge for edge in edges_to_visit]
        if edges[0][0]==edges[1][0] :
            edges_to_visit.add((edges[0][1],edges[1][1]))
        if edges[0][0]==edges[1][1] :
             edges_to_visit.add((edges[0][1],edges[1][0]))
        if edges[0][1]==edges[1][0] :
            edges_to_visit.add((edges[0][0],edges[1][1]))
        if edges[0][1]==edges[1][1] :
            edges_to_visit.add((edges[0][1],edges[1][0]))



    closed=False

    #pdb.set_trace()

   # print("Edges to visit:",edges_to_visit)
    subpolygon=[]

    set_of_points=set([j for i in edges_to_visit for j in i])

    if starting_vertex not in set_of_points:
        return

    found_vertex=starting_vertex
    target_edge=[]
    visited_added_edge=False
    deleted_edges=set()
    count=0
    while not closed:
        count+=1
        for index,edge in enumerate(edges_to_visit.copy()):
            visiting_vertex=found_vertex

            if target_edge:
 #               pdb.set_trace()
                if edge!= target_edge[0] and  edge!= tuple(reversed(target_edge[0])) :
                    continue
                else:
                    target_edge.pop()
            #if visiting_vertex not in set(edge) and index==len(edges_to_visit.copy()):
               # Tracer()()
                #print("Not found in list of edges")
                #closed=True
                #break
            adjacent_edge1=(visiting_vertex,(visiting_vertex+1)%polygon.shape[0])
            adjacent_edge2=(visiting_vertex,(visiting_vertex-1)%polygon.shape[0])



            if visiting_vertex not in set(edge):
                if index==int(len(edges_to_visit))-1 and visited_added_edge:
                   # print(" Reached end found no matching vertex after visiting added edge ")
                    for edge in deleted_edges:
                        edges_to_visit.add(edge)
                    return 0
                if len(edges_to_visit)==0 and visited_added_edge:
                   # print(" Reached end found no matching vertex after visiting added edge ")
                    for edge in deleted_edges:
                        edges_to_visit.add(edge)
                    return 0
                continue
            subpolygon.append(visiting_vertex)


          #  print("Visiting vertex",visiting_vertex)

        #    found_starting_vtx=False
            subpolygon.append(found_vertex)


           # print(visiting_vertex," in ", edge)
            #  Starting from a visiting vertex may not be a good idea because we don't know if it will be included to close a polygon
            if (edge in added_edges or edge[::-1] in added_edges) and count==1:
                continue




            for index in set(edge):
                if visiting_vertex!= index:
                    found_vertex=index
                #    print("Found vertex:",found_vertex)
                    subpolygon.append(found_vertex)
            found_crossroad=False
            found_in_set=False
            # Check if edge is part of a crossroad (check if found vertex is point of multiple polygons)
            if found_vertex in set_of_common_vertices:
                found_crossroad=True

            # If yes then the next visiting edge should be the one is the pair of adjacent edges
            duplicate_edge=False



            if found_crossroad:




                for  edges_in_same_polygon in pair_of_adjacent_edges.copy():
                    if edge in set(edges_in_same_polygon) or tuple(reversed(edge)) in set(edges_in_same_polygon):
                        for edges in edges_in_same_polygon:
                            if edges!=edge and edges!=tuple(reversed(edge)):
                                target_edge.append(edges)
                                found_in_set=True
                              #  print("edge {} should be followed by {}".format(edge,edges))
                                count=0
                                for edges_in_same_polygon in pair_of_adjacent_edges:
                                    for edges in edges_in_same_polygon:
                                        if edge==edges or edge[::-1]==edges:
                                            count+=1
                                if count>1:
                               #     print("found duplicate edge ",edge)
                                    duplicate_edge=True



                                pair_of_adjacent_edges.discard(edges_in_same_polygon)



                                break
                    if found_in_set:
                        break

            if not duplicate_edge :
                #print("Removing edge",edge)
                if edge in added_edges:
                    visited_added_edge=True
                if edge  not in added_edges:
                    deleted_edges.add(edge)
                edges_to_visit.discard(edge)
            #print(edges_to_visit)
            if found_vertex==starting_vertex:
                subpolygon=list(unique_everseen(subpolygon))
             #   print("Back to starting vertex")
                closed=True
                break

    if  len(subpolygon)<3:
        return
    else:
        return subpolygon


# Checking if a point inside the contour is closed
def is_closed_interior_point(interior_point,set_of_interior_edges,set_of_elements):
    is_closed = False
    # print("Checking if interior point {} is closed".format(interior_point))
    found_edge=False
    for edge in set_of_interior_edges:
        if interior_point in edge:
            for index in edge:
                if index!=interior_point:
                    first_found_index= index
                    found_edge=True
            # print("found {} in {}".format(interior_point,edge))
            break
    # the interior is not linked with any point
    if not found_edge:
        return is_closed

    keep_looking=True
    visited_elements=set()
    while keep_looking:
        for index,element in enumerate(set_of_elements):
            if set(edge).issubset(set(element)) and element not in visited_elements:
                visited_elements.add(element)
                found_index=[int(i) for i in set(element)-set(edge)]
                # print("found index {} in element {} ".format(found_index,element))
                # Change edge value
                lst=list(edge)
                lst=[interior_point,found_index[0]]
                edge=tuple(lst)
                if found_index==first_found_index:
                    is_closed=True
                    keep_looking=False
                    break
            elif not set(edge).issubset(set(element)) and index==len(set_of_elements)-1:
                break
            keep_looking=False
            # print("Interior vertex {} is open".format(interior_point))



    return is_closed


# Checking if a point inside the contour is closed
def is_closed_interior_point_pure(interior_point,set_of_interior_edges,set_of_elements):
    is_closed = False
    #print("Checking if interior point {} is closed".format(interior_point))
    found_edge=False
    for edge in set_of_interior_edges:
        if interior_point in edge:
            for index in edge:
                if index!=interior_point:
                    first_found_index= index
                    found_edge=True
     #       print("found {} in {}".format(interior_point,edge))
            break
    # the interior is not linked with any point
    if not found_edge:
        return is_closed

    keep_looking=True
    visited_elements=set()
    while keep_looking:
        for index,element in enumerate(set_of_elements):
            if set(edge).issubset(set(element)) and element not in visited_elements:
                visited_elements.add(element)
                found_index=[int(i) for i in set(element)-set(edge)]
      #          print("found index {} in element {} ".format(found_index,element))
                # Change edge value
                lst=list(edge)
                lst=[interior_point,found_index[0]]
                edge=tuple(lst)
                if found_index==first_found_index:
                    is_closed=True
                    keep_looking=False
                    break
            elif not set(edge).issubset(set(element)) and index==len(set_of_elements)-1:
                break
            keep_looking=False
       #     print("Interior vertex {} is open".format(interior_point))



    return is_closed





def check_for_sub_polygon_pure(set_orphan_vertices,set_of_open_vertices,set_of_interior_edges,set_of_elements,polygon,points):

    set_polygon_edges=set(tuple(i) for i in get_contour_edges(polygon))


    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []


    sub_polygon_list=[]
    modified_interior_edge_set=set_of_interior_edges.copy()




    polygon_connectivity=[tuple(i) for i in get_contour_edges(polygon)]

    for edge in modified_interior_edge_set.copy():
        if edge[0] not in set_of_open_vertices or edge[1] not in set_of_open_vertices:
            modified_interior_edge_set.discard(edge)





    # Taking care of vertices that are locked but the element is not seen

    set_of_unfound_locked_vertices=set()
    continue_looking=True


    while continue_looking:

        if not set_of_open_vertices :
            continue_looking=False

        set_of_open_vertices_copy=set_of_open_vertices

        for vtx in set_of_open_vertices_copy :
                found_locked_vtx=False
                if vtx>=polygon.shape[0]:
                    is_closed=is_closed_interior_point_pure(vtx,set_of_interior_edges,set_of_elements)
                    if is_closed:
                        set_of_open_vertices.discard(vtx)
                       # print("vtx {} is closed after all".format(vtx))
                        continue_looking=False
                    else:
                        continue_looking=False
                    break
                vtx1,vtx2 =connection_indices(vtx,get_contour_edges(polygon))
                found_edges1,isclosed1=is_closed_ring(vtx,set_of_elements,vtx2,vtx1)
                found_edges2,isclosed2=is_closed_ring(vtx,set_of_elements,vtx1,vtx2)
                #print("Examining if vtx {} is locked".format(vtx))

                if isclosed1 or isclosed2:
                 #   print(vtx,"locked after all")
                    continue_looking=True

                    set_of_open_vertices.discard(vtx)
                    for edge in modified_interior_edge_set.copy():
                        if vtx in edge:
                            modified_interior_edge_set.discard(edge)
                    break

                for edge in found_edges1:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges1.remove(edge)
                for edge in found_edges2:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges2.remove(edge)
                between_edges=[]
                for edge in found_edges1:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in found_edges2:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in set_of_interior_edges.copy():
                    found_locked_vtx=False
                    if set(between_edges)==set(edge):
                  #      print(vtx,"locked after all")
                        found_locked_vtx=True
                        set_of_unfound_locked_vertices.add(vtx)
                        #Tracer()()
                        if edge in set_of_interior_edges or edge[::-1] in set_of_interior_edges:
                            #modified_interior_edge_set.discard(edge)
                            #print(edge,"removed")
                            #modified_interior_edge_set.discard(edge[::-1])
                            modified_interior_edge_set.discard((vtx,between_edges[0]))
                            modified_interior_edge_set.discard((between_edges[0],vtx))


                            modified_interior_edge_set.discard((vtx,between_edges[1]))
                            modified_interior_edge_set.discard((between_edges[1],vtx))
                            element=(vtx,between_edges[0],between_edges[1])
                   #         print("Removed:",(vtx),"from set of open vertices")

                    #        print("Added new element:",element)
                     #       print("Removed:",(vtx,between_edges[0]),"from set of edges")
                      #      print("Removed:",(vtx,between_edges[1]),"from set of edges")

                            set_of_elements.add(element)
                       #     print("New set of elements",set_of_elements)
                            set_of_open_vertices.discard(vtx)

                    if found_locked_vtx:
                        #Tracer()()
                        continue_looking=True
                        #print("Re-evaluting set of open vertices")
                        break

                    else: continue_looking=False
                if found_locked_vtx:
                    break



    #    for edge in modified_interior_edge_set.copy():
    #        if set(edge).issubset(set_of_unfound_locked_vertices):
    #            modified_interior_edge_set.discard(edge)
#            modified_interior_edge_set.discard(edge[::-1])
#            print("removed",edge)

            #print("inbetween",between_edges)

    #print("set of open vertices",set_of_open_vertices)

    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []

    # In the set of open vertices there may be vertices that are part of  of multiple polygon
    #found_common_vertex=False

    set_of_common_vertices=set()
    pair_of_adjacent_edges=set()
    for vertex in set_of_open_vertices:
        nb_of_polygon=0
        count=0
        for edge in modified_interior_edge_set.copy():
            counter2=0
            if vertex in set(edge):
                count+=1
            for element in set_of_elements:
                if set(edge).issubset(set(element)):
                    counter2+=1
            if counter2==2:
     #           print("Edge {} is common for two elements".format(edge))
                count-=1
                modified_interior_edge_set.discard(edge)

        if count>=3:

            adj_vertices=sorted(list(vtx for edge in modified_interior_edge_set if vertex  in set(edge) for  vtx in edge if vtx!=vertex))
#            if len(adj_vertices)%2==1:
#                break
            counter=0
            # Checking if vertice are linked , if they are then aren't part of the same polygon
        #    keep_looking=True
           # while keep_looking:
            for index,_ in enumerate(adj_vertices.copy()):
                edge=tuple((adj_vertices[index],adj_vertices[(index+1)%len(adj_vertices)]))
                condition3=True
                # CHECK  CONDITION TO FIND OUT WHICH EDGE IS PAIRED WITH WHICH TO FORM A POLYGON


                # Connections could form elements that are not discovered
                if ((edge in set_of_interior_edges or tuple(reversed(edge)) in set_of_interior_edges )and
                ((vertex,edge[0]) in set_of_interior_edges or tuple(reversed((vertex,edge[0]))) in set_of_interior_edges) and
                ((vertex,edge[1]) in set_of_interior_edges or tuple(reversed((vertex,edge[1]))) in set_of_interior_edges) ):
      #              print("Found new element:",(vertex,edge[0],edge[1]))
       #             print("({},{}) and ({},{}) are part of the same element".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
  #                  adj_vertices.remove(edge[0])
    #                adj_vertices.remove(edge[1])
                    continue

                if(edge[0]<edge[1]):
                    #if edge[0]<=polygon.shape[0] and edge[1]<=polygon.shape[0]:
                        for i in range(edge[0]+1,edge[1]) :
                            if i< polygon.shape[0]:
                                if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                    or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                    if i==vertex:continue
                                    condition3=False
                            else:#interior point#
                                for i in range(edge[0]+1,edge[1]) :
                                    if ((vertex,i)  in set_of_interior_edges  or (i,vertex) in set_of_interior_edges ):
                                        if i==vertex:continue
                                        condition3=False


                else:
                    for i in range(edge[0]+1,len(polygon)-1) :

                        if i< polygon.shape[0]:

                            if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                if i==vertex:continue
                                condition3=False
                        else:

                            if ((vertex,i)  in set_of_interior_edges  or  (i,vertex) in set_of_interior_edges ):
                                if i==vertex:continue
                                condition3=False

                    for i in range(edge[1]) :
                        if i<polygon.shape[0]:
                            if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                if i == vertex:continue
                                condition3=False
                        else:
                            if ((vertex,i)  in set_of_interior_edges  or  (i,vertex) in set_of_interior_edges ):
                                if i == vertex:continue
                                condition3=False
                d,s=is_traversable(vertex,set_of_elements,edge[0],edge[1])
                elements_around_vertex=vert2elem(vertex,set_of_elements)
                edges_around_vertex=edge2vert(vertex,polygon,set_of_interior_edges)
                edge_star=sort_edges_around_vertex(vertex,edges_around_vertex,polygon,points)
        #        print("edges star of common vertex" ,vertex, "is : ",edge_star)
        #        print("elements around {} are {}".format(vertex,elements_around_vertex))

                for edge1 in edge_star:
                    position_of_edge1=edge_star.index(edge1)

                    for edge2 in edge_star:
                        if edge2==edge1:
                            continue


                        position_of_edge2=edge_star.index(edge2)
                        if (abs(position_of_edge1-position_of_edge2)==1  or abs(position_of_edge1-position_of_edge2)==len(edge_star)+1) and not found_element_with_edges(edge1,edge2,elements_around_vertex):
                            if (edge1,edge2) not in pair_of_adjacent_edges and (edge2,edge1) not in pair_of_adjacent_edges:
                                 if (edge1  not in set_polygon_edges and edge1[::-1] not in set_polygon_edges) or (edge2 not in set_polygon_edges and edge2[::-1]  not in set_polygon_edges):
         #                            print(edge1,"is in  the same polygon with", edge2)

                                     pair_of_adjacent_edges.add(((edge2,edge1)))
                                     if edge2 not in modified_interior_edge_set and edge2[::-1]  not in modified_interior_edge_set:
                                         modified_interior_edge_set.add(edge2[::-1])
                                     if edge1 not in modified_interior_edge_set and edge1[::-1]  not in modified_interior_edge_set:
                                         modified_interior_edge_set.add(edge1[::-1])



#                print(edges_around_vertex)



                condition1=edge  in set_polygon_edges
                condition2=tuple(reversed(edge)) in set_polygon_edges
                nb_of_polygons=[]
                if not condition1 and not condition2  and condition3 :
                    counter+=1
#                        adj_vertices.remove(edge[0])
#                        adj_vertices.remove(edge[1])
          #          print("({},{}) and ({},{}) are part of the same polygon".format(edge[0],vertex,edge[1],vertex))
                  #  pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
#                        if len(adj_vertices)==0:
#                            keep_looking=False
#                        break
    #        keep_looking=False


            set_of_common_vertices.add(vertex)
            nb_of_polygons.append(counter)
           # print("vertex {} is adjacent to {} polygons".format(vertex,counter))
           # print("Set of adjacent edges to visit:",pair_of_adjacent_edges)
       # found_common_vertex=True

    # An edge could be part of more than one polygons. This means that the vertices of this edge
    # are already in the set of common vertices and the edges is inside the set of the of modi
    # fied interior edges
   # set_of_common_edges=set()
    #for vtx1 in  set_of_common_vertices:
     #   for vtx2 in set_of_common_vertices:
      #      pass





    # if the set found is les than 4 then now polygon is formed
    if len(set_of_open_vertices)<4:
        return []

    edges_to_visit=modified_interior_edge_set



    sub_polygon_list=[]
 #   pdb.set_trace()
    initial_edges_to_visit=copy.deepcopy(edges_to_visit)
    initial_pair_of_adjacent_edges=copy.deepcopy(pair_of_adjacent_edges)
    try:
        if set_of_common_vertices:
            for vtx in set_of_common_vertices:
                subpolygon=polygon_2_vtx_pure(vtx,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon)
                if subpolygon is not None and subpolygon != 0:
                   sub_polygon_list.append(subpolygon)
                if subpolygon == 0:
                    continue

        #print(sub_polygon_list)
    except:
        pass
        #print("Failed")



    # Removing eges where one vertex is locked and the other is not:
    for edge in edges_to_visit.copy():
        if (edge[0] in set_of_open_vertices and edge[1] not in set_of_open_vertices) or (edge[1] in set_of_open_vertices and edge[0] not in set_of_open_vertices):
            edges_to_visit.discard(edge)
            #print("Removing",edge,"from edges to visit")
            #print("Edges to visit are now",edges_to_visit)


    while edges_to_visit:
        for vtx in set_of_open_vertices.copy():
          #  print("Starting with vertex",vtx)
            subpolygon=polygon_2_vtx_pure(vtx,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon)
            if subpolygon is not None and subpolygon != 0:
                   sub_polygon_list.append(subpolygon)
            if subpolygon == 0:
                    continue


#    for sub_polygon in sub_polygon_list:
#        if len(sub_polygon)>3:
#            print("found polygon",sub_polygon)
#        else:
#            print("found element",sub_polygon)
    return sub_polygon_list


def check_for_sub_polygon(set_orphan_vertices,set_of_open_vertices,set_of_interior_edges,set_of_elements,polygon,points) :

    set_polygon_edges=set(tuple(i) for i in get_contour_edges(polygon))


    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []


    sub_polygon_list=[]
    modified_interior_edge_set=set_of_interior_edges.copy()




    polygon_connectivity=[tuple(i) for i in get_contour_edges(polygon)]

    for edge in modified_interior_edge_set.copy():
        if edge[0] not in set_of_open_vertices or edge[1] not in set_of_open_vertices:
            modified_interior_edge_set.discard(edge)





    # Taking care of vertices that are locked but the element is not seen

    set_of_unfound_locked_vertices=set()
    continue_looking=True


    while continue_looking:

        if not set_of_open_vertices :
            continue_looking=False

        set_of_open_vertices_copy=set_of_open_vertices

        for vtx in set_of_open_vertices_copy :
                found_locked_vtx=False
                if vtx>=polygon.shape[0]:
                    is_closed=is_closed_interior_point(vtx,set_of_interior_edges,set_of_elements)
                    if is_closed:
                        set_of_open_vertices.discard(vtx)
                        # print("vtx {} is closed after all".format(vtx))
                        continue_looking=False
                    else:
                        continue_looking=False
                    break
                vtx1,vtx2 =connection_indices(vtx,get_contour_edges(polygon))
                found_edges1,isclosed1=is_closed_ring(vtx,set_of_elements,vtx2,vtx1)
                found_edges2,isclosed2=is_closed_ring(vtx,set_of_elements,vtx1,vtx2)
                # print("Examining if vtx {} is locked".format(vtx))

                if isclosed1 or isclosed2:
                    # print(vtx,"locked after all")
                    continue_looking=True

                    set_of_open_vertices.discard(vtx)
                    for edge in modified_interior_edge_set.copy():
                        if vtx in edge:
                            modified_interior_edge_set.discard(edge)
                    break

                for edge in found_edges1:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges1.remove(edge)
                for edge in found_edges2:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges2.remove(edge)
                between_edges=[]
                for edge in found_edges1:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in found_edges2:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in set_of_interior_edges.copy():
                    found_locked_vtx=False
                    if set(between_edges)==set(edge):
                        # print(vtx,"locked after all")
                        found_locked_vtx=True
                        set_of_unfound_locked_vertices.add(vtx)
                        #Tracer()()
                        if edge in set_of_interior_edges or edge[::-1] in set_of_interior_edges:
                            #modified_interior_edge_set.discard(edge)
                            #print(edge,"removed")
                            #modified_interior_edge_set.discard(edge[::-1])
                            modified_interior_edge_set.discard((vtx,between_edges[0]))
                            modified_interior_edge_set.discard((between_edges[0],vtx))


                            modified_interior_edge_set.discard((vtx,between_edges[1]))
                            modified_interior_edge_set.discard((between_edges[1],vtx))
                            element=(vtx,between_edges[0],between_edges[1])
                            # print("Removed:",(vtx),"from set of open vertices")

                            # print("Added new element:",element)
                            # print("Removed:",(vtx,between_edges[0]),"from set of edges")
                            # print("Removed:",(vtx,between_edges[1]),"from set of edges")

                            set_of_elements.add(element)
                            # print("New set of elements",set_of_elements)
                            set_of_open_vertices.discard(vtx)

                    if found_locked_vtx:
                        #Tracer()()
                        continue_looking=True
                        # print("Re-evaluting set of open vertices")
                        break

                    else: continue_looking=False
                if found_locked_vtx:
                    break



    #    for edge in modified_interior_edge_set.copy():
    #        if set(edge).issubset(set_of_unfound_locked_vertices):
    #            modified_interior_edge_set.discard(edge)
#            modified_interior_edge_set.discard(edge[::-1])
#            print("removed",edge)

            #print("inbetween",between_edges)

    # print("set of open vertices",set_of_open_vertices)

    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []

    # In the set of open vertices there may be vertices that are part of  of multiple polygon
    #found_common_vertex=False

    set_of_common_vertices=set()
    pair_of_adjacent_edges=set()
    for vertex in set_of_open_vertices:
        nb_of_polygon=0
        count=0
        for edge in modified_interior_edge_set.copy():
            counter2=0
            if vertex in set(edge):
                count+=1
            for element in set_of_elements:
                if set(edge).issubset(set(element)):
                    counter2+=1
            if counter2==2:
                # print("Edge {} is common for two elements".format(edge))
                count-=1
                modified_interior_edge_set.discard(edge)

        if count>=3:

            adj_vertices=sorted(list(vtx for edge in modified_interior_edge_set if vertex  in set(edge) for  vtx in edge if vtx!=vertex))
#            if len(adj_vertices)%2==1:
#                break
            counter=0
            # Checking if vertice are linked , if they are then aren't part of the same polygon
        #    keep_looking=True
           # while keep_looking:
            for index,_ in enumerate(adj_vertices.copy()):
                edge=tuple((adj_vertices[index],adj_vertices[(index+1)%len(adj_vertices)]))
                condition3=True
                # CHECK  CONDITION TO FIND OUT WHICH EDGE IS PAIRED WITH WHICH TO FORM A POLYGON


                # Connections could form elements that are not discovered
                if ((edge in set_of_interior_edges or tuple(reversed(edge)) in set_of_interior_edges )and
                ((vertex,edge[0]) in set_of_interior_edges or tuple(reversed((vertex,edge[0]))) in set_of_interior_edges) and
                ((vertex,edge[1]) in set_of_interior_edges or tuple(reversed((vertex,edge[1]))) in set_of_interior_edges) ):
                    # print("Found new element:",(vertex,edge[0],edge[1]))
                    # print("({},{}) and ({},{}) are part of the same element".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
  #                  adj_vertices.remove(edge[0])
    #                adj_vertices.remove(edge[1])
                    continue

                if(edge[0]<edge[1]):
                    #if edge[0]<=polygon.shape[0] and edge[1]<=polygon.shape[0]:
                        for i in range(edge[0]+1,edge[1]) :
                            if i< polygon.shape[0]:
                                if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                    or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                    if i==vertex:continue
                                    condition3=False
                            else:#interior point#
                                for i in range(edge[0]+1,edge[1]) :
                                    if ((vertex,i)  in set_of_interior_edges  or (i,vertex) in set_of_interior_edges ):
                                        if i==vertex:continue
                                        condition3=False


                else:
                    for i in range(edge[0]+1,len(polygon)-1) :

                        if i< polygon.shape[0]:

                            if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                if i==vertex:continue
                                condition3=False
                        else:

                            if ((vertex,i)  in set_of_interior_edges  or  (i,vertex) in set_of_interior_edges ):
                                if i==vertex:continue
                                condition3=False

                    for i in range(edge[1]) :
                        if i<polygon.shape[0]:
                            if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges
                                or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                                if i == vertex:continue
                                condition3=False
                        else:
                            if ((vertex,i)  in set_of_interior_edges  or  (i,vertex) in set_of_interior_edges ):
                                if i == vertex:continue
                                condition3=False
                d,s=is_traversable(vertex,set_of_elements,edge[0],edge[1])
                elements_around_vertex=vert2elem(vertex,set_of_elements)
                edges_around_vertex=edge2vert(vertex,polygon,set_of_interior_edges)
                edge_star=sort_edges_around_vertex(vertex,edges_around_vertex,polygon,points)
                # print("edges star of common vertex" ,vertex, "is : ",edge_star)
                # print("elements around {} are {}".format(vertex,elements_around_vertex))

                for edge1 in edge_star:
                    position_of_edge1=edge_star.index(edge1)

                    for edge2 in edge_star:
                        if edge2==edge1:
                            continue


                        position_of_edge2=edge_star.index(edge2)
                        if (abs(position_of_edge1-position_of_edge2)==1  or abs(position_of_edge1-position_of_edge2)==len(edge_star)+1) and not found_element_with_edges(edge1,edge2,elements_around_vertex):
                            if (edge1,edge2) not in pair_of_adjacent_edges and (edge2,edge1) not in pair_of_adjacent_edges:
                                 if (edge1  not in set_polygon_edges and edge1[::-1] not in set_polygon_edges) or (edge2 not in set_polygon_edges and edge2[::-1]  not in set_polygon_edges):
                                     # print(edge1,"is in  the same polygon with", edge2)

                                     pair_of_adjacent_edges.add(((edge2,edge1)))
                                     if edge2 not in modified_interior_edge_set and edge2[::-1]  not in modified_interior_edge_set:
                                         modified_interior_edge_set.add(edge2[::-1])
                                     if edge1 not in modified_interior_edge_set and edge1[::-1]  not in modified_interior_edge_set:
                                         modified_interior_edge_set.add(edge1[::-1])



#                print(edges_around_vertex)



                condition1=edge  in set_polygon_edges
                condition2=tuple(reversed(edge)) in set_polygon_edges
                nb_of_polygons=[]
                if not condition1 and not condition2  and condition3 :
                    counter+=1
#                        adj_vertices.remove(edge[0])
#                        adj_vertices.remove(edge[1])
                    # print("({},{}) and ({},{}) are part of the same polygon".format(edge[0],vertex,edge[1],vertex))
                  #  pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
#                        if len(adj_vertices)==0:
#                            keep_looking=False
#                        break
    #        keep_looking=False


            set_of_common_vertices.add(vertex)
            nb_of_polygons.append(counter)
            # print("vertex {} is adjacent to {} polygons".format(vertex,counter))
            # print("Set of adjacent edges to visit:",pair_of_adjacent_edges)
       # found_common_vertex=True

    # An edge could be part of more than one polygons. This means that the vertices of this edge
    # are already in the set of common vertices and the edges is inside the set of the of modi
    # fied interior edges
   # set_of_common_edges=set()
    #for vtx1 in  set_of_common_vertices:
     #   for vtx2 in set_of_common_vertices:
      #      pass





    # if the set found is les than 4 then now polygon is formed
    if len(set_of_open_vertices)<4:
        return []

    edges_to_visit=modified_interior_edge_set



    sub_polygon_list=[]
 #   pdb.set_trace()
    initial_edges_to_visit=copy.deepcopy(edges_to_visit)
    initial_pair_of_adjacent_edges=copy.deepcopy(pair_of_adjacent_edges)
    try:
        if set_of_common_vertices:
            for vtx in set_of_common_vertices:
                subpolygon=polygon_2_vtx(vtx,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon)
                if subpolygon is not None and subpolygon != 0:
                   sub_polygon_list.append(subpolygon)
                if subpolygon == 0:
                    continue

        # print(sub_polygon_list)
    except:
        # print("Failed")
        pass



    # Removing eges where one vertex is locked and the other is not:
    for edge in edges_to_visit.copy():
        if (edge[0] in set_of_open_vertices and edge[1] not in set_of_open_vertices) or (edge[1] in set_of_open_vertices and edge[0] not in set_of_open_vertices):
            edges_to_visit.discard(edge)
            # print("Removing",edge,"from edges to visit")
            # print("Edges to visit are now",edges_to_visit)


    while edges_to_visit:
        for vtx in set_of_open_vertices.copy():
            # print("Starting with vertex",vtx)
            subpolygon=polygon_2_vtx(vtx,set_of_elements,initial_edges_to_visit,edges_to_visit,set_of_common_vertices,initial_pair_of_adjacent_edges,pair_of_adjacent_edges,set_of_open_vertices,set_orphan_vertices,polygon)
            if subpolygon is not None and subpolygon != 0:
                   sub_polygon_list.append(subpolygon)
            if subpolygon == 0:
                    continue


#    for sub_polygon in sub_polygon_list:
#        if len(sub_polygon)>3:
#            print("found polygon",sub_polygon)
#        else:
#            print("found element",sub_polygon)
    return sub_polygon_list





def export_contour(filename,contour):
    path=os.path.join('contour_cases',filename+'.txt')
    file=open(path,'w')
    for i in contour:
        file.write(np.array2string(i)+"\n")
    file.close()


def read_contour(filename):
    path=os.path.join('contour_cases',filename+'.txt')
    contour=[]
    file=open(path,'r')
    for line in file:
        coord=np.fromstring(line.strip('[\n]'), dtype=float, sep=' ')
        contour.append(coord)
    file.close()
    return np.array(contour)
