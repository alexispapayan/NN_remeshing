import meshio
# import pyvista
import numpy as np
from functools import reduce
from itertools import combinations
from Triangulation import retriangulate
from Triangulation_with_points import retriangulate_with_interior
from Smoothing import smooth_interior_point, smooth_boundary_point, smooth_interface_point
from Neural_network import *
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from matplotlib import pyplot as plt

class ModifiableMesh(meshio.Mesh):
    def __init__(self, points, cells, point_data=None, cell_data=None, field_data=None, point_sets=None, cell_sets=None, gmsh_periodic=None, info=None, normal=None, target_edgelength_boundary=None, target_edgelength_interface=None):
        super().__init__(points, cells, point_data, cell_data, field_data, point_sets, cell_sets, gmsh_periodic, info)

        self.vertex_index = None
        self.line_index = None
        self.triangle_index = None
        self.tetra_index = None
        for c, cell in enumerate(self.cells):
            if cell.type == 'vertex':
                self.vertex_index = c
            elif cell.type == 'line':
                self.line_index = c
            elif cell.type == 'triangle':
                self.triangle_index = c
            elif cell.type == 'tetra':
                self.tetra_index = c

        if self.tetra_index is not None:
            self.dimension = 3
        elif self.triangle_index is not None:
            self.dimension = 2
        else:
            self.dimension = 1

        try:
            self.fixed_vertices = self.get_nodes().reshape(-1)
        except TypeError:
            self.fixed_vertices = np.array([], dtype=np.int)

        boundary = np.zeros(self.points.shape[0], dtype=np.bool)
        try:
            for object in self.get_lines():
                for vertex in object:
                    if vertex not in self.fixed_vertices:
                        boundary[vertex] = True
        except TypeError:
            pass

        interior = np.zeros(self.points.shape[0], dtype=np.bool)
        for vertex in range(len(self.points)):
            objects = self.get_neighbourhood(vertex)
            try:
                _, index, _ = self.get_contour(objects)
            except:
               continue
            interior[vertex] = vertex not in index

        self.interior_vertices = np.setdiff1d(np.nonzero(~boundary & interior)[0], self.fixed_vertices)
        self.boundary_vertices = np.setdiff1d(np.nonzero(boundary & ~interior)[0], self.fixed_vertices)
        self.interface_vertices = np.setdiff1d(np.nonzero(boundary & interior)[0], self.fixed_vertices)

        if normal is None:
            normal = [0,0,1]
        self.normal = np.array(normal)

        # if target_edgelength is None:
        #     elements = self.get_triangles()
        #     all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        #     edges = np.unique(all_edges, axis=0)
        #     length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
        #     target_edgelength = np.mean(length)
        # self.target_edgelength = target_edgelength
        if target_edgelength_boundary is None:
            edges = np.copy(self.get_lines())
            valid = np.concatenate([self.boundary_vertices, self.fixed_vertices])
            edges = edges[np.isin(self.get_lines()[:,0], valid) & np.isin(self.get_lines()[:,1], valid)]
            length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
            target_edgelength_boundary = np.mean(length)
        self.target_edgelength_boundary = target_edgelength_boundary

        if target_edgelength_interface is None:
            try:
                edges = np.copy(self.get_lines())
                valid = np.concatenate([self.interface_vertices, self.fixed_vertices])
                edges = edges[np.isin(self.get_lines()[:,0], valid) & np.isin(self.get_lines()[:,1], valid)]
                length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
                target_edgelength_interface = np.mean(length)
            except:
                target_edgelength_interface = target_edgelength_boundary
        self.target_edgelength_interface = target_edgelength_interface

        self.refinement_threshold = 1.6
        self.coarsen_threshold = 0.7

        # if interface_edges is None:
        #   self.interface_edges=[]
        self.generator = np.random.Generator(np.random.PCG64())

    def get_nodes(self):
        if self.vertex_index is None:
            return None
        else:
            return self.cells[self.vertex_index].data

    def set_nodes(self, nodes):
        self.cells[self.vertex_index] = self.cells[self.vertex_index]._replace(data=nodes)

    def get_lines(self):
        if self.line_index is None:
            return None
        else:
            return self.cells[self.line_index].data

    def set_lines(self, lines):
        self.cells[self.line_index] = self.cells[self.line_index]._replace(data=lines)

    def get_triangles(self):
        if self.triangle_index is None:
            return None
        else:
            return self.cells[self.triangle_index].data

    def set_triangles(self, triangles):
        self.cells[self.triangle_index] = self.cells[self.triangle_index]._replace(data=triangles)

    def get_tetras(self):
        if self.tetra_index is None:
            return None
        else:
            return self.cells[self.tetra_index].data

    def set_tetras(self, tetras):
        self.cells[self.tetra_index] = self.cells[self.tetra_index]._replace(data=tetra)

    def get_elements(self):
        if self.dimension == 3:
            return self.get_tetras()
        elif self.dimension == 2:
            return self.get_triangles()

    def set_elements(self, elements):
        if self.dimension == 3:
            self.set_tetras(elements)
        elif self.dimension == 2:
            self.set_triangles(elements)

    def quality(self):
        '''
        Calculates the quality of each element in the mesh.
        '''
        # dim = len(self.cells) - 1
        if self.dimension == 2:
            quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles())
        elif self.dimension == 3:
            quality = self.to_pyvista().quality
        else:
            raise DimensionError('Mesh must be a surface or volume mesh')
        return quality

    def plot_quality(self, plot_vertices=False):
        quality = self.quality()
        q = quality/2 + 0.5
        print('Mesh quality (min mean):', np.min(quality), np.mean(quality))

        # plt.figure(figsize=(8,6))
        cmap = plt.get_cmap('CMRmap')

        for t, triangle in enumerate(self.get_triangles()):
            plt.fill(self.points[triangle,0], self.points[triangle,1], fc=cmap(q[t]), alpha=0.8, ec='black')

        if plot_vertices:
            marker_size = 20
            plt.scatter(self.points[self.interior_vertices,0], self.points[self.interior_vertices,1], color='black', zorder=2, s=marker_size)
            plt.scatter(self.points[self.boundary_vertices,0], self.points[self.boundary_vertices,1], color='green', zorder=2, s=marker_size)
            plt.scatter(self.points[self.interface_vertices,0], self.points[self.interface_vertices,1], color='blue', zorder=2, s=marker_size)
            plt.scatter(self.points[self.fixed_vertices,0], self.points[self.fixed_vertices,1], color='red', zorder=2, s=marker_size)

        plt.axis('scaled')
        # plt.show()

    def smooth(self, maxiter=10):
        '''
        Smooth a mesh using neural networks. Iterates until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            try:
                partition = self.smoothing_partition()
            except:
                accepted=0
                break
            if len(partition) > 0:
                accepted = 0
                for v in partition:
                    if self.smooth_vertex(v):
                        accepted += 1
                print('Quality after {} smoothing iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def smooth_vertex(self, vertex):
        '''
        Smooth a vertex.
        '''
        objects = self.get_neighbourhood(vertex)
        try:
            contour, index, _ = self.get_contour(objects)
        except:
           accepted=False
           return accepted
        if len(contour) > 10:
            accepted = False
        else:
            quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
            q = np.min(quality)
            old_point = np.copy(self.points[vertex])
            contour = contour[:-1,:2] # 2D only !
            new_point = smooth_interior_point(contour) #, len(interior)
            # for p, point in enumerate(interior):
            self.points[vertex][:2] = new_point # 2D only !
            quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
            accepted = True
            if np.min(quality) <= q:
                 self.points[vertex] = old_point
                 accepted = False
        return accepted

    def smoothing_partition(self):
        '''
        Returns an ordered list of interior vertices to be smoothed.
        '''
        quality = self.quality()
        vertex_quality = np.zeros(self.interior_vertices.shape)
        for v, vertex in enumerate(self.interior_vertices):
            objects = objects_boundary_includes(self.get_triangles(), vertex)
            vertex_quality[v] = np.min(quality[objects])
        partition = self.interior_vertices[vertex_quality < 0.9]
        vertex_quality = vertex_quality[vertex_quality < 0.9]
        if len(vertex_quality) > 0:
            partition = partition[np.argsort(vertex_quality)]
        return partition

    def smooth_boundary(self, maxiter=10):
        '''
        Smooth the boundary vertices of a mesh using neural networks. Iterates until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            partition = self.boundary_partition()
            if len(partition) > 0:
                accepted = 0
                for vertex in partition:
                    if self.smooth_boundary_vertex(vertex):
                        accepted += 1
                print('Quality after {} boundary smoothing iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def smooth_boundary_vertex(self, vertex):
        '''
        Smooth a boundary vertex.
        '''
        objects = self.get_neighbourhood(vertex)
        if np.count_nonzero(objects) == 1:
            return False

        contour, index = self.get_open_contour(objects, vertex)
        if len(contour) < 3:
            return False
        if len(contour) > 6:
            return False
        contour = contour[:,:2] # 2D only !

        old_point = np.copy(self.points[vertex])
        quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        q = np.min(quality)

        try:
            spline, derivative = self.get_spline([index[0], vertex, index[-1]])
        except:
            self.points[vertex] = old_point
            accepted = False
            return accepted

        tangents = derivative(np.array([0,1]))
        tangents /= np.linalg.norm(tangents, axis=1)[:,None]

        new_point = smooth_boundary_point(contour, tangents)

        fun = lambda s: np.dot(new_point - spline(s), derivative(s))
        try:
            s0 = brentq(fun, 0, 1)
            new_point = spline(s0)
            self.points[vertex][:2] = new_point # 2D only !
            quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
            accepted = np.min(quality) > q
        except ValueError:
            accepted = False

        if not accepted:
             self.points[vertex] = old_point

        return accepted

    def boundary_partition(self):
        '''
        Returns an ordered list of boundary vertices to be smoothed.
        '''
        quality = self.quality()
        vertex_quality = np.zeros(self.boundary_vertices.shape)
        for v, vertex in enumerate(self.boundary_vertices):
            objects = objects_boundary_includes(self.get_triangles(), vertex)
            vertex_quality[v] = np.min(quality[objects])
        partition = self.boundary_vertices[vertex_quality < 0.9] #np.mean(quality)
        vertex_quality = vertex_quality[vertex_quality < 0.9]
        if len(vertex_quality) > 0:
            partition = partition[np.argsort(vertex_quality)]
        return partition

    def smooth_interface(self, maxiter=10):
        '''
        Smooth the interface vertices of a mesh using neural networks. Iterates until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            partition = self.interface_partition()
            if len(partition) > 0:
                accepted = 0
                for vertex in partition:
                    if self.smooth_interface_vertex(vertex):
                        accepted += 1
                print('Quality after {} interface smoothing iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def smooth_interface_vertex(self, vertex):
        '''
        Smooth a interface vertex.
        '''
        objects = self.get_neighbourhood(vertex)
        if np.count_nonzero(objects) == 1:
            return False
        old_point = np.copy(self.points[vertex])

        try:
            contour, index, _ = self.get_contour(objects)
        except:
            self.points[vertex] = old_point
            accepted = False
            return accepted

        contour = contour[:,:2] # 2D only !

        quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        q = np.min(quality)

        interface = np.intersect1d(index, np.union1d(self.interface_vertices, self.fixed_vertices))

        try:
            spline, derivative = self.get_spline([interface[0], vertex, interface[-1]])
        except:
            self.points[vertex] = old_point
            accepted=False
            return accepted

        tangents = derivative(np.array([0,1]))
        tangents /= np.linalg.norm(tangents, axis=1)[:,None]

        try:
            new_point = smooth_interface_point(contour, self.points[interface,:2], tangents)
        except:
            self.points[vertex] = old_point
            accepted=False
            return accepted

        fun = lambda s: np.dot(new_point - spline(s), derivative(s))
        try:
            s0 = brentq(fun, 0, 1)
            new_point = spline(s0)
            self.points[vertex][:2] = new_point # 2D only !
            quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
            accepted = np.min(quality) > q
        except ValueError:
            accepted = False

        if not accepted:
             self.points[vertex] = old_point

        return accepted

    def interface_partition(self):
        '''
        Returns an ordered list of interface vertices to be smoothed.
        '''
        quality = self.quality()
        vertex_quality = np.zeros(self.interface_vertices.shape)
        for v, vertex in enumerate(self.interface_vertices):
            objects = objects_boundary_includes(self.get_triangles(), vertex)
            vertex_quality[v] = np.min(quality[objects])
        partition = self.interface_vertices[vertex_quality < 0.9] #np.mean(quality)
        vertex_quality = vertex_quality[vertex_quality < 0.9]
        if len(vertex_quality) > 0:
            partition = partition[np.argsort(vertex_quality)]
        return partition

    def reconnect(self, maxiter=10):
        '''
        Reconnect the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            try:
                partition = self.connectivity_partition()
            except:
                accepted=0
                break


            if len(partition) > 0:
                groups = np.unique(partition)
                groups = groups[groups >= 0]
                keep_elements = np.ones(len(self.get_triangles()), dtype=np.bool)
                new_elements = []
                accepted = 0
                for i, g in enumerate(groups):
                    objects = partition == g
                    if np.count_nonzero(objects) > 1:
                        accept, new = self.reconnect_objects(objects)
                        if accept:
                            keep_elements = np.logical_and(~objects, keep_elements)
                            try:
                                new_elements = np.append(new_elements, new, axis=0)
                            except:
                                new_elements = new
                            accepted += 1

                elements = self.get_triangles()[keep_elements]

                if len(new_elements) > 0:
                    elements = np.append(elements, new_elements, axis=0)
                self.set_triangles(elements)
                print('Quality after {} reconnecting iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def reconnect_objects(self, objects):
        '''
        Reconnect the vertices inside a cavity given by objects using a neural network.
        '''
        quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        q = np.min(quality)
        contour, index, _ = self.get_contour(objects)

        if len(index) > 10:
            return False, objects

        contour = contour[:-1,:2] # 2D only !
        index = index[:-1]
        rolled = np.roll(contour, 1, axis=0)
        contour_direction = np.sign(np.sum(contour[:,1]*rolled[:,0] - contour[:,0]*rolled[:,1]))
        if contour_direction < 0:
            contour = contour[::-1]
            index = index[::-1]

        if len(index) == 3:
            return True, index[None,:]

        new = retriangulate(contour)

        new_elements = np.take(index, new)
        new_quality = np.apply_along_axis(self.triangle_quality, 1, new_elements)
        if np.min(new_quality) > q:
            accepted = True
        else:
            accepted = False

        return accepted, new_elements

    def connectivity_partition(self):
        '''
        Partition the mesh into cavities to be reconnected.
        '''
        quality = self.quality()
        bad = quality < 0.9

        partition = np.arange(len(quality))
        partition[~bad] = -1

        elements = self.get_triangles()[bad]
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        edges, counts = np.unique(all_edges, axis=0, return_counts=True)

        is_interior = counts > 1
        edges = edges[is_interior]

        if len(self.interface_vertices > 0):
            is_interface = objects_boundary_includes_some(edges, 2, *self.interface_vertices)
            edges = edges[~is_interface]

        edge_quality = np.apply_along_axis(self.edge_quality, 1, edges, quality)
        edges = edges[np.argsort(edge_quality)]

        not_accepted = []
        for edge in edges:
            triangle_pair = self.find_triangles_with_common_edge(edge)
            group = np.min(partition[triangle_pair])
            other_group = np.max(partition[triangle_pair])
            first = partition == group
            second = partition == other_group
            partition[np.logical_or(first, second)] = group
            new_polygon_objects = partition == group
            accept_group = True
            try:
                contour, _, interior = self.get_contour(new_polygon_objects)
                contour = contour[:-1,:]
            except:
                accept_group = False
                reason = 'Not simply connected'

            if accept_group and len(interior) > 0: # no interior vertices
                accept_group = False
                reason = 'Interior vertex'
            elif accept_group and len(contour) > 9: # at most 9 edges
                accept_group = False
                reason = 'Contour too large'
            elif accept_group and not is_convex(contour, tol=3e-1): # (nearly) convex
                accept_group = False
                reason = 'Not convex'

            if not accept_group:
                partition[second] = other_group
                # t0, t1 = np.nonzero(triangle_pair)[0]
                # not_accepted.append((t0, t1, reason))

        # for t0, t1, reason in not_accepted:
        #     print(partition[t0], partition[t1], reason)

        return partition

    def refine(self, maxiter=10):
        '''
        Refine the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            partition, new_points = self.refinement_partition()
            groups = new_points.keys()
            if len(groups) > 1:
                keep_elements = np.ones(len(self.get_triangles()), dtype=np.bool)
                new_elements = []
                accepted = 0
                for g in groups:
                    if g >= 0:
                        objects = partition == g
                        if np.count_nonzero(objects) > 1:
                            accept, new, interior = self.refine_objects(objects, new_points[g])
                            if len(interior) > 0:
                                raise ValueError('Points to be deleted during refinement')
                            if accept:
                                keep_elements = keep_elements & ~objects
                                try:
                                    new_elements = np.append(new_elements, new, axis=0)
                                except:
                                    new_elements = new
                                accepted += 1

                if len(new_elements) > 0:
                    elements = self.get_triangles()[keep_elements]
                    elements = np.append(elements, new_elements, axis=0)
                    self.set_triangles(elements)
                    print('Quality after {} refinement iterations: {}'.format(iter, np.min(self.quality())))
                    iter += 1
                else:
                    accepted = 0
            else:
                accepted = 0

    def refine_boundary(self, maxiter=10):
        '''
        Refine the boundary of the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            edges = np.copy(self.get_lines())
            valid = np.concatenate([self.boundary_vertices, self.fixed_vertices])
            edges = edges[np.isin(self.get_lines()[:,0], valid) & np.isin(self.get_lines()[:,1], valid)]

            length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
            long = length > self.target_edgelength_boundary * self.refinement_threshold
            edges = edges[long]
            edges = edges[np.argsort(-length[long])]

            keep_elements = np.ones(len(self.get_triangles()), dtype=np.bool)
            new_elements = []
            for edge in edges:
                accept, new, old = self.refine_interface_or_boundary_objects(edge)

                if accept:
                    new_index = len(self.points)-1
                    self.boundary_vertices = np.append(self.boundary_vertices, [new_index], axis=0)
                    edge_index = ((self.get_lines()[:,0] == edge[0]) & (self.get_lines()[:,1] == edge[1])) | ((self.get_lines()[:,1] == edge[0]) & (self.get_lines()[:,0] == edge[1]))
                    new_lines = self.get_lines()[~edge_index]
                    new_lines = np.append(new_lines, np.array([[edge[0], new_index], [new_index, edge[1]]], dtype=np.int), axis=0)
                    self.set_lines(new_lines)

                    keep_elements = np.logical_and(~old, keep_elements)
                    if len(new_elements) > 0:
                        new_elements = np.append(new_elements, new, axis=0)
                    else:
                        new_elements = new
                    accepted += 1

            if len(new_elements) > 0:
                elements = self.get_triangles()[keep_elements]
                elements = np.append(elements, new_elements, axis=0)
                self.set_triangles(elements)
                print('Quality after {} boundary refinement iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def refine_interface(self, maxiter=10):
        '''
        Refine interfaces in the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            edges = np.copy(self.get_lines())
            edges = edges[np.isin(self.get_lines()[:,0], self.interface_vertices) | np.isin(self.get_lines()[:,1], self.interface_vertices)]

            length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
            long = length > self.target_edgelength_interface * self.refinement_threshold
            edges = edges[long]
            edges = edges[np.argsort(-length[long])]

            keep_elements = np.ones(len(self.get_triangles()), dtype=np.bool)
            new_elements = []
            for edge in edges:
                accept, new, old = self.refine_interface_or_boundary_objects(edge)

                if accept:
                    new_index = len(self.points)-1
                    self.interface_vertices = np.append(self.interface_vertices, [new_index], axis=0)
                    edge_index = ((self.get_lines()[:,0] == edge[0]) & (self.get_lines()[:,1] == edge[1])) | ((self.get_lines()[:,1] == edge[0]) & (self.get_lines()[:,0] == edge[1]))
                    new_lines = self.get_lines()[~edge_index]
                    new_lines = np.append(new_lines, np.array([[edge[0], new_index], [new_index, edge[1]]], dtype=np.int), axis=0)
                    self.set_lines(new_lines)

                    keep_elements = np.logical_and(~old, keep_elements)
                    if len(new_elements) > 0:
                        new_elements = np.append(new_elements, new, axis=0)
                    else:
                        new_elements = new
                    accepted += 1

            if len(new_elements) > 0:
                elements = self.get_triangles()[keep_elements]
                elements = np.append(elements, new_elements, axis=0)
                self.set_triangles(elements)
                print('Quality after {} interface refinement iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def refine_objects(self, objects, new_points):
        '''
        Refine a cavity given by objects using a neural network.
        '''
        new_index = np.arange(len(self.points), len(self.points) + len(new_points))
        self.points = np.append(self.points, new_points, axis=0)
        self.interior_vertices = np.append(self.interior_vertices, new_index)

        quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        q = np.min(quality)
        try:
            contour, index, interior = self.get_contour(objects)
        except ValueError:
            print('Invalid contour')
            return False, None, None

        contour = contour[:-1,:2] # 2D only !
        index = index[:-1]
        rolled = np.roll(contour, 1, axis=0)
        contour_direction = np.sign(np.sum(contour[:,1]*rolled[:,0] - contour[:,0]*rolled[:,1]))
        if contour_direction < 0:
            contour = contour[::-1]
            index = index[::-1]
        index = np.append(index, new_index)

        new = retriangulate_with_interior(contour, *new_points[:,:2])
        new_elements = np.take(index, new)
        # print(new_elements)

        # plt.clf()
        # for e in new_elements:
        #     plt.fill(self.points[e][:,0], self.points[e][:,1], ec='black', fc='white', alpha=0.8)
        # plt.scatter(contour[:,0], contour[:,1])
        # plt.scatter(new_points[:,0], new_points[:,1])
        # plt.show()

        new_quality = np.apply_along_axis(self.triangle_quality, 1, new_elements)
        if np.min(new_quality) > q:
            accepted = True
        else:
            accepted = False
            self.points = self.points[:-len(new_points)]
            self.interior_vertices = self.interior_vertices[:-len(new_points)]

        return accepted, new_elements, interior

    def refine_interface_or_boundary_objects(self, edge):
        new_point = np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])
        objects = objects_boundary_includes_some(self.get_triangles(), 2, *edge)
        self.points = np.append(self.points, new_point, axis=0)

        print(len(self.get_triangles()[objects]))
        quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        q = np.min(quality)
        new_elements = []
        for triangle in self.get_triangles()[objects]:
            while not np.all(np.isin(triangle[:2], edge)):
                triangle = np.roll(triangle, 1)
            index = np.concatenate([triangle[:1], [len(self.points)-1], triangle[1:]])
            # contour = self.points[index][:,:2]
            # new = retriangulate(contour)
            new = np.array([[1,2,3], [0,1,3]], dtype=np.int)
            if len(new_elements) == 0:
                new_elements = np.take(index, new)
            else:
                new_elements = np.append(new_elements, np.take(index, new), axis=0)

        new_quality = np.apply_along_axis(self.triangle_quality, 1, new_elements)
        if np.min(new_quality) > q:
            accept = True
        else:
            accept = False
            self.points = self.points[:-1]

        return accept, new_elements, objects

    def refinement_partition(self):
        '''
        Partition the mesh into cavities to be refined.
        '''
        partition = np.arange(len(self.get_triangles()))

        elements = self.get_triangles()
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        edges, counts = np.unique(all_edges, axis=0, return_counts=True)

        is_interior = counts > 1
        edges = edges[is_interior]

        if len(self.interface_vertices > 0):
            is_interface = objects_boundary_includes_some(edges, 2, *self.interface_vertices)
            edges = edges[~is_interface]

        # edges = np.array([i for i in edges if i not in self.interface_edges])
        length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
        long = length > self.target_edgelengths(edges) * self.refinement_threshold
        edges = edges[long]
        edges = edges[np.argsort(-length[long])]

        new_points = {}
        not_accepted = []
        for edge in edges:
            triangle_pair = self.find_triangles_with_common_edge(edge)
            group = np.min(partition[triangle_pair])
            other_group = np.max(partition[triangle_pair])

            first = partition == group
            second = partition == other_group
            partition[np.logical_or(first, second)] = group

            accept_group = True
            if group not in new_points and other_group not in new_points:
                new_points[group] = np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])
            else:
                new_polygon_objects = partition == group
                contour, _, interior = self.get_contour(new_polygon_objects)
                nodes = [new_points[g] for g in [group, other_group] if g in new_points]
                new = sum([len(n) for n in nodes])
                if len(contour) > 8 or len(interior) + new > len(contour) - 4:
                    accept_group = False
                else:
                    new_points[group] = np.concatenate(nodes + [np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])], axis=0)
                    if other_group in new_points:
                        del new_points[other_group]

            if not accept_group:
                partition[second] = other_group

        partition[np.isin(partition, list(new_points.keys()), invert=True)] = -1

        return partition, new_points

    def coarsen(self, maxiter=10):
        '''
        Coarsen the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            # self.coarsen_near_boundary_or_interface()
            partition, new_points = self.coarsen_partition()
            groups = new_points.keys()
            if len(groups) > 1:
                keep_elements = np.ones(len(self.get_triangles()), dtype=np.bool)
                new_elements = []
                accepted = 0
                for g in groups:
                    if g >= 0:
                        objects = partition == g
                        if np.count_nonzero(objects) > 1:
                            accept, new, remove = self.refine_objects(objects, new_points[g]['coords'])
                            if accept:
                                keep_elements = np.logical_and(~objects, keep_elements)
                                try:
                                    new_elements = np.append(new_elements, new, axis=0)
                                except:
                                    new_elements = new
                                # self.points = np.delete(self.points, remove, axis=0)
                                remains = np.isin(self.interior_vertices, remove, invert=True)
                                self.interior_vertices = self.interior_vertices[remains]
                                for old in remove:
                                    if old in np.concatenate([self.fixed_vertices, self.boundary_vertices, self.interface_vertices]):
                                        replacement = np.nonzero(np.any(new_points[g]['old'] == old, axis=1))[0] - new_points[g]['old'].shape[0] + len(self.points)
                                        self.points = np.delete(self.points, remove, axis=0)
                                        new_elements[new_elements == replacement] = old
                                    else:
                                        self.points = np.delete(self.points, old, axis=0)
                                        remove[remove > old] -= 1
                                        new_elements[new_elements > old] -= 1
                                        for G in groups:
                                            new_points['old'][new_points['old'] > old] -= 1
                                        self.interior_vertices[self.interior_vertices > old] -= 1
                                        self.interface_vertices[self.interface_vertices > old] -= 1
                                        self.boundary_vertices[self.boundary_vertices > old] -= 1
                                        self.fixed_vertices[self.fixed_vertices > old] -= 1
                                        for cell in self.cells:
                                            cell.data[cell.data > old] -= 1
                                accepted += 1

                if len(new_elements) > 0:
                    elements = self.get_triangles()[keep_elements]
                    elements = np.append(elements, new_elements, axis=0)
                    self.set_triangles(elements)
                    print('Quality after {} coarsening iterations: {}'.format(iter, np.min(self.quality())))
                    iter += 1
                else:
                    accepted = 0
            else:
                accepted = 0


    def coarsen_boundary(self, maxiter=10):
        '''
        Coarsen boundaries in the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            edges = np.copy(self.get_lines())
            edges = edges[np.isin(self.get_lines()[:,0], self.boundary_vertices) & np.isin(self.get_lines()[:,1], self.boundary_vertices)]

            length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
            short = length < self.target_edgelength_boundary * self.coarsen_threshold
            edges = edges[short]
            edges = edges[np.argsort(length[short])]

            e = 0
            while e < len(edges)-1:
                edge = edges[e]
                unique = np.sum(np.concatenate([edges == edge[0], edges == edge[1]], axis=1), axis=1) != 1
                unique[:e] = True
                edges = edges[unique]
                e += 1

            changed = False
            for edge in edges:
                accept, new, objects = self.coarsen_boundary_objects(edge)

                if accept:
                    self.points = np.delete(self.points, edge, axis=0)
                    elements = np.append(self.get_triangles()[~objects], new, axis=0)
                    for old in edge:
                        edges[edges > old] -= 1
                        elements[elements > old] -= 1
                        self.interior_vertices[self.interior_vertices > old] -= 1
                        self.interface_vertices[self.interface_vertices > old] -= 1
                        self.boundary_vertices[self.boundary_vertices > old] -= 1
                        self.fixed_vertices[self.fixed_vertices > old] -= 1
                        for cell in self.cells:
                            cell.data[cell.data > old] -= 1
                    self.set_triangles(elements)
                    accepted += 1
                    changed = True

            if changed:
                print('Quality after {} boundary coarsening iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def coarsen_boundary_objects(self, edge):
        new_point = np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])
        objects = objects_boundary_includes_some(self.get_triangles(), 1, *edge)
        self.points = np.append(self.points, new_point, axis=0)
        new_index = len(self.points)-1

        # quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        # q = np.min(quality)

        contour, index, _ = self.get_contour(objects)

        contour = contour[:-1,:2]
        index = index[:-1]
        rolled = np.roll(contour, 1, axis=0)
        contour_direction = np.sign(np.sum(contour[:,1]*rolled[:,0] - contour[:,0]*rolled[:,1]))
        if contour_direction < 0:
            contour = contour[::-1]
            index = index[::-1]

        i = 0
        while ~np.all(np.isin(index[-2:], edge)):
            if i == len(index):
                print(edge, self.get_triangles()[objects], index)
                raise ValueError('Invalid contour')
            index = np.roll(index, 1, axis=0)
            contour = np.roll(contour, 1, axis=0)
            i += 1
        contour = np.append(contour[:-2], new_point[:,:2], axis=0)
        index = np.append(index[:-2], [new_index], axis=0)

        new = retriangulate(contour)
        new_elements = np.take(index, new)

        if new_elements.ndim == 1:
            new_elements = np.array([new_elements])
        new_quality = np.apply_along_axis(self.triangle_quality, 1, new_elements)
        if np.min(new_quality) > 0:
            accept = True
            self.boundary_vertices = self.boundary_vertices[np.isin(self.boundary_vertices, edge, invert=True)]
            self.boundary_vertices = np.append(self.boundary_vertices, [new_index], axis=0)
            edge_index = np.any(np.isin(self.get_lines(), edge), axis=1)
            new_lines = self.get_lines()[~edge_index]
            new_lines = np.append(new_lines, np.array([[index[-2], new_index], [new_index, index[0]]], dtype=np.int), axis=0)
            self.set_lines(new_lines)
        else:
            accept = False
            self.points = self.points[:-1]

        return accept, new_elements, objects

    def coarsen_interface(self, maxiter=10):
        '''
        Coarsen interfaces in the mesh using neural networks. Iterated until no further improvement is made.
        '''
        accepted = 1
        iter = 1
        while accepted > 0 and iter <= maxiter:
            edges = np.copy(self.get_lines())
            edges = edges[np.isin(self.get_lines()[:,0], self.interface_vertices) & np.isin(self.get_lines()[:,1], self.interface_vertices)]

            length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
            short = length < self.target_edgelength_interface * self.coarsen_threshold
            edges = edges[short]
            edges = edges[np.argsort(length[short])]

            e = 0
            while e < len(edges)-1:
                edge = edges[e]
                unique = np.sum(np.concatenate([edges == edge[0], edges == edge[1]], axis=1), axis=1) != 1
                unique[:e] = True
                edges = edges[unique]
                e += 1

            changed = False
            for edge in edges:
                accept, new, objects = self.coarsen_interface_objects(edge)

                if accept:
                    print(self.interface_vertices)
                    print(self.get_lines())
                    self.points = np.delete(self.points, edge, axis=0)
                    elements = np.append(self.get_triangles()[~objects], new, axis=0)
                    for old in edge:
                        edges[edges > old] -= 1
                        elements[elements > old] -= 1
                        self.interior_vertices[self.interior_vertices > old] -= 1
                        self.interface_vertices[self.interface_vertices > old] -= 1
                        self.boundary_vertices[self.boundary_vertices > old] -= 1
                        self.fixed_vertices[self.fixed_vertices > old] -= 1
                        for cell in self.cells:
                            cell.data[cell.data > old] -= 1
                    self.set_triangles(elements)
                    accepted += 1
                    changed = True

            if changed:
                print('Quality after {} interface coarsening iterations: {}'.format(iter, np.min(self.quality())))
                iter += 1
            else:
                accepted = 0

    def coarsen_interface_objects(self, edge):
        new_point = np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])
        objects = objects_boundary_includes_some(self.get_triangles(), 1, *edge)
        self.points = np.append(self.points, new_point, axis=0)
        new_index = len(self.points)-1

        # quality = np.apply_along_axis(self.triangle_quality, 1, self.get_triangles()[objects])
        # q = np.min(quality)

        contour, index, _ = self.get_contour(objects)
        contour = contour[:-1,:2]
        index = index[:-1]
        rolled = np.roll(contour, 1, axis=0)
        contour_direction = np.sign(np.sum(contour[:,1]*rolled[:,0] - contour[:,0]*rolled[:,1]))
        if contour_direction < 0:
            contour = contour[::-1]
            index = index[::-1]

        i = np.intersect1d(index, self.interface_vertices)
        iv = np.sort(np.nonzero(np.isin(index, i))[0])
        i0 = np.append(np.arange(iv[0], iv[1]+1), len(index))
        i1 = np.concatenate([np.arange(iv[1], len(index)), np.arange(iv[0]+1), [len(index)]])

        contour = np.append(contour, new_point[:,:2], axis=0)
        index = np.append(index, new_index)

        new = retriangulate(contour[i0])
        new_elements = np.take(index[i0], new)
        new = retriangulate(contour[i1])
        new_elements = np.append(new_elements, np.take(index[i1], new), axis=0)

        new_quality = np.apply_along_axis(self.triangle_quality, 1, new_elements)
        if np.min(new_quality) > 0:
            accept = True
            self.interface_vertices = self.interface_vertices[np.isin(self.interface_vertices, edge, invert=True)]
            self.interface_vertices = np.append(self.interface_vertices, [new_index], axis=0)
            edge_index = np.any(np.isin(self.get_lines(), edge), axis=1)
            new_lines = self.get_lines()[~edge_index]
            new_lines = np.append(new_lines, np.array([[index[iv[0]], new_index], [new_index, index[iv[1]]]], dtype=np.int), axis=0)
            self.set_lines(new_lines)
        else:
            accept = False
            self.points = self.points[:-1]

        return accept, new_elements, objects

    def coarsen_partition(self):
        '''
        Partition the mesh into cavities to be coarsened.
        '''
        partition = np.arange(len(self.get_triangles()))

        elements = self.get_triangles()
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        edges = np.unique(all_edges, axis=0)

        boundary_or_interface = np.concatenate([self.fixed_vertices, self.boundary_vertices, self.interface_vertices])
        includes_boundary = objects_boundary_includes_some(edges, 2, *boundary_or_interface)
        edges = edges[~includes_boundary]

        # edges = np.array([i for i in edges if  i not in self.interface_edges])
        length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
        short = length < self.target_edgelengths(edges) * self.coarsen_threshold
        edges = edges[short]
        edges = edges[np.argsort(length[short])]

        new_points = {}
        replacements = {}
        not_accepted = []
        for edge in edges:
            if edge[0] in boundary_or_interface:
                potential = objects_boundary_includes_some(self.get_triangles(), 1, edge[1])
            elif edge[1] in boundary_or_interface:
                potential = objects_boundary_includes_some(self.get_triangles(), 1, edge[0])
            else:
                potential = objects_boundary_includes_some(self.get_triangles(), 1, *edge)
            group = np.min(partition[potential])
            all_groups = np.unique(partition[potential])
            selected = np.isin(partition, all_groups)

            if np.all(np.isin(all_groups, list(new_points.keys()), invert=True)):
                partition[selected] = group
                if edge[0] in boundary_or_interface:
                    new_points[group] = dict(coords=self.points[edge[0]][None,:], old=edge[None,:])
                elif edge[1] in boundary_or_interface:
                    new_points[group] = dict(coords=self.points[edge[1]][None,:], old=edge[None,:])
                else:
                    new_points[group] = dict(coords=np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2]), old=edge[None,:])
            else:
                try:
                    contour, _, interior = self.get_contour(selected)
                except:
                    continue
                interior = interior[np.isin(interior, edge, invert=True)]
                nodes = [new_points[g]['coords'] for g in all_groups if g in new_points]
                old_edges = [new_points[g]['old'] for g in all_groups if g in new_points]
                new = sum([len(n) for n in nodes])
                if len(contour) < 9 and len(interior) + new < len(contour) - 3:
                    partition[selected] = group
                    if edge[0] in boundary_or_interface:
                        new_point = self.points[edge[0]][None,:]
                    elif edge[1] in boundary_or_interface:
                        new_point = self.points[edge[1]][None,:]
                    else:
                        new_point = np.array([(self.points[edge[0]] + self.points[edge[1]]) / 2])
                    new_points[group] = dict(coords=np.concatenate(nodes + [new_point], axis=0), old=np.concatenate(old_edges + [edge[None,:]], axis=0))
                    for g in all_groups:
                        if g != group and g in new_points:
                            del new_points[g]

        partition[np.isin(partition, list(new_points.keys()), invert=True)] = -1

        return partition, new_points

    def coarsen_near_boundary_or_interface(self):
        elements = self.get_triangles()
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        edges = np.unique(all_edges, axis=0)

        boundary_or_interface = np.concatenate([self.fixed_vertices, self.boundary_vertices, self.interface_vertices])
        # near_boundary_or_interface = objects_boundary_includes_some(edges, 1, *boundary_or_interface)
        # on_boundary_or_interface = (np.isin(edges[:,0], self.get_lines()[:,0]) & np.isin(edges[:,1], self.get_lines()[:,1])) | (np.isin(edges[:,0], self.get_lines()[:,1]) & np.isin(edges[:,1], self.get_lines()[:,0]))
        # on_boundary_or_interface = objects_boundary_includes_some(edges, 2, *boundary_or_interface)

        valid = np.sum(np.isin(edges, boundary_or_interface), axis=1) == 1
        edges = edges[valid]

        length = np.linalg.norm(self.points[edges[:,0]] - self.points[edges[:,1]], axis=1)
        short = length < self.target_edgelengths(edges) * self.coarsen_threshold
        edges = edges[short]
        edges = edges[np.argsort(length[short])]

        e = 0
        while e < len(edges)-1:
            is_boundary_or_interface = np.isin(edges[e], boundary_or_interface)
            remove = edges[e][is_boundary_or_interface]
            unique = np.all(edges != remove, axis=1)
            unique[:e] = True
            edges = edges[unique]
            e += 1

        for edge in edges:
            is_boundary_or_interface = np.isin(edge, boundary_or_interface)
            if np.count_nonzero(is_boundary_or_interface) != 1:
                raise ValueError('Cannot coarsen this edge:', edge)
            contract_to_vertex = edge[is_boundary_or_interface]
            remove_vertex = edge[~is_boundary_or_interface]
            # collapsed = objects_boundary_includes_some(self.get_elements(), 2, *edge)

            # if np.count_nonzero(collapsed) > 0:
            affected = objects_boundary_includes(self.get_elements(), remove_vertex)
            if np.count_nonzero(affected) < 2:
                raise ValueError('Orphaned edge')

            # old_point = np.copy(self.points[remove_vertex])
            # self.points[remove_vertex] = self.points[contract_to_vertex]

            # contour, index, interior = self.get_contour(affected)
            # if len(interior) != 1:
            #     print(self.get_triangles()[affected])
            #     print(index[:-1])
            #     print(remove_vertex, interior)
            #     print(remove_vertex in self.boundary_vertices)
            #     raise ValueError('Vertex to be removed not in interior')
            accept, new = self.reconnect_objects(affected)
            # objects = self.get_elements()[np.logical_and(affected, ~collapsed)]

            if accept:
                self.points = np.delete(self.points, remove_vertex, axis=0)
                self.set_elements(np.append(self.get_elements()[~affected], new, axis=0))

                if np.count_nonzero(objects_boundary_includes(self.get_elements(), remove_vertex)) > 0:
                    print(remove_vertex, new)
                    raise ValueError('Elements not removed')

                edges[edges > remove_vertex] -= 1
                remains = self.interior_vertices != remove_vertex
                if np.count_nonzero(~remains) > 1:
                    raise ValueError('Repeated interior vertex')
                self.interior_vertices = self.interior_vertices[remains]
                self.interior_vertices[self.interior_vertices > remove_vertex] -= 1
                self.interface_vertices[self.interface_vertices > remove_vertex] -= 1
                self.boundary_vertices[self.boundary_vertices > remove_vertex] -= 1
                self.fixed_vertices[self.fixed_vertices > remove_vertex] -= 1
                boundary_or_interface = np.concatenate([self.fixed_vertices, self.boundary_vertices, self.interface_vertices])

                for cell in self.cells:
                    # cell.data[cell.data == remove_vertex] = contract_to_vertex
                    cell.data[cell.data > remove_vertex] -= 1
            # else:
            #     self.points = np.insert(self.points, remove_vertex, old_point, axis=0)

    def target_edgelengths(self, edges):
        targets = np.zeros(len(edges))
        for e, edge in enumerate(edges):
            triangle_pair = self.find_triangles_with_common_edge(edge)
            lengths = np.zeros(6)
            for t, triangle in enumerate(self.get_elements()[triangle_pair]):
                if t > 1:
                    print(edge)
                    print(self.get_elements()[triangle_pair])
                    raise ValueError('Edge belongs to more than 2 triangles')
                lengths[3*t:3*(t+1)] = np.linalg.norm(self.points[triangle] - self.points[np.roll(triangle, 1)], axis=1)
            targets[e] = (np.sum(lengths) - 2*np.linalg.norm(self.points[edge[0]] - self.points[edge[1]])) / 4
        min_length = min(self.target_edgelength_boundary, self.target_edgelength_interface)
        max_length = max(self.target_edgelength_boundary, self.target_edgelength_interface)
        targets = np.clip(targets, min_length, max_length)
        # if len(self.interface_vertices) > 0:
        #     midpoints = (self.points[edges[:,0]] + self.points[edges[:,1]]) / 2
        #     from_boundary = midpoints[:,None,:] - self.points[self.boundary_vertices][None,:,:]
        #     min_from_boundary = np.min(np.linalg.norm(from_boundary, axis=2), axis=1)
        #     from_interface = midpoints[:,None,:] - self.points[self.interface_vertices][None,:,:]
        #     min_from_interface = np.min(np.linalg.norm(from_interface, axis=2), axis=1)
        #     proportion = min_from_boundary / (min_from_boundary + min_from_interface)
        #     targets = self.target_edgelength_boundary * (1 - proportion) + self.target_edgelength_interface * proportion
        # else:
        #     targets = self.target_edgelength_boundary * np.ones(len(edges))
        return targets

    def translate_vertex(self, vertex, translation, check_injectivity=True, project_inwards=False):
        '''
        Translate vertex by translation. Optionally check whether elements remain injective after the translation (default True). Works for surface and volume meshes.
        '''
        new_vertex = self.points[vertex] + translation
        if check_injectivity:
            objects = self.get_neighbourhood(vertex)
            for o in np.nonzero(objects)[0]:
                boundary = self.points[self.get_elements()[o][self.get_elements()[o] != vertex]]
                if self.dimension == 2:
                    line = boundary[1] - boundary[0]
                    direct = self.points[vertex] - boundary[0]
                    normal = direct - np.dot(direct, line)/np.dot(line, line) * line
                elif self.dimension == 3:
                    normal = np.cross(boundary[1] - boundary[0], boundary[2] - boundary[0])
                else:
                    raise DimensionError('Mesh must be a surface volume mesh')
                old = np.dot(self.points[vertex] - boundary[0], normal)
                new = np.dot(new_vertex - boundary[0], normal)
                old_side = np.sign(old)
                new_side = np.sign(new)
                if old_side != new_side:
                    if project_inwards:
                        new_vertex += old_side * (np.abs(new) + 1e-6) * normal / np.dot(normal, normal)
                    else:
                        raise ValueError('Mesh is no longer injective')
        self.points[vertex] = new_vertex

    def translate_vertex_towards_center(self, vertex,center , epsilon , check_injectivity=True, project_inwards=False):


            new_vertex =(1-epsilon) * self.points[vertex] + epsilon * center
            if check_injectivity:
                objects = self.get_neighbourhood(vertex)
                for o in np.nonzero(objects)[0]:
                    boundary = self.points[self.get_elements()[o][self.get_elements()[o] != vertex]]
                    if self.dimension == 2:
                        line = boundary[1] - boundary[0]
                        direct = self.points[vertex] - boundary[0]
                        normal = direct - np.dot(direct, line)/np.dot(line, line) * line
                    elif self.dimension == 3:
                        normal = np.cross(boundary[1] - boundary[0], boundary[2] - boundary[0])
                    else:
                        raise DimensionError('Mesh must be a surface volume mesh')
                    old = np.dot(self.points[vertex] - boundary[0], normal)
                    new = np.dot(new_vertex - boundary[0], normal)
                    old_side = np.sign(old)
                    new_side = np.sign(new)
                    if old_side != new_side:
                        if project_inwards:
                            new_vertex += old_side * (np.abs(new) + 1e-6) * normal / np.dot(normal, normal)
                        else:
                            raise ValueError('Mesh is no longer injective')
            self.points[vertex] = new_vertex

    def contract_edge(self, contract_to_vertex, remove_vertex):
        '''
        Contract an edge. The vertex that remains is contract_to_vertex. Works for surface and volume meshes.

        Raises ValueError if no there is no edge between vertex1 and vertex2.
        '''
        collapsed = objects_boundary_includes_some(self.get_elements(), 2, contract_to_vertex, remove_vertex)
        edge_exists = np.any(collapsed)
        if not edge_exists:
            raise ValueError('There is no edge bewteen vertices {} and {}'.format(contract_to_vertex, remove_vertex))
        self.points = np.delete(self.points, remove_vertex, axis=0) # remove vertex
        self.set_elements(self.get_elements()[~collapsed])
        for c in range(1, len(self.cells)-1): # remove collapsed elements
            collapsed = objects_boundary_includes_some(self.cells[c].data, 2, contract_to_vertex, remove_vertex)
            self.cells[c] = self.cells[c]._replace(data=self.cells[c].data[~collapsed])
        for cell in self.cells:
            cell.data[cell.data == remove_vertex] = contract_to_vertex # relabel end to start
            cell.data[cell.data > remove_vertex] -= 1 # account for removed vertex

    def swap_edge(self, vertex1, vertex2, check_angle=True):
        '''
        Removes the edge from vertex1 to vertex2 and adds an edge between the other corners of two triangles that used to share the edge from vertex1 to vertex2. Works for surface meshes only.

        Raises ValueError if no there is no edge between vertex1 and vertex2 and optionally if the angle along the new edge is too small (default True).
        Raises DimensionError if the mesh is not a surface mesh.
        '''
        if self.dimension != 2:
            raise DimensionError('Mesh must be a surface mesh')

        triangle_pair = self.find_triangles_with_common_edge([vertex1, vertex2])
        triangle0 = self.get_triangles()[triangle_pair][0]
        triangle1 = self.get_triangles()[triangle_pair][1]
        while triangle0[0] != vertex1:
            triangle0 = np.roll(triangle0, 1)
        if triangle0[1] == vertex2:
            other_vertex2 = triangle0[2]
            while triangle1[0] != vertex1:
                triangle1 = np.roll(triangle1, 1)
            other_vertex1 = triangle1[1]
        else:
            other_vertex1 = triangle0[1]
            while triangle1[0] != vertex1:
                triangle1 = np.roll(triangle1, 1)
            other_vertex2 = triangle1[2]

        try:
            triangle_pair = self.find_triangles_with_common_edge([other_vertex1, other_vertex2])
            new_edge_exists = True
        except ValueError:
            new_edge_exists = False
        if new_edge_exists:
            raise ValueError('Edge {} - {} already exists!'.format(other_vertex1, other_vertex2))

        if check_angle:
            other_span = np.reshape(self.points[other_vertex2] - self.points[other_vertex1], -1)
            side1 = np.reshape(self.points[vertex1] - self.points[other_vertex1], -1)
            side2 = np.reshape(self.points[vertex2] - self.points[other_vertex1], -1)
            projected1 = side1 - np.dot(side1, other_span) / np.dot(other_span, other_span) * other_span
            projected2 = side2 - np.dot(side2, other_span) / np.dot(other_span, other_span) * other_span
            if np.allclose(projected1, 0) or np.allclose(projected2, 0):
                raise ValueError('Angle between new triangle pair is too small')
            projected1 /= np.linalg.norm(projected1)
            projected2 /= np.linalg.norm(projected2)
            if np.dot(projected1, projected2) > 1 - 1e-2:
                raise ValueError('Angle between new triangle pair is too small')

        t0, t1 = np.nonzero(triangle_pair)[0]
        self.get_triangles()[t0] = np.array([vertex1, other_vertex1, other_vertex2], dtype=np.int)
        self.get_triangles()[t1] = np.array([vertex2, other_vertex2, other_vertex1], dtype=np.int)

    def split_object(self, object):
        '''
        Adds a new vertex by splitting object into 3 objects. The object must be an element of maximal dimension, i.e. for a surface mesh object must be a triangle and for a volume mesh object must be a tetrahedron. Works for surface and volume meshes.
        '''
        vertices = self.get_elements()[object]
        new_vertex = np.mean(self.points[vertices], axis=0)
        self.points = np.append(self.points, new_vertex[None,:], axis=0)
        new_data = np.delete(self.get_elements(), object, axis=0)
        for v in range(len(vertices)):
            new_object = np.copy(vertices)
            new_object[v] = self.points.shape[0] - 1
            new_data = np.append(new_data, new_object[None,:], axis=0)
        self.set_elements(new_data)

    def perturb_vertices(self, orientation_preverving=True):
        '''
        Randomly perturbs all the interior vertices in the mesh. Works for surface and volume meshes. Optionally ensure that the orientation of each element is preserved (default True).
        '''
        for vertex in self.interior_vertices:
            objects = self.get_neighbourhood(vertex)
            _, index, _ = self.get_contour(objects)
            weights = self.generator.random(len(index))
            weights /= np.linalg.norm(weights)
            directions = self.points[index] - self.points[vertex][None,:]
            try:
                self.translate_vertex(vertex, np.dot(weights[None,:], directions), orientation_preverving)
            except ValueError:
                pass

    def perturb_boundary(self):
        t = self.generator.random(len(self.boundary_vertices))
        for v, vertex in enumerate(self.boundary_vertices):
            objects = objects_boundary_includes(self.get_lines(), vertex)
            index = self.get_lines()[objects]
            index = index[index != vertex]
            spline, derivative = self.get_spline([index[0], vertex, index[-1]])
            self.points[vertex,:2] = spline(t[v])

    def perturb_interface(self):
        t = self.generator.random(len(self.interface_vertices))
        for v, vertex in enumerate(self.interface_vertices):
            objects = objects_boundary_includes(self.get_lines(), vertex)
            index = self.get_lines()[objects]
            index = index[index != vertex]
            spline, derivative = self.get_spline([index[0], vertex, index[-1]])
            self.points[vertex,:2] = spline(t[v])

    def random_swap(self, orientation_preverving=True):
        '''
        Randomly swaps edges in the mesh. Optionally ensure that the orientation of each element is preserved (default True). Works for surface meshes only.

        Raises DimensionError if the mesh is not a surface mesh.
        '''
        if self.dimension != 2:
            raise DimensionError('Mesh must be a surface mesh')

        elements = self.get_triangles()
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        # is_interior = np.any(np.isin(edges, self.interior_vertices, assume_unique=False), axis=1)
        # interior_edges = edges[is_interior]
        all_edges = all_edges[np.isin(all_edges[:,0], self.get_lines()[:,0], invert=True) | np.isin(all_edges[:,1], self.get_lines()[:,1], invert=True)]
        edges = np.unique(all_edges, axis=0)

        swap = self.generator.integers(2, size=len(interior_edges), dtype=np.bool)
        for e, edge in enumerate(interior_edges):
            if swap[e]:
                try:
                    self.swap_edge(edge[0], edge[1], orientation_preverving)
                except ValueError:
                    pass

    def triangle_quality(self, triangle):
        '''
        Computes the quality of an indivdual triangle.
        '''
        factor = 4/np.sqrt(3)

        area = self.triangle_area(triangle)

        sum_edge_lengths = 0
        edge_length2 = np.empty([2,3])
        for i in range(2):
            for j in range(i+1,3):
                eij = self.points[triangle[j]] - self.points[triangle[i]]
                edge_length2[i][j] = np.dot(eij,eij)
        for i in range(2):
            for j in range(i+1,3):
                sum_edge_lengths += edge_length2[i][j]

        quality = area / (sum_edge_lengths/3)

        return quality*factor

    def triangle_area(self, triangle):
        e01 = self.points[triangle[1]] - self.points[triangle[0]]
        e02 = self.points[triangle[2]] - self.points[triangle[0]]
        e01_cross_e02 = np.cross(e01,e02)
        e01_cross_e02_norm = np.linalg.norm(e01_cross_e02)
        area = e01_cross_e02_norm / 2
        if self.normal is not None:
            area *= np.sign(np.dot(e01_cross_e02, self.normal))
        return area

    def get_contour(self, objects):
        '''
        Returns the coordinates of the contour (boundary) containing objects. Currently only works for surface meshes (need to find an ordering for points in volume meshes).

        Raises ValueError if objects do not form a single connected component.
        '''
        elements = self.get_triangles()[objects]
        all_edges = [np.sort(np.roll(e, r)[:2]) for r in range(3) for e in elements]
        submesh_edges, count = np.unique(all_edges, axis=0, return_counts=True)

        is_boundary = count == 1
        boundary_edges = submesh_edges[is_boundary]

        interior_points = np.setdiff1d(elements, boundary_edges)

        boundary = list(boundary_edges[0])
        boundary_edges = np.delete(boundary_edges, 0, axis=0)
        while boundary_edges.shape[0] > 0:
            current_vertex = np.nonzero(boundary_edges == boundary[-1])
            try:
                next_vertex = boundary_edges[current_vertex[0][0]][(current_vertex[1] + 1) % 2][0]
            except:
                raise ValueError('Objects do not form a simply connected component')
            boundary.append(next_vertex)
            boundary_edges = np.delete(boundary_edges, current_vertex[0], axis=0)

        return self.points[boundary], np.array(boundary), interior_points

    def get_open_contour(self, objects, vertex):
        '''
        Get the open contour that surrounds objects
        '''
        contour, index, _ = self.get_contour(objects)
        contour = contour[:-1]
        index = index[:-1]

        i = 0
        while index[-1] != vertex:
            if i == len(index):
                raise ValueError('Invalid open contour')
            contour = np.roll(contour, 1, axis=0)
            index = np.roll(index, 1, axis=0)
            i += 1

        return contour[:-1], index[:-1]

    def contour_length(self, contour):
        vertices = self.points[contour]
        return np.sum(np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1))

    def edge_quality(self, edge, quality=None):
        if quality is None:
            quality = self.quality()
        triangle_pair = self.find_triangles_with_common_edge(edge)
        # return np.min(quality[triangle_pair])
        t0, t1 = np.nonzero(triangle_pair)[0]
        return -np.abs(quality[t0] - quality[t1])

    def get_neighbourhood(self, vertex):
        '''
        Returns a list of object that contain vertex.
        '''
        return objects_boundary_includes(self.get_elements(), vertex)

    def get_spline(self, index):
        points = self.points[index,:2]
        s = np.linalg.norm(points - points[0], axis=1)
        s /= s[-1]
        spline = CubicSpline(s, points, axis=0)
        # spline = make_interp_spline(s, points, bc_type='natural')
        derivative = spline.derivative()
        return spline, derivative

    def find_triangles_with_common_edge(self, edge):
        objects = objects_boundary_includes_some(self.get_triangles(), 2, *edge)
        if np.count_nonzero(objects) < 2:
            # print(*args, np.count_nonzero(objects))
            error_msg = 'Vertices {}{} and {} do not define an interior edge'
            comma_list = reduce(lambda a,b: a+b, [str(a)+', ' for a in edge[:-2]], '')
            raise ValueError(error_msg.format(comma_list, edge[-2], edge[-1]))
        return objects

    def to_pyvista(self):
        """
        Convert to a PyVista mesh.
        """
        # Extract cells from meshio.Mesh object
        offset = []
        cells = []
        cell_type = []
        cell_data = {}
        next_offset = 0
        # for cell in self.cells:
        vtk_type = meshio.vtk._vtk.meshio_to_vtk_type[self.cells[-1].type]
        numnodes = meshio.vtk._vtk.vtk_type_to_numnodes[vtk_type]
        offset += [next_offset+i*(numnodes+1) for i in range(len(self.cells[-1].data))]
        cells.append(np.hstack((np.full((len(self.cells[-1].data), 1), numnodes), self.cells[-1].data)).ravel())
        cell_type += [vtk_type] * len(self.cells[-1].data)
        next_offset = offset[-1] + numnodes + 1

            # # Extract cell data
            # if cell.type in self.cell_data.keys():
            #     for kk, vv in self.cell_data[k].items():
            #         if kk in cell_data:
            #             cell_data[kk] = np.concatenate((cell_data[kk], np.array(vv, np.float64)))
            #         else:
            #             cell_data[kk] = np.array(vv, np.float64)

        # Create pyvista.UnstructuredGrid object
        points = self.points
        if points.shape[1] == 2:
            points = np.hstack((points, np.zeros((len(points),1))))

        grid = pyvista.UnstructuredGrid(
            np.array(offset),
            np.concatenate(cells),
            np.array(cell_type),
            np.array(points, np.float64),
        )

        # # Set point data
        # grid.point_arrays.update({cell.type: np.array(v, np.float64) for cell.type, v in self.point_data.items()})
        # # Set cell data
        # grid.cell_arrays.update(cell_data)

        return grid

def objects_boundary_includes_some(objects, some=1, *args):
    return np.sum(np.array([objects_boundary_includes(objects, a) for a in args]), axis=0) >= some

def objects_boundary_includes(objects, vertex):
    return np.any(np.isin(objects, vertex), axis=1)

def is_convex(contour, tol=0):
    # 2D only
    vectors = contour - np.roll(contour, 1, axis=0)
    rolled = np.roll(vectors, 1, axis=0)
    cross = vectors[:,0]*rolled[:,1] - rolled[:,0]*vectors[:,1]
    cross /= np.linalg.norm(vectors, axis=1) * np.linalg.norm(rolled, axis=1)
    return np.all(cross >= -tol) or np.all(cross <= tol)

def read(filename, file_format=None, normal=None):
    mesh = meshio.read(filename, file_format)
    return ModifiableMesh(mesh.points, mesh.cells, mesh.point_data, mesh.cell_data, mesh.field_data, mesh.point_sets, mesh.cell_sets, mesh.gmsh_periodic, mesh.info, normal)

def write(filename, mesh, file_format=None, **kwargs):
    meshio.write(filename, mesh, file_format=None, **kwargs)

class DimensionError(Exception):
    pass
