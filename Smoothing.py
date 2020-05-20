import numpy as np
import torch
from grid_patch_regression import seperate_to_sectors
from bilinear_interpolation_grid import *
from Triangulation import apply_procrustes
from Neural_network import get_smoothing_network, get_boundary_network, get_interface_network
# from matplotlib import pyplot as plt

def smooth_interior_point(contour):
    net = get_smoothing_network(contour.shape[0])
    procrustes_transform, inverse_transform, _ = get_procrustes_transform(contour)
    procrustes = procrustes_transform(contour)
    shape = procrustes.reshape(-1)
    shape = np.asarray(shape, dtype=np.float32)
    input = torch.from_numpy(shape)
    prediction = net(input[None,:])[0]
    return inverse_transform(prediction.detach().numpy())

def smooth_boundary_point(contour, tangents):
    net = get_boundary_network(contour.shape[0]-1)
    procrustes_transform, inverse_transform, tangent_transform = get_procrustes_transform(contour)
    procrustes = procrustes_transform(contour)
    shape = np.concatenate([procrustes.reshape(-1), tangent_transform(tangents).reshape(-1)])
    shape = np.asarray(shape, dtype=np.float32)
    input = torch.from_numpy(shape)
    prediction = net(input[None,:])[0]

    return inverse_transform(prediction.detach().numpy())

def smooth_interface_point(contour, points, tangents):
    net = get_interface_network(contour.shape[0])
    procrustes_transform, inverse_transform, tangent_transform = get_procrustes_transform(contour)
    procrustes = procrustes_transform(contour)
    interface = procrustes_transform(points)
    shape = np.concatenate([procrustes.reshape(-1), interface.reshape(-1), tangent_transform(tangents).reshape(-1)])
    shape = np.asarray(shape, dtype=np.float32)
    input = torch.from_numpy(shape)
    prediction = net(input[None,:])[0]
    return inverse_transform(prediction.detach().numpy())

def smooth_points(contour, net, n_interior, nb_of_grid_points=20, target_edge_length=1):
    procrustes, inverse_transform, _ = apply_procrustes(contour, full_output=True)

    X=np.linspace(-1.3,1.3,nb_of_grid_points)
    Y=np.linspace(-1.3,1.3,nb_of_grid_points)
    grid_points=np.array([[x,y] for x in X for y in Y])

    nb_sectors=int(nb_of_grid_points / 2)
    sectors,indices=seperate_to_sectors(grid_points, nb_sectors, nb_of_grid_points)
    grid_step_size=int(nb_of_grid_points / nb_sectors)

    # use network to extract predicted points
    polygon_with_target_edge_length=np.hstack([procrustes.reshape(-1), np.array(target_edge_length).reshape(1)])

    # Adding grid points of each patch for the input of the NN
    # polygon_with_grid_points=[]
    sector_qualities = np.zeros([len(sectors), grid_step_size**2])
    for s, sector in enumerate(sectors):
        polygon_with_sector_points=np.hstack([polygon_with_target_edge_length.reshape(1,len(polygon_with_target_edge_length)),sector.reshape(1,2*len(sector))])
        polygon_with_sector_points=torch.from_numpy(polygon_with_sector_points)
        polygon_with_sector_points=polygon_with_sector_points.expand(1,polygon_with_sector_points.shape[1]).type(torch.FloatTensor)
        # polygon_with_grid_points.append(polygon_with_sector_points)
        sector_quality=net(polygon_with_sector_points)
        # sector_qualities.append(sector_quality.data[0].numpy())
        sector_qualities[s] = sector_quality.data[0].numpy()

    # sector_qualities=np.array(sector_qualities)

    grid_qualities=np.empty((grid_step_size**2)*(nb_sectors**2))
    for index,point_index in enumerate(indices):
        grid_qualities[point_index]=sector_qualities.flatten()[index]

    # Point selection
    predicted_points, surrounding_points_list, grid_qualities_surrounding = select_points_updated(procrustes, grid_points, grid_qualities, contour.shape[0], nb_of_grid_points, target_edge_length)

    # Interpolate
    predicted_points=[point for i in range(contour.shape[0]) for point in bilineaire_interpolation(surrounding_points_list[i], grid_qualities_surrounding[i], predicted_points[i])]
    predicted_points=np.array(predicted_points).reshape(contour.shape[0],2)
    predicted_points=np.unique(predicted_points,axis=0)

    # print(predicted_points[0])

    return inverse_transform(predicted_points[:n_interior])
