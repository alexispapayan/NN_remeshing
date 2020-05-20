# from connectivity import ConnectivityNet as Net
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pickle
from functools import lru_cache

@lru_cache(maxsize=8)
def get_smoothing_network(n_pts): #, n_interior
    # filename = 'parameters/{}_{}_20_grid_NN.pkl'.format(n_pts, n_interior)
    filename = 'parameters/{}_smoothing_NN.pkl'.format(n_pts)
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    return net.eval()

@lru_cache(maxsize=8)
def get_boundary_network(n_pts):
    filename = 'parameters/{}_boundary_NN.pkl'.format(n_pts)
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    return net.eval()

@lru_cache(maxsize=8)
def get_interface_network(n_pts):
    filename = 'parameters/{}_interface_NN.pkl'.format(n_pts)
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    return net.eval()

@lru_cache(maxsize=32)
def get_connectivity_network(n_pts, n_interior=0):
    if n_interior == 0:
        filename = 'parameters/{}_connection_NN.pkl'.format(n_pts)
    else:
        filename = 'parameters/{}_{}_connection_NN.pkl'.format(n_pts, n_interior)
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    return net.eval()

class Net(nn.Module):

    def __init__(self, in_features_dimension, out_features_dimension, nb_of_hidden_layers, nb_of_hidden_nodes, batch_normalization=False):

        super(Net,self).__init__()

        self.nb_hidden_layers=nb_of_hidden_layers
        self.do_bn=batch_normalization
        self.fcs=[]
        self.bns=[]
        self.bn_input=nn.BatchNorm1d(in_features_dimension,momentum=0.5) #for input data

        for i in range(nb_of_hidden_layers):    # build hidden layers and BN layers

            input_size=in_features_dimension if i==0 else nb_of_hidden_nodes
            fc=nn.Linear(input_size,nb_of_hidden_nodes)
            setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)

            if self.do_bn:
                bn = nn.BatchNorm1d(nb_of_hidden_nodes, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)                         # IMPORTANT set layer to the Module
                self.bns.append(bn)

            self.predict = nn.Linear(nb_of_hidden_nodes,out_features_dimension)         # output layer
            self._set_init(self.predict)                                              # parameters initialization


    def _set_init(self, layer):
            init.normal(layer.weight, mean=0., std=.1)
            init.constant(layer.bias, B_INIT)




    def forward(self, x):
        ACTIVATION=F.relu
        # pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)     # input batch normalization
        # layer_input = [x]
        for i in range(self.nb_hidden_layers):
            x = self.fcs[i](x)
            # pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION(x)
            # layer_input.append(x)
        out = self.predict(x)
        return out

class alt_2d_conv_net(nn.Module):

    def __init__(self,nb_of_filters,nb_of_hidden_nodes,out_dimension,nb_of_edges,nb_of_points):
        super(alt_2d_conv_net,self).__init__()

        self.nb_of_edges=nb_of_edges
        self.nb_of_points=nb_of_points

        self.nb_of_filters=nb_of_filters

        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=(2,1)),
                                 nn.MaxPool2d(stride=1,kernel_size=(2,1)),nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=(2,1)),
                                 nn.MaxPool2d(stride=1,kernel_size=(2,1)),nn.ReLU(inplace=True))



        self.fc=nn.Sequential(  nn.BatchNorm1d(num_features=nb_of_filters*2*(nb_of_edges-1)+2*(nb_of_points)),

                                nn.Linear(2*nb_of_filters*(nb_of_edges-1)+2*(nb_of_points),nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),

                              nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),

                              nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),



                               nn.Linear(nb_of_hidden_nodes,out_dimension) )




    def forward(self,x):

        polygons_points=x.narrow(1,0,1).narrow(2,0,self.nb_of_edges+1)
        inner_points=x.narrow(1,0,1).narrow(2,self.nb_of_edges+1,self.nb_of_points).resize(len(x),2*self.nb_of_points)


        conv_result1=self.conv1(polygons_points)
        #conv_result2=self.conv2(inner_points)

        # reshape the convolution results

        conv_result1=conv_result1.view(-1,self.nb_of_filters*(2*(self.nb_of_edges-1)))
        #conv_result2=conv_result2.view(-1,self.nb_of_filters*(2*(self.nb_of_points-1)))

        concat_tensor=torch.cat([conv_result1,inner_points],1)
        output=self.fc(concat_tensor)
        return output
