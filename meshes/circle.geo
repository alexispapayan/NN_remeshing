// Gmsh project created on Thu May 21 09:02:33 2020
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Plane Surface(1) = {1};
