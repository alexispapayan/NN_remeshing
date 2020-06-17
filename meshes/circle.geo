// Gmsh project created on Thu May 21 09:02:33 2020
SetFactory("OpenCASCADE");
//+
Circle(1) = {.5, 0.75, 0, 0.15, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Plane Surface(1) = {1};
