// Gmsh project created on Tue May 12 17:16:18 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 0.2};
//+
Point(2) = {2, 0, 0, 0.2};
//+
Point(3) = {0, 1, 0, 0.2};
//+
Point(4) = {2, 1, 0, 0.2};
//+
Line(1) = {3, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 3};
//+
Circle(5) = {0.5, 0.5, 0, 0.25, 0, 2*Pi};
//+
Curve Loop(1) = {5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Curve Loop(3) = {1, 2, 3, 4};
//+
Curve Loop(4) = {5};
//+
Plane Surface(2) = {3, 4};
//+
Characteristic Length {5} = 0.05;

