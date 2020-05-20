// Gmsh project created on Tue Mar 31 14:40:17 2020
SetFactory("OpenCASCADE");

lc=0.05;
//+
Point(1) = {1, 0, 0, 1.0};
//+
Point(2) = {0, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0, 0.5, 0, lc};
//+
Point(6) = {1, 0.5, 0, lc};
//+
Point(7) = {0.7, 0.7, 0, lc};
//+
Point(8) = {0.4, .4, 0, lc};
//+
Line(1) = {5, 4};
//+
Line(2) = {6, 3};
//+
Line(3) = {3, 4};
//+
Spline(4) = {5, 8, 7, 6};
//+
Line(5) = {6, 1};
//+
Line(6) = {2, 1};
//+
Line(7) = {5, 2};

//+
Curve Loop(2) = {3, -1, 4, 2};

//+
Curve Loop(3) = {6, -5, -4, 7};
//+
Plane Surface(2) = {2};
Plane Surface(3) = {3};

