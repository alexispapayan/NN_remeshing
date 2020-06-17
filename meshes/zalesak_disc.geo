// Gmsh project created on Sat Jun 13 18:04:42 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 4, 0, 1.0};
//+
Point(3) = {4, 4, 0, 1.0};
//+
Point(4) = {4, 0, 0, 1.0};
//+
Line(1) = {1, 4};
//+
Line(2) = {4, 3};
//+
Line(3) = {3, 2};
//+
Line(4) = {2, 1};
//+
Circle(5) = {2., 2.75, 0, 0.5, 2*Pi,0 };
//+
Point(6) = {1.97, 2.85, 0, 0.1};
//+
Point(7) = {2.03, 2.85, 0, 0.1};
//+
Point(8) = {1.97, 2.25, 0, 0.1};
//+
Point(9) = {2.03, 2.25, 0, 0.1};
//+
Line Loop(1) = {5};
//+
Surface(1) = {1};
//+
Line(6) = {7, 6};
//+
Line(7) = {7, 9};
//+
Line(8) = {8, 6};
//+
Line(9) = {9, 8};
//+
Line Loop(3) = {6,-7,-8,-9};
//+
Surface(2) = {3};
//+
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };//+
Line Loop(5) = {4, 1, 2, 3};
//+
Line Loop(6) = { 8,-7,6,-5,-9};
//+
Plane Surface(4) = {5, 6};
//+
Characteristic Length {5, 8, 7, 6, 9} = 0.1;
