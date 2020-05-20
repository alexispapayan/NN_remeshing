// Gmsh project created on Thu Apr 16 10:29:48 2020
SetFactory("OpenCASCADE");
lc_boundary=0.5;
lc=1e-1;

Point(1)={1,-1,0,lc_boundary};
Point(2)={-1,-1,0,lc_boundary};
Point(3)={-1,1,0,lc_boundary};
Point(4)={1,1,0,lc_boundary};
//+
Point(8)={-1,-0.5,0,lc};
//+
Point(9)={-0.5,-1,0,lc};


//+
Line(1) = {2, 9};
//+
Line(2) = {9, 1};
//+
Line(3)={1,4};
//+
Line(4)={4,3};
//+
Line(5) = {3, 8};
//+
Line(6)={8,2};



// Define the parabola with y=a*x^2 with xmax extremes and origin
a = 3;
x_max = 0.3;



Point(5) = {0,0,0,lc};
Point(6) = {x_max , a*x_max*x_max, 0,lc};
Point(7) = {-x_max , a*x_max*x_max, 0,lc};


//+
// Translate the origin by x_origin_translation and y_origin_translation and rotate 3*pi/4 around the origin point (z-axis)

x_origin_translation=-0.4;
y_origin_translation=-0.4;

Translate {x_origin_translation, y_origin_translation, 0} {
  Point{5}; 
  Point{6}; 
  Point{7};
}

Rotate{{0,0,1},{x_origin_translation,  y_origin_translation, 0},3*Pi/4}{Point{6};Point{7};}



//+
Spline(7) = {8, 6, 5, 7, 9};



//+
Curve Loop(1) = {1,2,3,4,5,6};
//+
Plane Surface(1) = {1};
//+
Line{7} In Surface{1};//+

