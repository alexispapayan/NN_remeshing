from Triangulation_with_points import retriangulate_with_interior
import pickle
import numpy as np

with open('contour.pkl', 'rb') as file:
    input = pickle.load(file)

contour = input[:-1]
interior = input[-1:]

retriangulate_with_interior(contour, *interior)
