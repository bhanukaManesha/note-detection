# Parameters for the generate script

png_path = "images/"
output_path = "output/"

WIDTH = 64
HEIGHT = 64
CHANNELS = 3
params = {
    "type" : "noise",
}

# Parameters for the rotate generator

location_x = 0.5
location_y = 0.5

class_label_index = ["RM50", "RM1", "RM10", "RM20","RM100"]

class_label = {
    "RM50" : "0",
}

GRID_X = 8
GRID_Y = 8
GRID_HEIGHT = int(HEIGHT / GRID_X)
GRID_WIDTH = int(WIDTH / GRID_Y)



