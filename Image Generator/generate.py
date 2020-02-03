import cv2
import os
import numpy as np
import random
import math
import imutils
from uuid import uuid4
import json
import sys
import copy
import glob

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    if width is not None and height is not None:
        dim = (width,height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def generate_background(__params__):
    
    if __params__["type"] == "white" :

        return np.full((HEIGHT,WIDTH,CHANNELS), 255.)

    elif __params__["type"] == "black" :

        return np.full((HEIGHT,WIDTH,CHANNELS), 0.)

    elif __params__["type"] == "noise" :

        return np.random.randint(256, size=(HEIGHT, WIDTH,CHANNELS))


def generate() :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    # Read the images
    images, filenames = read_subimages(png_path)

    filenames = [x.replace(".png","") for x in filenames]
    filenames = [x.replace(png_path,"") for x in filenames]

    images = np.asarray(images)

    for i,image in enumerate(images):
        # Generate the background
        background = generate_background(params)

        # Resize the image
        image = image_resize(image, width=WIDTH, height=HEIGHT)

        # Overlay the image
        final_image = overlay_transparent(background,image,0,0)

        # Write the image
        print(output_path + filenames[i] + '.jpg')

        cv2.imwrite(output_path + filenames[i] + '.jpg', final_image)

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def read_subimages(path_to_folder):
    images = []
    files = glob.glob(path_to_folder + "*.png")

    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)

    return images, files

def overlay_image(layer0_img, layer1_img, x, y):

    height, width, channel = layer1_img.shape

    layer0_img[y: y + height, x: x + width ] = layer1_img

    return layer0_img

def doOverlap(first, second): 
    x_axis_not_overlap = False
    y_axis_not_overlap = False

    if(int(first["x1"]) > int(second["x2"]) or int(first["x2"]) < int(second["x1"])):
        x_axis_not_overlap = True
  
    if(int(first["y1"]) > int(second["y2"]) or int(first["y2"]) < int(second["y1"])):
        y_axis_not_overlap = True
  
    if x_axis_not_overlap and y_axis_not_overlap:
        return False
    else:
        return True

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

if __name__ == "__main__" :

    png_path = "images/"
    output_path = "output/"

    WIDTH = 64
    HEIGHT = 64
    CHANNELS = 3
    params = {
        "type" : "noise",
    }


    generate()
