import cv2
import os
import numpy as np
import random
import math
import imutils
from uuid import uuid4
import json


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

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def main(rows = 2,columns = 2) :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    print("Reading all images...")

    # Reading the images
    background_images = read_subimages(background_images_path)
    sub_images = read_subimages(sub_images_path)

    print("Read all images")

    print("Starting background image...")

    bounding_boxes = {}


    for back_img in background_images:

        no_of_images = 10

        resized_images = []

        # Calculating the height and width
        height, width, channels = back_img.shape
        height_per_image = int(math.floor(height / rows))
        width_per_image = int(math.floor(width / columns))

        total_sub_images = rows * columns

        imageName = uuid4()

        default_box = {
                'confidence' : 0,
                'x': 0,
                'y': 0,
                'height': 0,
                'width': 0,
                'max_width' : 0,
                'max_height' : 0
            }

        bounding_box_for_image = [default_box for i in range(total_sub_images)]


        for i in range(0, total_sub_images):

            timer = int((rows * random.random()))
            if no_of_images == 0 and generate_mode == "random":
                continue

            if generate_mode == "random" and timer > 0:
                timer -= 1
                continue

            no_of_images -= 1

            print("Starting sub image :" + str(i) )
            img = random.choice(sub_images)
            row_index = i // rows
            column_index = i % rows

            resized_img = image_resize(img, width=width_per_image)

            resize_height, resize_width, resize_channnel = resized_img.shape
            # resized_img = cv2.resize(img, (width_per_image,height_per_image), interpolation = cv2.INTER_CUBIC)

            # Overlay the images to the background image
            back_img = overlay_transparent(back_img,
                resized_img,
                (column_index * width_per_image),
                (row_index * height_per_image)
                )

            box = {
                    'confidence': 1,
                    'x': str(((column_index * width_per_image) + ((column_index + 1) * width_per_image))/2),
                    'y': str(((row_index * height_per_image) + ((row_index + 1) * height_per_image))/2),
                    'height': str(resize_height),
                    'width':str(resize_width),
                    'max_width' : str(width),
                    'max_height' : str(height)
                }

            bounding_box_for_image[i] = box

        cv2.imwrite(output_folder + "/images/" +  str(imageName) + '.jpg', back_img)

        bounding_boxes[str(imageName)] = bounding_box_for_image

        if not save_as_json:
            with open(output_folder + mode + ".txt", "a") as txtfile:
                txtfile.write(output_folder + "images/" + str(imageName) + ".jpg\n")

            with open(output_folder + "labels/" + str(imageName) + ".txt", "a") as txtfile:
                for box in bounding_box_for_image:

                    confidence = float(box['confidence'])

                    if  confidence == 1.0:
                        max_height = float(box['max_height'])
                        max_width = float(box['max_width'])

                        x = float(box['x'])
                        y = float(box['y'])

                        height = float(box['height'])
                        width = float(box['width'])

                        scaled_x = x/max_width
                        scaled_y = y/max_height

                        scaled_height = height/max_height
                        scaled_width = width/max_width

                        write_str = str(confidence) + " " + class_label[output_currency] + " " + str(scaled_x) + " " + str(scaled_y) + " " + str(scaled_width) + " " + str(scaled_height) + "\n"

                    else:
                        write_str = str(confidence) + " 0 0 0 0 0\n"

                    txtfile.write(write_str)

    if save_as_json:
        with open(output_folder + 'bounding_boxes.json', 'w') as f:
            json.dump(bounding_boxes, f)

def read_subimages(path_to_folder):
    images = []
    for filename in os.listdir(path_to_folder):
        img = cv2.imread(os.path.join(path_to_folder,filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images


def overlay_image(layer0_img, layer1_img, x, y):

    height, width, channel = layer1_img.shape

    layer0_img[y: y + height, x: x + width ] = layer1_img

    return layer0_img

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

    class_label = {
        "RM50" : "0",
        "RM100" : "1"
    }

    # Define the parameters here
    r = 2
    c = 2
    mode = "test" # train or test
    generate_mode = "random" # grid or random

    background_images_path = "background/"
    sub_images_path = "RM50/"

    # output_folder = "data/train/"
    output_folder = "data/test/"

    output_currency = "RM50"

    save_as_json = False

    for i in range(20):
        print(str(i))
        main(rows=8, columns=8)

#    n = 20
#    for i in range(n,0,-1):
#        main(rows=i, columns=i)
