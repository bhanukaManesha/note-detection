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


def main() :

    print("Reading all images...")

    # Reading the images
    background_images = read_subimages(background_images_path)
    sub_images = read_subimages(sub_images_path)

    print("Read all images")

    print("Starting background image...")

    bounding_boxes = {}

    for back_img in background_images:

       

        resized_images = []

        # Calculating the height and width
        height, width, channels = back_img.shape
        height_per_image = math.floor(height / rows)
        width_per_image = math.floor(width / columns)

        total_sub_images = rows * columns

        imageName = uuid4()
        bounding_box_for_image = []

        for i in range(0, total_sub_images):

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

            box = { 'x': str(column_index * width_per_image),
                    'y': str(row_index * height_per_image),
                    'height': str(resize_height),
                    'width':str(resize_width)}

            bounding_box_for_image.append(box)

        cv2.imshow('image',back_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite(output_folder + "/" + output_currency + "/" + '%d.jpg' % imageName, back_img)

        bounding_boxes[str(imageName)] = bounding_box_for_image

    with open(output_folder + '/bounding_boxes.json', 'w') as f:
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


    # Define the parameters here
    rows = 2
    columns = 2

    background_images_path = "background/"
    sub_images_path = "RM50/"
    output_folder = "final_data"
    output_currency = "RM50"

    main()

