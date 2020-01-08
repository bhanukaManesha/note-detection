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

GRID_X = 8
GRID_Y = 8

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


def main(output_currency) :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    print("Reading background background images\n")
    # Reading the images
    background_images_ori = read_subimages(background_images_path)

    for i in range(len(background_images_ori)):
        background_images_ori[i] = image_resize(background_images_ori[i], width = 128, height = 128)

        # print(background_images_ori[i].shape)

    if mode == "test":
        sub_images = [0,0,0,0]
        print("Reading RM50 notes and background images\n")
        sub_images[0] = read_subimages(folder_path + "RM50")
        print("Reading RM1 notes and background images\n")
        sub_images[1] = read_subimages(folder_path + "RM1")
        print("Reading RM10 notes and background images\n")
        sub_images[2] = read_subimages(folder_path + "RM10")
        print("Reading RM20 notes and background images\n")
        sub_images[3] = read_subimages(folder_path + "RM20")
    else:
        print("Reading " + str(output_currency) + " notes and background images\n")
        sub_images = read_subimages(sub_images_path)

    print()
    print("Read all images")

    print("Generating the images")

    count = 0
    total = sum(np.dot(groups,range(size,0,-1)))


    for j in range(groups):

        for i in range(size,0,-1):

            progress(count, total)

            rows = i
            columns = i

            bounding_boxes = {}

            count += 1

            background_images = copy.deepcopy(background_images_ori)

            for back_img in background_images:

                if size >= 6:
                    no_of_images = 8
                elif size > 4:
                    no_of_images = 3
                else:
                    no_of_images = 1

                if empty_images:
                    no_of_images = 0



                # Calculating the height and width
                height, width, channels = back_img.shape

                height_per_image = int(math.floor(height / rows))
                width_per_image = int(math.floor(width / columns))

                GRID_HEIGHT = int(math.floor(height / GRID_X))
                GRID_WIDTH = int(math.floor(width / GRID_Y))


                total_sub_images = GRID_X * GRID_Y

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

                    # print("Starting sub image :" + str(i) )


                    if mode == "train":
                        img = random.choice(sub_images)

                    else:
                        currency_index = random.randint(0,len(class_label) - 1)
                        output_currency = class_label_index[currency_index]
                        img = random.choice(sub_images[currency_index])


                    row_index = i // GRID_X
                    column_index = i % GRID_Y

                    resized_img = image_resize(img, width=width_per_image)

                    resize_height, resize_width, resize_channnel = resized_img.shape

                    rand_x = int((random.random() * GRID_WIDTH))
                    rand_y = int((random.random() * GRID_HEIGHT))

                    # Overlay the images to the background image
                    back_img = overlay_transparent(
                        back_img,
                        resized_img,
                        rand_x + (column_index * GRID_WIDTH),
                        rand_y + (row_index * GRID_HEIGHT)
                        )

                    box = {
                            'confidence': 1,
                            'x': str(rand_x/GRID_WIDTH),
                            'y': str(rand_y/GRID_HEIGHT),
                            'height': str(resize_height),
                            'width':str(resize_width),
                            'max_width' : str(width),
                            'max_height' : str(height),
                            'class' : str(class_label[output_currency])
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

                                write_str = str(confidence) + " " + box["class"] + " " + str(scaled_x) + " " + str(scaled_y) + " " + str(scaled_width) + " " + str(scaled_height) + "\n"

                            else:
                                write_str = str(confidence) + " 0 0 0 0 0\n"

                            txtfile.write(write_str)

                    if save_as_json:
                        with open(output_folder + 'bounding_boxes.json', 'w') as f:
                            json.dump(bounding_boxes, f)

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def read_subimages(path_to_folder):
    images = []
    files = os.listdir(path_to_folder)
    no_of_files = len(files)
    files_read = 0

    for filename in files:
        img = cv2.imread(os.path.join(path_to_folder,filename), cv2.IMREAD_UNCHANGED)

        progress(files_read,no_of_files)
        if img is not None:
            images.append(img)

        files_read += 1
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

    class_label_index = ["RM50", "RM1", "RM10", "RM20"]
    class_label = {
        "RM50" : "0",
        "RM1" : "1",
        "RM10" : "2",
        "RM20" : "3",

    }

    # Define the parameters here
    mode = "test" # train or test
    generate_mode = "random" # grid or random
    empty_images = False # determine whether to generete empty images

    if generate_mode == "grid":
        groups = 2
        size = 8
    else:
        groups = 10
        size = 8

    if empty_images:
        groups = 3
        size = 8


    background_images_path = "background/"
    folder_path = "images/"

    output_currency_str = "RM1"                             # Change this

    sub_images_path = folder_path + output_currency_str

    # output_folder = "data/train/"
    output_folder = "data/test/"

    save_as_json = False

    main(output_currency_str)
