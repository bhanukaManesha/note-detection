from generate import generate_background
from common import *
from utils import *


def generate(output_currency = "RM50", angle = 0) :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    # Read the images
    images, _ = read_subimages(png_path)

    count = 0

    for i,image in enumerate(images):

        # Generate the background
        background = generate_background(params)

        # Rotate the image
        rimage = rotate_image(image, angle)
        rotate_height, rotate_width, _ = rimage.shape


        # Calculating the height and width
        height, width, channels = background.shape

        height_of_note = int(math.floor(height * 0.8))
        width_of_note = int(math.floor(width * 0.8))

        default_box = {
            'confidence' : 0,
            'x': 0,
            'y': 0,
            'height': 0,
            'width': 0,
            'max_width' : 0,
            'max_height' : 0
            }

        bounding_box_for_image = [default_box for i in range(GRID_X*GRID_Y)]

        # Calculate the x and y
        x_center = width * location_x
        y_center = height * location_y

        # Resize the image
        if rotate_height > rotate_width:
            resize_image = image_resize(rimage, height=height_of_note)
        else:
            resize_image = image_resize(rimage, width=width_of_note)

        rheight, rwidth, rchannel = resize_image.shape
        
        # Calculate the top left x and y 
        x_top = abs(int(x_center - (rwidth // 2)))
        y_top = abs(int(y_center - (rheight // 2)))

        # Overlay the image to the background image
        final_image = overlay_transparent(background,resize_image,x_top,y_top)

        box = {
            'confidence': 1,
            'x': str(x_center - (x_center // GRID_WIDTH) * GRID_WIDTH ),
            'y': str(y_center - (y_center // GRID_HEIGHT) * GRID_HEIGHT ),
            'height': str(rheight),
            'width':str(rwidth),
            'max_width' : str(width),
            'max_height' : str(height),
            'class' : str(class_label[output_currency])
        }

        
        col_index = (x_center // GRID_WIDTH)
        row_index = (y_center // GRID_HEIGHT)
        loc = GRID_X * row_index + col_index
        bounding_box_for_image[int(loc)] = box


        # Write the image
        cv2.imwrite(output_path + str(angle)+ "_" + str(count) + ".jpg",final_image)

        # Create the label
        with open(output_path + str(angle)+ "_" + str(count) + ".txt", "w") as txtfile:
            
            for box in bounding_box_for_image:

                confidence = float(box['confidence'])

                if  confidence == 1.0:

                    x = float(box['x'])
                    y = float(box['y'])

                    height = float(box['height'])
                    width = float(box['width'])

                    scaled_x = x/GRID_WIDTH
                    scaled_y = y/GRID_HEIGHT

                    scaled_height = height/float(box['max_height'])
                    scaled_width = width/float(box['max_width'])

                    write_str = str(confidence) + " " + box["class"] + " " + str(scaled_x) + " " + str(scaled_y) + " " + str(scaled_width) + " " + str(scaled_height) + "\n"

                else:
                    write_str = str(confidence) + " 0 0 0 0 0\n"

                txtfile.write(write_str)

        count += 1

def generate_multiple_angles(step=5):

    for i in range(0,360,5):
        print(str(i) + "/360" )
        generate(angle=i)

if __name__ == "__main__" :
    generate_multiple_angles()
    
