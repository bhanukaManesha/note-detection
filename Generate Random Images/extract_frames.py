import cv2
import numpy as np
from uuid import uuid4

def resize_image(image, height, width, channels):

    o_height, o_width, _ = image.shape

    resized = np.zeros((height, width, channels))

    scale_factor = o_height / width

    resized_width = int(o_width/scale_factor)
    resized_height = int(o_height/scale_factor)

    res = cv2.resize(image, dsize=(resized_width,resized_height), interpolation=cv2.INTER_CUBIC)

    resized = res[0:height, 0:width, :]

    return resized/255.0


def extract(save_path):

    boxes = [[0.0,0,0,0,0,0] for i in range(49)]

    namefile = open(output_dir + "/validation.txt", "a+")

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('1.MOV')

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            small_frame = resize_image(frame, 224, 224, 3) * 255.0

            image_name = str(uuid4())
            cv2.imwrite(save_path + "/images/" + image_name + ".jpg",small_frame)

            with open(save_path + "/labels/" + image_name + ".txt", "w+") as txtfile:
                for box in boxes:
                    write_str = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " " + str(box[4]) + " " + str(box[5]) + "\n"
                    txtfile.write(write_str)

            namefile.write(save_path + "/images/" + image_name + ".jpg\n")
                    
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    output_dir = "generate_data"

    extract(output_dir)