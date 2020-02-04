from common import *
from utils import *

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

if __name__ == "__main__" :

    generate()
