import cv2, glob, pathlib, json
import numpy as np


def get_polygons(folder):

    for apath in sorted(glob.glob('{}/images/*.png'.format(folder))):
        name = pathlib.Path(apath).stem

        img = cv2.imread('{}'.format(apath), 0)
        img2 = cv2.imread('{}'.format(apath), cv2.IMREAD_UNCHANGED)

        img = cv2.medianBlur(img, 5)
        contours, hierarchy =   cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        label = {
            'points' : []
        }

        for contour in contours:
            for [[x,y]] in contour:
                label['points'].append([float(x),float(y)])

        with open('{}/labels/{}.json'.format(folder,name), 'w') as f:
            json.dump(label, f)

        img2 = cv2.drawContours(img2, contours, -1, (0,255,0), 3)

        # cv2.imshow('Contours', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img2 = cv2.drawContours(img2.copy(), contours, -1, (0,255,0), 3)

        cv2.imwrite('{}/render/{}.png'.format(folder,name), img2)


if __name__ == "__main__":
    get_polygons("data")