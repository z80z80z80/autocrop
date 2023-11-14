import cv2
import numpy as np
import argparse
import os, glob, pathlib
from multiprocessing import Pool
from pathlib import Path


RATIO = 2.0

def order_rect(points):
    # initialize result -> rectangle coordinates (4 corners, 2 coordinates (x,y))
    res = np.zeros((4, 2), dtype=np.float32)

    left_to_right = points[points[:, 0].argsort()] # Sorted by x

    left_points = left_to_right[:2,:]
    left_points = left_points[left_points[:, 1].argsort()] # Sorted by y
    right_points = left_to_right[2:,:]
    right_points = right_points[right_points[:, 1].argsort()] # Sorted by y

    res[0] = left_points[0]
    res[1] = right_points[0]
    res[2] = right_points[1]
    res[3] = left_points[1]

    return res

def four_point_transform(img, points):
    # copied from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_rect(points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def cont(img, gray, user_thresh, crop, filename):

    im_h, im_w = img.shape[:2]
    im_area = im_w * im_h

    Blur = cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi

    # TODO Always resize to the same size (instead of using a constant ratio)
    res_gray = cv2.resize(Blur,(int(im_w/RATIO), int(im_h/RATIO)), interpolation = cv2.INTER_CUBIC)

    factor = 0.07
    prev_user_thresh = set()
    while user_thresh > 0 and user_thresh <= 255:
        prev_user_thresh.add(user_thresh)
        print(f"Detect with threshold: {user_thresh}")

        ret, thresh = cv2.threshold(res_gray, user_thresh, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        large_contours = 0
        kept_contours = []
        thres_incr = 0

        for cnt in contours:
            # Resize the image for the detection
            cnt[:, :, 0] = cnt[:, :, 0] * RATIO
            cnt[:, :, 1] = cnt[:, :,  1] * RATIO
            area = cv2.contourArea(cnt)
            if (im_area / 100) < area < (im_area / 1.01):
                large_contours += 1

                epsilon = factor * cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                print(f"len(approx): {len(approx)}")
                if len(approx) == 4:
                    print(f"Found an image !")
                    kept_contours.append(approx)
                elif len(approx) > 4:
                    thres_incr -= 1
                elif len(approx) < 4:
                    thres_incr += 1

        print(f"Contours {len(contours)} with {large_contours} large and {len(kept_contours)} images found. "
              f"Factor: {factor}. "
              f"Filename: {filename}")

        if large_contours == len(kept_contours):
            break
        elif thres_incr == 0:
            print("WARNING: This seems to be an edge case.")
            factor = factor + 0.01
        else:
            user_thresh += thres_incr
        if user_thresh in prev_user_thresh:
            print("WARNING: This seems to be an edge case (reusing user_thresh).")
            factor = factor + 0.01

    found_images = []

    for approx in kept_contours:

        rect = np.zeros((4, 2), dtype = np.float32)
        rect[0] = approx[0]
        rect[1] = approx[1]
        rect[2] = approx[2]
        rect[3] = approx[3]

        dst = four_point_transform(img, rect)

        dst_h, dst_w = dst.shape[:2]
        sub_img = dst[crop:dst_h-crop, crop:dst_w-crop]
        found_images.append(sub_img)

    return len(found_images), found_images

def autocrop(params):
    thresh = params['thresh']
    crop = params['crop']
    filename = params['filename']
    out_path = params['out_path']
    black_bg = params['black']
    rotation = params['rotation']
    quality = params['quality']

    print(f"Opening: {filename}")
    name = Path(filename).stem # only the part after the folder
    img = cv2.imread(filename)
    if black_bg: # invert the image if the background is black
        img = invert(img)

    if rotation:
        img = cv2.rotate(img, rotation)

    # add white background (in case one side is cropped right already, otherwise script would fail finding contours)
    img = cv2.copyMakeBorder(img,100,100,100,100, cv2.BORDER_CONSTANT,value=[255,255,255])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    found, found_images = cont(img, gray, thresh, crop, filename)

    if found:
        for idx, img in enumerate(found_images):
            print(f"Saving to: {out_path}/{name}-{idx}.jpg")
            try:
                if black_bg:
                    img = ~img
                cv2.imwrite(f"{out_path}/{name}-{idx}.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            except:
                print(f"{out_path}/{name}-{idx}.jpg cannot be saved")
            # TODO: this is always writing JPEG, no matter what was the input file type, can we detect this?

    else:
        # if no contours were found, write input file to "failed" folder
        print(f"Failed finding any contour. Saving original file to {out_path}/failed/{name}")
        if not os.path.exists(f"{out_path}/failed/"):
            os.makedirs(f"{out_path}/failed/")

        with open(filename, "rb") as in_f, open(f"{out_path}/failed/{name}", "wb") as out_f:
            while True:
                buf = in_f.read(1024**2)
                if not buf:
                    break
                else:
                    out_f.write(buf)

def invert(img):
    return ~img

def main():
    parser = argparse.ArgumentParser(description = "Crop/Rotate images automatically. Images should be large enough on white background.")
    parser.add_argument("-i", metavar="INPUT_PATH", default=".",
                        help="Input path. Specify the folder containing the images you want be processed.")
    parser.add_argument("-o", metavar="OUTPUT_PATH", default="crop/",
                        help="Output path. Specify the folder name to which processed images will be written.")
    parser.add_argument("-r", metavar="ROTATE", type=int, default=0,
                        help="Rotation value.")
    parser.add_argument("-t", metavar="THRESHOLD", type=int, default=200,
                        help="Threshold value. Higher values represent less aggressive contour search. \
                                If it's chosen too high, a white border will be introduced")
    parser.add_argument("-c", metavar="CROP", type=int, default=0,
                        help="Standard extra crop. After crop/rotate often a small white border remains. \
                                This removes this. If it cuts off too much of your image, adjust this.")
    parser.add_argument("-b", "--black", action="store_true",
                        help="Set this if you are using black/very dark (but uniform) backgrounds.")
    parser.add_argument("-q", "--quality", type=int, default=92,
                        help="JPEG quality for output images (Default = 92).")

    parser.add_argument("-p", metavar="THREADS", type=int, default=None,
                        help="Specify the number of threads to be used to process the images in parallel. \
                                If not provided, the script will try to find the value itself \
                                (which doesn't work on Windows or MacOS -> defaults to 1 thread only).")
    parser.add_argument("-s", "--single", action="store_true",
                        help="Process single image. i.e.: -i img.jpg -o crop/")
    args = parser.parse_args()

    in_path = pathlib.PureWindowsPath(args.i).as_posix() # since windows understands posix too: let's convert it to a posix path.
    out_path = pathlib.PureWindowsPath(args.o).as_posix() # (works on all systems and conveniently also removes additional '/' on posix systems)

    thresh = args.t
    crop = args.c
    num_threads = args.p
    single = args.single
    black = args.black
    match args.r:
        case 180:
            rotation = cv2.ROTATE_180
        case 90:
            rotation = cv2.ROTATE_90_CLOCKWISE
        case -90:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        case 0:
            rotation = None
        case _:
            print("Invalid roation")
            return
    quality = args.quality
    if quality < 0 or quality > 100:
        print("Invalid JPEG quality")
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    files = []

    if not single:
        types = ('*.bmp','*.BMP','*.tiff','*.TIFF','*.tif','*.TIF','*.jpg', '*.JPG','*.JPEG', '*.jpeg', '*.png', '*.PNG') #all should work but only .jpg was tested

        for t in types:
            if glob.glob(f"{in_path}/{t}") != []:
                f_l = glob.glob(f"{in_path}/{t}")
                for f in f_l:
                    files.append(f)
    else:
        files.append(in_path)

    files.sort()

    if len(files) == 0:
        print(f"No image files found in {in_path}\n Exiting.")
    else:
        if num_threads == None:
            try:
                num_threads = len(os.sched_getaffinity(0))
                print(f"Using {num_threads} threads.")
            except:
                print("Automatic thread detection didn't work. Defaulting to 1 thread only. \
                        Please specify the correct number manually via the '-p' argument.")
                num_threads = 1

        params = []
        for f in files:
            params.append({"thresh": thresh,
                            "crop": crop,
                            "filename": f,
                            "out_path": out_path,
                            "black": black,
                            "rotation": rotation,
                            "quality": quality})

        with Pool(num_threads) as p:
            results = p.map(autocrop, params)

if __name__ == "__main__":
    main()
