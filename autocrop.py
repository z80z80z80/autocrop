import cv2
import numpy as np
import argparse
import os, glob, pathlib
from multiprocessing import Pool

def order_rect(points):
    # idea: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialize result -> rectangle coordinates (4 corners, 2 coordinates (x,y))
    res = np.zeros((4, 2), dtype=np.float32)    

    # top-left corner: smallest sum
    # top-right corner: smallest difference
    # bottom-right corner: largest sum
    # bottom-left corner: largest difference

    s = np.sum(points, axis = 1)    
    d = np.diff(points, axis = 1)

    res[0] = points[np.argmin(s)]
    res[1] = points[np.argmin(d)]
    res[2] = points[np.argmax(s)]
    res[3] = points[np.argmax(d)]

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

def cont(img, gray, user_thresh, crop):
    found = False
    loop = False
    old_val = 0 # thresh value from 2 iterations ago
    i = 0 # number of iterations

    im_h, im_w = img.shape[:2]
    while found == False: # repeat to find the right threshold value for finding a rectangle
        if user_thresh >= 255 or user_thresh == 0 or loop: # maximum threshold value, minimum threshold value 
                                                 # or loop detected (alternating between 2 threshold values 
                                                 # without finding borders            
            break # stop if no borders could be detected

        ret, thresh = cv2.threshold(gray, user_thresh, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]        
        im_area = im_w * im_h
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (im_area/100) and area < (im_area/1.01):
                epsilon = 0.1 * cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)

                if len(approx) == 4:
                    found = True
                elif len(approx) > 4:
                    user_thresh = user_thresh - 1
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val + 1:
                        loop = True
                    break
                elif len(approx) < 4:
                    user_thresh = user_thresh + 5
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val - 5:
                        loop = True
                    break

                rect = np.zeros((4, 2), dtype = np.float32)
                rect[0] = approx[0]
                rect[1] = approx[1]
                rect[2] = approx[2]
                rect[3] = approx[3]
                
                dst = four_point_transform(img, rect)
                dst_h, dst_w = dst.shape[:2]
                img = dst[crop:dst_h-crop, crop:dst_w-crop]
            else:
                if i > 100:
                    # if this happens a lot, increase the threshold, maybe it helps, otherwise just stop
                    user_thresh = user_thresh + 5
                    if user_thresh > 255:
                        break
                    print(f"Adjust Threshold: {user_thresh}")
                    print("WARNING: This seems to be an edge case. If the result isn't satisfying try lowering the threshold using -t")
                    if user_thresh == old_val - 5:
                        loop = True                
        i += 1
        if i%2 == 0:
            old_value = user_thresh

    return found, img

def get_name(filename):
    f_reversed = filename[::-1]
    index = -1 * f_reversed.find('/') - 1

    return filename[index:]

def autocrop(params):
    thresh = params['thresh']
    crop = params['crop']
    filename = params['filename']
    out_path = params['out_path']
    black_bg = params['black']

    print(f"Opening: {filename}")
    name = get_name(filename) # only the part after the folder
    img = cv2.imread(filename)
    if black_bg: # invert the image if the background is black
        img = invert(img)

    #add white background (in case one side is cropped right already, otherwise script would fail finding contours)
    img = cv2.copyMakeBorder(img,100,100,100,100, cv2.BORDER_CONSTANT,value=[255,255,255])
    im_h, im_w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res_gray = cv2.resize(img,(int(im_w/6), int(im_h/6)), interpolation = cv2.INTER_CUBIC)
    found, img = cont(img, gray, thresh, crop)

    if found:
        print(f"Saveing to: {out_path}/{name}")
        try:
            if black_bg:
                img = ~img
            cv2.imwrite(f"{out_path}/{name}", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
        except:
            None
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
    parser = argparse.ArgumentParser(description = "Crop/Rotate images automatically. Images should be single images on white background.")
    parser.add_argument("-i", metavar="INPUT_PATH", default=".",
                        help="Input path. Specify the folder containing the images you want be processed.")
    parser.add_argument("-o", metavar="OUTPUT_PATH", default="crop/",
                        help="Output path. Specify the folder name to which processed images will be written.")
    parser.add_argument("-t", metavar="THRESHOLD", type=int, default=200,
                        help="Threshold value. Higher values represent less aggressive contour search. \
                                If it's chosen too high, a white border will be introduced")
    parser.add_argument("-c", metavar="CROP", type=int, default=15,
                        help="Standard extra crop. After crop/rotate often a small white border remains. \
                                This removes this. If it cuts off too much of your image, adjust this.")
    parser.add_argument("-b", "--black", action="store_true",
                        help="Set this if you are using black/very dark (but uniform) backgrounds.")

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
                            "black": black})

        with Pool(num_threads) as p:
            results = p.map(autocrop, params)

if __name__ == "__main__":
    main()
