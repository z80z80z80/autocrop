import cv2
import numpy as np
import argparse
import os
from multiprocessing import Pool
from pathlib import Path


# main calls autocrop (via the Pool).
#
# autocrop calls cv2.imread (to load the image).
#
# autocrop calls invert (if your background is black).
#
# autocrop calls cont (to find the shapes).
#
# cont calls four_point_transform (to fix the perspective).
#
# four_point_transform calls order_rect (to sort the corners).

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
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype = np.float32)

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, m, (max_width, max_height))

    # return the warped image
    return warped

def cont(img, gray, user_thresh, crop, filename,target_area = 1000000):
    im_h, im_w = img.shape[:2]
    im_area = im_w * im_h

    if im_area > target_area:
        scale = np.sqrt(target_area / im_area)
    else:
        scale = 1.0

    new_w = int(im_w * scale)
    new_h = int(im_h * scale)

    blur = cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi

    res_gray = cv2.resize(blur,(new_w,new_h), interpolation = cv2.INTER_AREA)

    factor = 0.07
    prev_user_thresh = set()

    while 0<user_thresh<255:
        prev_user_thresh.add(user_thresh)
        print(f"Detect with threshold: {user_thresh}")

        ret, thresh = cv2.threshold(res_gray, user_thresh, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        large_contours = 0
        kept_contours = []
        thresh_inc = 0

        for cnt in contours:
            # Resize the image for the detection
            cnt[:, :, 0] = cnt[:, :, 0] /scale
            cnt[:, :, 1] = cnt[:, :,  1] /scale

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
                    thresh_inc -= 1
                elif len(approx) < 4:
                    thresh_inc += 1

        print(f"Contours {len(contours)} with {large_contours} large and {len(kept_contours)} images found. "
              f"Factor: {factor}. "
              f"Filename: {filename}")

        if large_contours == len(kept_contours):
            break
        elif thresh_inc == 0:
            print("WARNING: This seems to be an edge case.")
            factor += 0.01
        else:
            user_thresh += thresh_inc
        if user_thresh in prev_user_thresh:
            print("WARNING: This seems to be an edge case (reusing user_thresh).")
            factor += 0.01

    found_images = []
    for approx in kept_contours:
        rect = approx.reshape(4, 2).astype(np.float32)

        dst = four_point_transform(img, rect)

        dst_h, dst_w = dst.shape[:2]
        sub_img = dst[crop:dst_h-crop, crop:dst_w-crop]
        found_images.append(sub_img)

    return len(found_images), found_images

def autocrop(params):
    valid_extensions = {'.bmp', '.tiff', '.tif', '.jpg', '.jpeg', '.png'}

    thresh = params['thresh']
    crop = params['crop']
    filename = params['filename']
    out_path = params['out_path']
    black_bg = params['black']
    rotation = params['rotation']
    quality = params['quality']

    print(f"Opening: {filename}")

    # Path handling
    file_path = Path(filename)
    name = file_path.stem
    ext = file_path.suffix.lower()

    # Fallback if extension is unknown or missing
    if ext not in valid_extensions:
        print(f"Warning: {ext} not in supported list. Defaulting to original extension.")

    img = cv2.imread(filename)
    if img is None:
        print(f"Error: Could not read {filename}")
        return

    if black_bg:
        img = cv2.bitwise_not(img)  # Using cv2.bitwise_not for clarity

    if rotation:
        img = cv2.rotate(img, rotation)

    # Add white border
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, found_images = cont(img, gray, thresh, crop, filename)

    if found:
        for idx, output_img in enumerate(found_images):
            # Dynamic output path with original extension
            out_filename = f"{name}-{idx}{ext}"
            full_out_path = os.path.join(out_path, out_filename)

            # Determine correct encoding parameters
            write_params = []
            if ext in ['.jpg', '.jpeg']:
                write_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            elif ext == '.png':
                # Map 0-100 quality to 0-9 compression (OpenCV PNG scale)
                png_comp = max(0, min(9, int((100 - quality) / 11)))
                write_params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_comp]

            print(f"Saving to: {full_out_path}")
            try:
                if black_bg:
                    output_img = cv2.bitwise_not(output_img)
                cv2.imwrite(full_out_path, output_img, write_params)
            except Exception as e:
                print(f"{full_out_path} cannot be saved: {e}")

    else:
        print(f"Failed finding any contour. Saving original file to {out_path}/failed/{name}{ext}")
        failed_dir = os.path.join(out_path, "failed")
        if not os.path.exists(failed_dir):
            os.makedirs(failed_dir)

        failed_path = os.path.join(failed_dir, f"{name}{ext}")
        with open(filename, "rb") as in_f, open(failed_path, "wb") as out_f:
            while True:
                buf = in_f.read(1024 ** 2)
                if not buf:
                    break
                out_f.write(buf)

def run_crop(input_path=".", output_path="crop/",rotate=0, threshold=200, crop=0, black_bg=False,quality=92, threads=None):
    # Convert both to objects immediately
    input_path = Path(input_path)
    output_path = Path(output_path)

    single = True if input_path.is_file() else False

    # Determine the base directory
    base_dir = input_path if input_path.is_dir() else input_path.parent

    output_path = base_dir/output_path if not output_path.is_absolute() else output_path

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Working with: {output_path.as_posix()}")

    match rotate:
        case 180:
            rotation = cv2.ROTATE_180
        case 90:
            rotation = cv2.ROTATE_90_CLOCKWISE
        case -90:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        case 0:
            rotation = None
        case _:
            print("Invalid rotation value")
            return

    if quality < 0 or quality > 100:
        print("Invalid JPEG quality")
        return

    folder = Path(input_path)

    if not single:
        # Extensions we want to look for
        extensions = {'.bmp', '.tiff', '.tif', '.jpg', '.jpeg', '.png'}

        # List comprehension:
        files = [str(f) for f in folder.iterdir() if f.suffix.lower() in extensions]
    else:
        files = [input_path]

    files.sort()

    if len(files) == 0:
        print(f"No image files found in {input_path}\n Exiting.")
    else:
        threads =  threads or os.cpu_count() or 1
        print(f"Using {threads} threads")

        params = []
        for f in files:
            params.append({
                            "out_path": output_path,
                            "rotation": rotation,
                            "thresh": threshold,
                            "crop": crop,
                            "black": black_bg,
                            "quality": quality,
                            "filename": f,
            })

        with Pool(threads) as p:
            _ = p.map(autocrop, params)

def pass_args(args_list = None):
    parser = argparse.ArgumentParser(
        description="Crop/Rotate images automatically. Images should be large enough on white background.")
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

    args = parser.parse_args(args_list)

    run_crop(input_path=args.i, output_path=args.o,rotate=args.r, threshold=args.t, crop=args.c, black_bg=args.black,quality=args.quality, threads=args.p)


if __name__ == "__main__":
    pass_args()