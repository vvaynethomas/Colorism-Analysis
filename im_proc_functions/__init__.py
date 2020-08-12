import os
import numpy as np
from bs4 import BeautifulSoup
import requests
import cv2
import pandas as pd
from im_proc_functions.SkinDetector import skin_detector


def list_images(directory):
    image_list = []
    if isinstance(directory, str):
        if directory.endswith('.jpg'):
            image_list.append(directory)
        else:
            for path in os.listdir(directory):
                if path.endswith('.jpg'):
                    # print(f'directory: {directory}')
                    # print(f'path: {path}')
                    image_list.append(os.path.join(directory, path))
    elif isinstance(directory, list):
        for item in directory:
            if isinstance(item, str):
                if item.endswith('.jpg'):
                    image_list.append(item)
                elif item.endswith('/'):
                    for path in os.listdir(item):
                        if path.endswith('.jpg'):
                            image_list.append(os.path.join(item, path))
    return image_list


def extract_image_properties(image):
    pil_im = image
    im_array = np.array(pil_im)
    N = im_array.shape[1]
    M = im_array.shape[0]
    if len(im_array.shape) == 3:
        color_channels = im_array.shape[2]
    else:
        color_channels = 'None'
    return N, M, color_channels


def scrape_images(site):
    r = requests.get(site)
    data = r.text
    soup = BeautifulSoup(data, "lxml")
    for link in soup.find_all('img'):
        image = link.get('src')
        if image.endswith('.jpg'):
            try:
                image_name = os.path.split(image)[1]
                r2 = requests.get(image)
                with open(image_name, "wb") as f:
                    f.write(r2.content)
                # print("got an image")
            except:
                try:
                    if not image.startswith('http'):
                        image = 'http:/' + image
                    image_name = os.path.split(image)[1]
                    r2 = requests.get(image)
                    with open(image_name, "wb") as f:
                        f.write(r2.content)
                    # print("got an image")
                except:
                    print("something went wrong with this one")
                    continue

def crop_by_scale(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def detect_face(image_path, store_locally=False, output_directory=""):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print(type(face_cascade))
    image = cv2.imread(image_path)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.equalizeHist(greyscale)
    detected_faces = face_cascade.detectMultiScale(greyscale, 1.1, 4)
    face_images = []
    i = 0
    for (x, y, w, h) in detected_faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        current_face = image[y:y + h, x:x + w]
        current_face = crop_by_scale(current_face, scale = 0.7)
        if store_locally:
            if not output_directory:
                output_directory = os.path.join(os.getcwd(), image.stem)
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            output_path = os.path.join(output_directory, '_'.join([image_path.stem, str(i)]))
            cv2.imwrite(output_path, current_face)
        face_images.append(current_face)
        i += 1
    print(f"{str(i)} faces detected in {os.path.basename(image_path)}")
    return face_images


def skin_or_nothing(image_path):
    face_images = detect_face(image_path)
    return [skin_detector.process(image) for image in face_images]


def add_greyscale_stats(image_list, df=None):
    if df:
        column_names = list(df)
    else:
        column_names = ['average', 'min (>0)', 'max (<255']
        df = pd.DataFrame(columns=column_names)
    new_data = []
    for image in image_list:
        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cur_mean = greyscale[np.nonzero(greyscale)].mean()
        cur_min = greyscale[np.nonzero(greyscale)].min()
        cur_max = greyscale[np.nonzero(greyscale)].max()
        print(column_names)
        print([cur_mean, cur_min, cur_max])
        zipped = zip(column_names, [cur_mean, cur_min, cur_max])
        output_dict = dict(zipped)
        new_data.append(output_dict)
    df = df.append(new_data, True)
    return df

# if __name__ == "__main__":
#     fileDir = os.path.dirname(os.path.realpath('__file__'))
#     print(fileDir)
#     image_directory = os.path.join(fileDir, 'Images')
#     print(image_directory)
#     image_list = list_images(image_directory)
#     print(image_list)
