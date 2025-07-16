import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import pywt

# img = cv2.imread('./images/lionel_messi/960.jpg')
# print(img.shape)
# plt.imshow(img)
# plt.show()


# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# black and white
# plt.imshow(gray, cmap='gray')
# plt.show()


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# faces = face_cascade.detectMultiScale(gray,1.3,5)
# print(faces) returns the x,y,height,width of face
# (x,y,w,h) = faces[0]
# face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #255 ae rgb 6e ae red rectangle dorshe 2 thickness ni
# plt.imshow(face_img)
# plt.show()


# cv2.destroyAllWindows()
# for (x,y,w,h) in faces:
#     face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h,x:x+w]
#     roi_color = face_img[y:y+h,x:x+w]
#     # eyes = eye_cascade.detectMultiScale(roi_gray)
#     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# plt.figure()
# plt.imshow(face_img,cmap='gray')
# plt.show()
# plt.imshow(roi_color)
# plt.show()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes)>=2):
            return roi_color

cropped_img = get_cropped_image_if_2_eyes('./images/lionel_messi/960.jpg')
# plt.imshow(cropped_img)
# plt.show()

path_to_data = "./images/"
path_to_cr_data = "./cropped/"

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
# print(img_dirs)


if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    # print(celebrity_name)
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
            roi_color = get_cropped_image_if_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cr_data+celebrity_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                cropped_file_name = celebrity_name + str(count) +".png"
                cropped_file_path = cropped_folder + "/" + cropped_file_name
                cv2.imwrite(cropped_file_path,roi_color)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count = count + 1

def w2d(img, mode='haar',level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray,cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray,mode,level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H,mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# im_har = w2d(cropped_img,'db1',5)
# plt.imshow(im_har,cmap='gray')
# plt.show()

class_dict = {}
cnt = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = cnt
    cnt = cnt + 1

x = []
y = []

for celebrity_name,training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(class_dict[celebrity_name]) 

x = np.array(x).reshape(len(x),4096).astype(float)
# training vakhte saru padshe
