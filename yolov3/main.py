import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import time
import util
import shutil
import streamlit as st
from PIL import Image

root_dir = os.path.join(os.getcwd(), os.pardir)
not_root_dir = os.path.join(os.getcwd())
model_cfg_path = os.path.join(not_root_dir, 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join(not_root_dir, 'model', 'weights', 'model.weights')
class_names_path = os.path.join(not_root_dir, 'model', 'class.names')
input_dir = os.path.join(root_dir, 'data')


_finalocr = []
ocr_dict = {'B 2228HM': 'Pawanesh Ranjan Kumar Mishra Gupta Singh Raizada G (Pro Engineer)', 'AG 397072': 'Keshar', 'BP - 199 - SN': 'Enayat'}


st.title("License Plate Recognition")
st.write("Upload the image of the vehicle")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image for detection.', use_column_width=True)
    st.write("")

    # load class names
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()
    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
    # load image
    img = np.array(image)
    H, W, _ = img.shape
    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # non-max suppression
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    #plot
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        
        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            15)
        st.info("Detecting license plate...")
        #loading screen
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.07)
            my_bar.progress(percent_complete + 1)  
        st.image(img, caption='Detected License Plate', use_column_width=True)

             
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        st.info("Converting to gray...")
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.image(license_plate_gray, caption='Converting to gray', use_column_width=True)
        

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        #show the image
        st.info("Thresholding...")
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.image(license_plate_thresh, caption='Thresholding', use_column_width=True)
        

        output = reader.readtext(license_plate_thresh)
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                _finalocr.append(text)
                st.info("Recognizing...")
                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)
                #write in bold
                st.write("The license plate number is : ", text)
                st.info("Searching for the owner in the database...")
                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.05)
                    my_bar.progress(percent_complete + 1)
                if text in ocr_dict:
                    st.write("The owner of the vehicle is : ", ocr_dict[text])
                    img = r'C:\notcdrive\jeevan\minor_proj\planb\cvdev\ANPR\dbimg\team-3.jpg'
                    image = Image.open(img)
                    st.image(image, width=100, caption='Owner of the vehicle', use_column_width=True)
                else:
                    st.write("The owner of the vehicle is not found in the database")
                break
    #st.balloons()