import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import time
import util
import shutil

_finalocr = []
ocr_dict = {'B 2228HM': 'Pawan', 'AG 397072': 'Keshar', 'BP - 199- SN': 'Enayat'}

#define all the paths
root_dir = os.path.join(os.getcwd(), os.pardir)
not_root_dir = os.path.join(os.getcwd())


model_cfg_path = os.path.join(not_root_dir, 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join(not_root_dir, 'model', 'weights', 'model.weights')
class_names_path = os.path.join(not_root_dir, 'model', 'class.names')
input_dir = os.path.join(root_dir, 'data')


print("Do you want to use webcam or directry?")
print("1. Webcam")
print("2. Directory")
user_input = int(input("Enter your choice: "))
if user_input == 1:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(os.path.join(input_dir, 'test.jpg'), frame)
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    pass


for img_name in os.listdir(input_dir):

    img_path = os.path.join(input_dir, img_name)

    # load class names
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image

    img = cv2.imread(img_path)

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

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # plot
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        """
        cv2.putText(img,
                    class_names[class_ids[bbox_]],
                    (int(xc - (w / 2)), int(yc + (h / 2) - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    7,
                    (0, 255, 0),
                    15)
        """

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            15)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_thresh)
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                # print(text, text_score)
                _finalocr.append(text)
                #if the text is present in ocr_dict, then print the owner name
                if text in ocr_dict:
                    print(ocr_dict[text])
                    cv2.putText(img,
                                'Owner: ' + ocr_dict[text],
                                #text position should be above the bounding box
                                (int(xc - (w / 2)), int(yc - (h / 2) - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4, #font size
                                (0, 0, 255), #font color
                                15) 



    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

    plt.show()

print(_finalocr)