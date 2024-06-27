import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Your existing imports and code
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the vehicle classification model
vehicle_model = load_model('Traffic_Detection.keras')

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Detector')
top.configure(background='#CDCDCD')

# Initialize the labels
label1 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
label2 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to detect objects using YOLO
def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return class_ids, confidences, boxes, indexes

# Function to classify vehicle color
def classify_vehicle_color(image, x, y, w, h):
    vehicle_roi = image[y:y + h, x:x + w]
    hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([110, 150, 50])
    blue_upper = np.array([130, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    if cv2.countNonZero(red_mask) > 0:
        color = 'red'
    elif cv2.countNonZero(blue_mask) > 0:
        color = 'blue'
    else:
        color = 'other'
    return color

# Function to detect and annotate vehicles
def detect_and_annotate(image):
    class_ids, confidences, boxes, indexes = detect_objects(image)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in ['car', 'truck', 'bus']:
                color = classify_vehicle_color(image, x, y, w, h)
                if color == 'red':
                    box_color = (255, 0, 0)  # Blue box for red vehicles
                elif color == 'blue':
                    box_color = (0, 0, 255)  # Red box for blue vehicles
                else:
                    box_color = (0, 255, 0)  # Green box for other vehicles

                cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return image

# Function to detect and update the image
def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = detect_and_annotate(image)
        result_image = Image.fromarray(result_image)
        im = ImageTk.PhotoImage(result_image)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(foreground="#011638", text="Detection Complete")
        label2.configure(foreground="#011638", text=" ")
    except Exception as e:
        print(f"Error during detection: {e}")

# Function to show the detect button
def show_Detect_Button(file_path):
    Detect_b = Button(top, text="Detect image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Function to upload an image
def Upload_image():
    try:
        file_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), 'raw-img'))
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text=' ')
        label2.configure(text=' ')
        show_Detect_Button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")

upload = Button(top, text="Upload an image", command=Upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)

label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
heading = Label(top, text='Traffic Detector', pady=20, font=("arial", 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
