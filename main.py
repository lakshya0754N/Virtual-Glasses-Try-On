from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
import torch
from PIL import Image


global capture,record_frame, grey, neg, face, rec, out, glass_image_path, model, region

## initiallize the variables with default values
capture=0
neg=0
face=0
glass_image_path = "static/images/glass0.png"
model="Haar"
region=0.05

#make snaps directory to save pics
try:
    os.mkdir('./snaps')
except OSError as error:
    pass

#Load pretrained face detection model    
face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
eye_detection_model = cv2.CascadeClassifier("models/haarcascade_eye.xml")


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

## Resnet model for face detection using DL.
class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x

def record(out):
    global record_frame
    while(rec):
        time.sleep(0.05)
        out.write(record_frame)


def detect_face_from_camera(image):
    global glass_image_path, face_detection_model, eye_detection_model
    glass_image = cv2.imread(glass_image_path)
    final_image = image
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ## detect the face using haar cascade
    if model=="Haar":
        faces = face_detection_model.detectMultiScale(gray_scale_image, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        if len(faces) > 0:
            for (face_x1, face_y1, face_x2, face_y2) in faces:
                
                eye_centers = []
                region_of_interest = gray_scale_image[face_y1: face_y1 + face_y2, face_x1 : face_x1 + face_x2]

                # detect eyes using haar cascade
                eyes = eye_detection_model.detectMultiScale(region_of_interest, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

                ## if eyes are detected, then only we will try to put glasses on the face
                ## find eye centers
                for (eye_x1, eye_y1, eye_x2, eye_y2) in eyes:
                    # print("Face detected")
                    eye_centers.append((face_x1 + int(eye_x1 + eye_x2/2), face_y1 + int(eye_y1 + eye_y2/2)))
                if len(eye_centers) >=2:
                    ## resize the width of the glass to be equal to the distance between the eye centers
                    glass_width = 2.5 * abs(eye_centers[1][0] - eye_centers[0][0])
                    scale_factor = glass_width / glass_image.shape[1]

                    glass_resized = cv2.resize(glass_image, None, fx= scale_factor, fy=scale_factor)

                    ## find the left most point of the left eye and right most point of the right eye
                    if eye_centers[0][0] <  eye_centers[1][0]:
                        left_eye_x1 = eye_centers[0][0]
                    else:
                        left_eye_x1 = eye_centers[1][0]

                    glass_resized_width = glass_resized.shape[1]
                    glass_resized_height = glass_resized.shape[0]
                    glass_x = left_eye_x1 - 0.28 * glass_resized_width
                    glass_y = face_y1 + 0.8 * glass_resized_height


                    ## overlay the glasses on the face
                    glass_image_mask = np.ones(image.shape, np.uint8)  * 255
                    glass_image_mask [int(glass_y): int(glass_y + glass_resized_height),
                                    int(glass_x): int(glass_x + glass_resized_width)] = glass_resized

                    ## create a mask of the glasses
                    glass_image_mask_gray = cv2.cvtColor(glass_image_mask, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(glass_image_mask_gray, 127, 255, cv2.THRESH_BINARY)

                    ## create the mask of the face
                    background_image = cv2.bitwise_and(image, image, mask = mask)

                    inverted_mask = cv2.bitwise_not(mask)

                    glasses = cv2.bitwise_and(glass_image_mask, glass_image_mask, mask=inverted_mask)

                    ## add the glasses to the face
                    final_image = cv2.add(background_image, glasses)
                    # print("Face detected")
        # cv2.imshow(f'output', final_image)
        return final_image
    elif model=="Resnet18":
        best_network = Network()
        best_network.load_state_dict(torch.load('models/face_landmarks.pth', map_location=torch.device('cpu')))
        best_network.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # image = cv2.imread('static/images/lakshya.jpg') 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert numpy array to PIL Image+
        image_pil = Image.fromarray(image_gray)

        # Apply transformations
        image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
        print(image_tensor.shape)
        # Make predictions
        # predictions = best_network(image_tensor)
        predictions = (best_network(image_tensor).cpu() + 0.5)*image.shape[1]
        predictions = predictions.view(-1,68,2)
        # pick left most and right most points
        left_most = np.min(predictions[0, :,0].detach().numpy())
        right_most = np.max(predictions[0, :,0].detach().numpy())
        # pick top most and bottom most points
        top_most = np.min(predictions[0, :,1].detach().numpy())
        bottom_most = np.max(predictions[0, :,1].detach().numpy())
        region_of_interest = gray_scale_image[top_most: bottom_most, left_most : right_most]
        eyes = eye_detection_model.detectMultiScale(region_of_interest, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        ## if eyes are detected, then only we will try to put glasses on the face
        ## find eye centers
        for (eye_x1, eye_y1, eye_x2, eye_y2) in eyes:
            # print("Face detected")
            eye_centers.append((face_x1 + int(eye_x1 + eye_x2/2), face_y1 + int(eye_y1 + eye_y2/2)))
        if len(eye_centers) >=2:
            ## resize the width of the glass to be equal to the distance between the eye centers
            glass_width = 2.5 * abs(eye_centers[1][0] - eye_centers[0][0])
            scale_factor = glass_width / glass_image.shape[1]

            glass_resized = cv2.resize(glass_image, None, fx= scale_factor, fy=scale_factor)

            ## find the left most point of the left eye and right most point of the right eye
            if eye_centers[0][0] <  eye_centers[1][0]:
                left_eye_x1 = eye_centers[0][0]
            else:
                left_eye_x1 = eye_centers[1][0]

            glass_resized_width = glass_resized.shape[1]
            glass_resized_height = glass_resized.shape[0]
            glass_x = left_eye_x1 - 0.28 * glass_resized_width
            glass_y = face_y1 + 0.8 * glass_resized_height


            ## overlay the glasses on the face
            glass_image_mask = np.ones(image.shape, np.uint8)  * 255
            glass_image_mask [int(glass_y): int(glass_y + glass_resized_height),
                            int(glass_x): int(glass_x + glass_resized_width)] = glass_resized

            ## create a mask of the glasses
            glass_image_mask_gray = cv2.cvtColor(glass_image_mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(glass_image_mask_gray, 127, 255, cv2.THRESH_BINARY)

            ## create the mask of the face
            background_image = cv2.bitwise_and(image, image, mask = mask)

            inverted_mask = cv2.bitwise_not(mask)

            glasses = cv2.bitwise_and(glass_image_mask, glass_image_mask, mask=inverted_mask)

            ## add the glasses to the face
            final_image = cv2.add(background_image, glasses)
                    # print("Face detected")
        # cv2.imshow(f'output', final_image)
        return final_image
    else:
        return final_image
    
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,record_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame = detect_face_from_camera(frame)
                time.sleep(region)
            if(neg):
                frame=cv2.bitwise_not(frame)   
                time.sleep(0.05) 
            if(capture):
                ## capture the image and save it
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['snaps', "snap_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)  
            try:
                _, stream = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = stream.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tasks',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('neg') == 'Negative':
            global neg
            if neg:
                neg=0
            else:
                neg=1
        elif  request.form.get('face') == 'Try On':
            global face
            if face:
                face=0
            else:
                face=1
            if(face):
                time.sleep(4)   
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/glass',methods=['POST','GET'])
def glass():
    global glass_image_path
    if request.method == 'POST':
        if request.form.get('glass1') == 'Glass1':
            glass_image_path = "static/images/glass0.png"
        elif  request.form.get('glass2') == 'Glass2':
            glass_image_path = "static/images/glass1.png"
        elif  request.form.get('glass3') == 'Glass3':
            glass_image_path = "static/images/glass2.png"
        elif  request.form.get('glass4') == 'Glass4':
            glass_image_path = "static/images/glass3.png"
                          
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')    

@app.route('/model_func',methods=['POST','GET'])
def model_func():
    global model, region
    if request.method == 'POST':
        if request.form.get('haar') == 'Harr Cascade':
            model="Haar"
        elif  request.form.get('dl') == 'Resnet18':
            region=0.1
            model="Resnet18"
                          
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')   

if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()