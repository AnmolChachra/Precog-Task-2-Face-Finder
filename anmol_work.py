from flask import Flask,render_template,request,redirect
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import dlib
import tensorflow as tf
import os
import time
import datetime
from imutils import face_utils
import argparse


FaceFinderModelPath =  'Models/haarcascade_frontalface_default.xml'
FaceAlignmentModelPath = 'Models/shape_predictor_68_face_landmarks.dat'


apps = Flask(__name__)

@apps.route('/')
def homepage():
    return render_template('homepage.html')

@apps.route('/uploader',methods=['GET','POST'])
def uploader():
    f = request.files['file']
    f.save('static'+'/'+secure_filename(f.filename))
    result = face_finder('static/'+str(f.filename))
    #name_of_file = f.filename
    return render_template('face.html',result=result)



def find_faces_and_rects(path):
#This Function will return the path to the image with or without boxes and array of rectangles(empty list for no face)
    
    if not(os.path.exists('picrectangles')):
        os.makedir('picrectangles')

    if not(os.path.exists('Faces Detected')):
        os.makedir('Faces Detected')

    face_cascade = cv2.CascadeClassifier(FaceFinderModelPath) # Trained Model for facial classification
    
    sub_count = 0
    
    image = cv2.imread(path)
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Changing to grayscale
    except:
        gray = image
        pass
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) #Detecting faces

    # Image with face rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        cv2.imwrite('Faces Detected/'+str(sub_count+1)+'.jpg',image)
        cv2.imwrite('static/'+str(sub_count+1)+'.jpg',image)
        sub_count+=1

    if sub_count==0:
        cv2.imwrite('Faces Detected/'+str(sub_count+1)+'.jpg',image)

    Rectangle_Image_Data = []
    for (x,y,w,h) in faces:  
        sub_count+=1
        cv2.imwrite('picrectangles/'+str(sub_count)+".jpg",image[y:y+h,x:x+h,:])
        Rectangle_Image_Data.append(image[y:y+h,x:x+h,:])

    return 'Faces Detected/', 'picrectangles/'


def align_face(images_dir):

    # Takes image_dir containing rectangular face images as input
    # Returns aligned images path

    if not(os.path.exists('aligned picrectangles')):
        os.makedir('aligned picrectangles')

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
    fa = face_utils.FaceAligner(predictor, desiredFaceWidth=100)

    source_path = images_dir

    destination_path = 'aligned picrectangles/'

    Records = []
    count=0
    for img in os.listdir(source_path):
        image_path = os.path.join(source_path, img)
        image = cv2.imread(image_path)
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (250,250))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)

        if len(rects) == 0:
            Records.append(img)

        for rect in rects:
            count+=1
            faceAligned = fa.align(image, gray, rect)
            cv2.imwrite(os.path.join(destination_path,str(count)+".jpg"), faceAligned)

    for i in Records:
        count+=1
        image = cv2.imread(os.path.join(source_path,i))
        image = cv2.resize(image,(100,100))
        cv2.imwrite(os.path.join(destination_path,str(count)+".jpg"),image)

    return destination_path
    
CNNModelChkptPath = 'Models/runs/Models/checkpoints/' 
CNNModel = 'model-180.meta'


def cnn_model_results(path):
    X = []
    for img in os.listdir(path):
        new_data = cv2.imread(os.path.join(path,img))
        new_data = cv2.resize(new_data, (100,100))
        gray_scale = cv2.cvtColor(new_data, cv2.COLOR_BGR2GRAY)
        gray_scale = gray_scale.reshape((100,100,1))
        X.append(gray_scale)
    X = np.array(X,dtype=np.float32)

    checkpoint_file = tf.train.latest_checkpoint(CNNModelChkptPath)
    graph=tf.Graph()
    with graph.as_default():
        print(checkpoint_file)
        session_conf = tf.ConfigProto(log_device_placement =False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            saver =  tf.train.import_meta_graph(os.path.join(CNNModelChkptPath,CNNModel))
            saver.restore(sess,checkpoint_file)
            input = graph.get_operation_by_name("input_x").outputs[0]
            prediction=graph.get_operation_by_name("output/predictions").outputs[0]
            probabilities = graph.get_operation_by_name("output/softmax_outputs").outputs
            newdata=X
            my_probabilities = sess.run(probabilities,feed_dict={input:newdata})
            my_predictions = sess.run(prediction, feed_dict={input:newdata})

    return my_probabilities, my_predictions



def face_finder(path):
    Narendra_Modi ="No"
    Arvind_Kejriwal = "No"
    Faces = "No"
    #results path of processed image
    main_image_path, rects_path = find_faces_and_rects(path)
    aligned_image_path = align_face(rects_path)
    probabilities, predictions = cnn_model_results(aligned_image_path)

    if len(os.listdir(rects_path)) >0:
        Faces = "Yes"
    if 0 in predictions:
        Arvind_Kejriwal = "Yes"

    if 1 in predictions:
        Narendra_Modi = "Yes"
    filename = os.listdir(main_image_path)
    return {"name":'1.jpg',"Namo":Narendra_Modi,"Arke":Arvind_Kejriwal,"Face":Faces}


apps.run(debug=True)
