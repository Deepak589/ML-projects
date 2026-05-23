from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Implementation of Deep Learning Based Neural Network Algorithm for Intracranial Hemorrhage Detection")
main.geometry("1200x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global model

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    
def preprocess():
    global X_train, X_test, y_train, y_test
    global filename
    global X, Y
    text.delete('1.0', END)
    X = np.load('yolomodel/X.txt.npy')
    Y = np.load('yolomodel/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split details\n\n")
    text.insert(END,"80% images used to train YOLO model: "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test YOLO model : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    test = cv2.resize(test, (200,200))
    cv2.imshow("Process Sampled Image",test)
    cv2.waitKey(0)

def trainYolo():
    global model
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('yolomodel/yolomodel.json'):
        with open('yolomodel/yolomodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("yolomodel/yolomodel_weights.h5")
        model._make_predict_function()      
    else:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 256, activation = 'relu'))
        model.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        model.save_weights('yolomodel/yolomodel_weights.h5')            
        model_json = model.to_json()
        with open("yolomodel/yolomodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('yolomodel/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    for i in range(0,6):
        predict[i] = 0
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'Yolo Model Accuracy  : '+str(a)+"\n")
    text.insert(END,'Yolo Model Precision : '+str(p)+"\n")
    text.insert(END,'Yolo Model Recall    : '+str(r)+"\n")
    text.insert(END,'Yolo Model FMeasure  : '+str(f)+"\n\n")
    text.update_idletasks()
    LABELS = ['Normal','Hemorrhage']
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Yolo Model Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def graph():
    f = open('yolomodel/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'blue')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Yolo Model Accuracy & Loss Graph')
    plt.show()

def predict():
    global model
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)

    if predict == 0:
        img = cv2.imread(filename)
        img = cv2.resize(img, (400,400))
        cv2.putText(img, 'No Hemorrhage Detected', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('No Hemorrhage Detected', img)
        cv2.waitKey(0)
    if predict == 1:
        img = cv2.imread(filename)
        img = cv2.resize(img, (400,400))
        cv2.putText(img, 'Hemorrhage Detected', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Hemorrhage Detected', img)
        cv2.waitKey(0)

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Implementation of Deep Learning Based Neural Network Algorithm for Intracranial Hemorrhage Detection')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload CT Scans Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=600,y=100)

processButton = Button(main, text="Normalize CT Scans Images", command=preprocess)
processButton.place(x=350,y=100)
processButton.config(font=font1)

yoloButton = Button(main, text="Train Yolo Model", command=trainYolo)
yoloButton.place(x=50,y=150)
yoloButton.config(font=font1)

graphButton = Button(main, text="Yolo Accuracy-Loss Graph", command=graph)
graphButton.place(x=350,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Hemorrhage from Test Image", command=predict)
predictButton.place(x=50,y=200)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=350,y=200)
exitButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
