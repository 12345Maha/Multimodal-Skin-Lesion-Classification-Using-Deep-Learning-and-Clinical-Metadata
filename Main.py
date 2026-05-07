#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import os
#loading resnet and attention model
from keras.applications.resnet50 import ResNet50, preprocess_input
from Attention import attention
from keras.preprocessing import image
from keras import layers, models, Input, Model
from keras.models import Sequential
from keras.layers import  MaxPooling2D, Conv2D, Flatten, Dense, Input, Concatenate, Dropout, RepeatVector
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv("Dataset/HAM/HAM10000_metadata.csv")
#applying onehot encoding on categorical data
image_id = dataset['image_id'].ravel()
ham_labels = np.unique(dataset['dx']).ravel()
dataset.drop(['lesion_id','image_id'], axis = 1,inplace=True)
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(dataset.mean(), inplace = True)
#normalizing metadata features
Y = dataset['dx'].ravel()
dataset.drop(['dx'], axis = 1,inplace=True)
X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
img = np.load("model/ham.npy")    
#shuffling images and metadata features
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices] #X contains metadata
Y = Y[indices] #Y contains labels
img = img[indices] #img contains image features
Y = to_categorical(Y)
data = np.load("model/ham_data.npy", allow_pickle=True)
X_img_train, X_img_val, X_meta_train, X_meta_val, y_trains, y_vals = data

def getModel():
    image_branch = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 32, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        RepeatVector(2),
        attention(return_sequences=True,name='imgattention'),
        Flatten(),
        Dense(64, activation='relu')
    ], name="Image_Branch")

    #defining metadata model with attention layer
    meta_branch = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        RepeatVector(2),
        attention(return_sequences=True,name='metaattention'),
        Flatten(),
        Dense(16, activation='relu')
    ], name="Metadata_Branch")

    #concatenate image and metadata model with attention layer
    combined_input = Concatenate()([image_branch.output, meta_branch.output])
    # Final dense layers to process merged data
    x = Dense(64, activation='relu')(combined_input)
    output = Dense(Y.shape[1], activation='softmax')(x) # softmax activation for classifictaion
    attention_model = Model(inputs=[image_branch.input, meta_branch.input], outputs=output)
    attention_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    attention_model.load_weights("model/attention_weights.hdf5")
    return attention_model

@app.route('/ClassifyAction', methods=['GET', 'POST'])
def classifyAction():
    if request.method == 'POST':
        global ham_labels, label_encoder, resnet_model, scaler
        dx_type = request.form['t1']
        age = request.form['t2']
        gender = request.form['t3']
        localization = request.form['t4']
        img = request.files['t5'].read()
        if os.path.exists("static/test.jpg"):
            os.remove("static/test.jpg")
        with open("static/test.jpg", "wb") as file:
            file.write(img)
        file.close()
        metadata = []
        metadata.append([dx_type, float(age), gender, localization])
        metadata = pd.DataFrame(metadata, columns=['dx_type','age','sex','localization'])
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            if le[0] != "dx":
                metadata[le[0]] = pd.Series(le[1].transform(metadata[le[0]].astype(str)))#encode all str columns to numeric
        metadata = metadata.values
        metadata = scaler.transform(metadata)
        imgs = []
        x = image.load_img("static/test.jpg", target_size=(64, 64))#loading image
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x)#process image as per resnet format
        imgs.append(x[0])
        imgs = np.asarray(imgs)
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        imgs = resnet_model.predict(imgs)#apply resnet to extract features
        imgs = np.reshape(imgs, (imgs.shape[0], 64, 32, 1))   
        attention_model = getModel()
        prob = attention_model.predict([imgs, metadata])
        predict = np.argmax(prob)
        print(predict)
        predict = ham_labels[predict]
        prob = round(np.amax(prob),3)
        print(prob)
        img = cv2.imread("static/test.jpg")
        img = cv2.resize(img, (600,400))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Predicted As : '+predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.putText(img, 'Probability    : '+str(prob)+"%", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        plt.imshow(img)   
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        return render_template('index.html', msg='Predicted As : <font size=4 color=blue>'+predict+"</font><br/>Predicted Probability : "+str(prob)+" %", img = img_b64)           

@app.route('/Predict', methods=['GET', 'POST'])
def predict():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

if __name__ == '__main__':
    app.run()    
