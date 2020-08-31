import tensorflow as tf 
import numpy as np
import random 
from flask import Flask , request
import json

# def training_the_model():
#     (x_train , y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
#     x_train=x_train.reshape((len(x_train),-1))
#     x_test=x_test.reshape((len(x_test),-1))
#     x_train=x_train/255.0
#     x_test=x_test/255.0
#     model=tf.keras.models.Sequential(
#         [
#         tf.keras.layers.Dense(32,activation='sigmoid',input_shape=(784,))
#         ,
#         tf.keras.layers.Dense(32,activation='sigmoid')
#         ,
#         tf.keras.layers.Dense(10,activation='softmax')
#         ]
#     )
#     model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
#     history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=2048,verbose=2)
#     model.save('model.h5')

model=tf.keras.models.load_model('model.h5')
feature_model=tf.keras.models.Model(model.inputs,[layer.output for layer in model.layers])
_,(x_test,_)=tf.keras.datasets.mnist.load_data()
x_test=x_test/255.0

def get_prediction():
    index =np.random.choice(x_test.shape[0])
    image=x_test[index,:,:]
    image_arr=np.reshape(image,(1,784))
    return feature_model.predict(image_arr) , image
    
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])

def index():
    if request.method=='POST':
        preds,image=get_prediction()
        final_preds=[p.tolist() for p in preds]
        return json.dumps(
            {
                "prediction":final_preds ,
                "image":image.tolist()
            }
        )
    return "Welcome to NN-Visualizer"

if __name__ == '__main__':
    app.run(debug=True)