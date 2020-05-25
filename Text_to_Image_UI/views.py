# import tensorflow as tf
# import numpy as np
from django.shortcuts import render

def index(request):
    # model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
    # model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.load_weights('static/sample.h5')
    # val = model.predict([[float(request.GET['a']), float(request.GET['b'])]])
    context = {
        'ans': float(request.GET['a']) + float(request.GET['b'])
    }
    return render(request, 'index.html', context)