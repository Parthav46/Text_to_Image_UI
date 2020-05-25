import tensorflow as tf
import numpy as np
import json

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

def index(request):
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.load_weights('static/sample.h5')
    val = model.predict([[float(request.GET['a']), float(request.GET['b'])]])
    context = {
        'ans': val
    }
    return render(request, 'index.html', context)

@csrf_exempt
def mnist(request):
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=(28,28)),
		tf.keras.layers.Dense(196, activation='relu'),
		tf.keras.layers.Dense(49, activation='relu'),
		tf.keras.layers.Dense(10, activation='softmax')
	])
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.load_weights('static/mnist.h5')
	data = json.loads(request.body)['data']
	val = np.argmax(model.predict([data]))
	context = {
		'ans': val
	}
	return render(request, 'index.html', context)
