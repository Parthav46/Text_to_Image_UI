import os

# setting up Environment
image_dir = os.path.join('./static/images')
if not os.path.exists(image_dir):
	os.mkdir(image_dir)
