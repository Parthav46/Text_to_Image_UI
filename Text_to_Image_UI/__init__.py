import os
from django.core.cache import cache

# setting up Environment
image_dir = os.path.join('./static/images')
if not os.path.exists(image_dir):
	os.mkdir(image_dir)

# clearing Cache
cache.clear()
