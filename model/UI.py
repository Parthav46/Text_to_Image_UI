import pickle
import cv2
import os
import time
from model_torch.eval import *

def imsave(image, size, path):
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, image)


def get_path():
    img_dir = './static/images'
    num = 0
    for f in os.listdir(img_dir):
        val = int(f[f.index('_') + 1: f.index('.')])
        if val > num: num = val
    return os.path.join(img_dir, "img_{}.jpg".format(num+1)), num+1

class Text_to_Image():
    def __init__(self):
        self.wordtoix, self.ixtoword = word_index()
        self.text_encoder, self.netG = models(len(self.wordtoix))
    def process(self, text):
        fake = generate(text, self.wordtoix, self.ixtoword, self.text_encoder, self.netG)
        path = get_path()
        print('working')
        return imsave(fake, [1, 1], path[0]), path[1], path[0]


class RandomCaption():
    def __init__(self):
        data = pickle.load(open('./static/test-captions.pickle', 'rb'))
        self.captions = data[0]
        self.idx_to_word = data[1]
    
    def get(self):
        pos = int(time.time() * 1000) % len(self.captions)
        caption = self.captions[pos]
        return ' '.join([self.idx_to_word[i] for i in caption])
