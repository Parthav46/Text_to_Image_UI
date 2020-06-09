import pickle
from tensorflow.python.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch # >= tf 1.15
from model.networks import *
import cv2
import os
import time

def imsave(image, size, path):
    image = ((image+1.) / 2) * 255.0
    image = merge(image, size)
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def get_path():
    img_dir = './static/images'
    num = 0
    for f in os.listdir(img_dir):
        val = int(f[f.index('_') + 1: f.index('.')])
        if val > num: num = val
    return os.path.join(img_dir, "img_{}.jpg".format(num+1)), num+1

class Text_to_Image():
    def __init__(self):
        self.checkpoint_dir = './checkpoint'
        self.word_to_idx = pickle.load(open('./static/id_map.pickle', 'rb'))
        self.rnn_encoder = RnnEncoder(n_words=len(self.word_to_idx), embed_dim=256,
                            drop_rate=0.5, n_hidden=128, n_layer=1,
                            bidirectional=True, rnn_type='lstm')
        self.ca_net = CA_NET(c_dim=100)
        self.generator = Generator(channels=32)

        self.ckpt = tf.train.Checkpoint(rnn_encoder=self.rnn_encoder,
                                ca_net = self.ca_net,
                                generator=self.generator)
        
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print('Checkpoint loaded')
        else:
            assert('Checkpoint error')

    def extract_caption(self, text, max_length):
        caption = [self.word_to_idx.get(word, 2) for word in text.split()]
        caption = caption[:max_length]
        caption = caption + [2] * (max_length - len(caption))

        return tf.reshape(tf.constant([caption, caption]), [2,max_length])

    def process(self, text):
        caption = self.extract_caption(text, 18)
        word_emb, sent_emb, mask = self.rnn_encoder(caption, training=False)
        c_code, mu, logvar = self.ca_net(sent_emb, training=False)

        z = tf.random.normal(shape=[2, 100])
        fake_imgs = self.generator([c_code, z, word_emb, mask], training=True)
        fake = np.expand_dims(fake_imgs[-1][0], axis=0)
        path = get_path()

        return imsave(fake, [1, 1], path[0]), path[1]


class RandomCaption():
    def __init__(self):
        data = pickle.load(open('./static/captions.pickle', 'rb'))
        self.captions = data[0]
        self.idx_to_word = data[1]
    
    def get(self):
        pos = int(time.time() * 1000) % len(self.captions)
        caption = self.captions[pos]
        return ' '.join([self.idx_to_word[i] for i in caption])


# Test program
if __name__ == "__main__":
    text = 'a small yellow bird with black beak' # input text here
    t2i = Text_to_Image()
    im = t2i.process(text)
    print(im)