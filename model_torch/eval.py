from __future__ import print_function

import os
import sys
import torch
import io
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
import cv2
from model_torch.miscc.utils import build_super_images2
from model_torch.model import RNN_ENCODER, G_NET
import pickle

def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    return captions.astype(int), cap_lens.astype(int)

def generate(caption, wordtoix, ixtoword, text_encoder, netG, copies=2):
    # load word vector
    captions, cap_lens  = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)

    # only one to generate
    batch_size = captions.shape[0]

    nz = 100
    with torch.no_grad():
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))
        noise = Variable(torch.FloatTensor(batch_size, nz))

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = (captions == 0)


    #######################################################
    # (2) Generate fake images
    #######################################################
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    # only look at first one
    #j = 0
    im = fake_imgs[1][0].data.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    
    return im

def word_index():
    #print("ix and word not cached")
    # load word to index dictionary
    x = pickle.load(open('./static/captions.pickle', 'rb'))
    ixtoword = x[2]
    wordtoix = x[3]
    del x

    return wordtoix, ixtoword

def models(word_len):
    #print(word_len)
    #print("text_encoder not cached")
    text_encoder = RNN_ENCODER(word_len, nhidden=256)
    state_dict = torch.load('./static/text_encoder200.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()

    #print("netG not cached")
    netG = G_NET()
    state_dict = torch.load('./static/bird_AttnGAN2.pth', map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.eval()

    return text_encoder, netG

if __name__ == "__main__":
    caption = "the bird has a blue crown and a black eyering that is round"

    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))

    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG)
    print(urls)
