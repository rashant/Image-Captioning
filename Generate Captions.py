import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import load_img, img_to_array, pad_sequences
from nltk.translate.bleu_score import corpus_bleu

working_path = 'Model Data'

with open(os.path.join(working_path, 'mapping.pkl'), 'rb') as f:
    mapping = pickle.load(f)

with open(os.path.join(working_path, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = str(captions[i])
            caption = caption.lower()
            caption = caption.replace(r'[^A-Z]a-z', ' ')
            caption = caption.replace('/s+', ' ')
            caption = caption.strip()
            caption = 'sss ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' eee'
            captions[i] = caption


clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

model = load_model(os.path.join(working_path, 'model.h5'))
print(model.summary())


actual, predicted = list(), list()


max_length = max(len(caption.split()) for caption in all_captions)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'sss'
    end_text = ' eee'
    for i in range(max_length):
        print(i)
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        print(max_length)
        print(np.array(image).shape)
        print(np.array(sequence).shape)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        print(yhat)
        word = idx_to_word(yhat, tokenizer)
        print(word)
        if word is None:
            break
        in_text += " " + word
        if word == 'eee':
            break
    print(in_text)
    return in_text


def generate_caption(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    imagex = preprocess_input(image)
    image_model = VGG16()
    transfer_layer = image_model.get_layer('fc2')
    vg = Model(inputs=image_model.input, outputs=transfer_layer.output)
    features = vg.predict(imagex, verbose=0)
    y_pred = predict_caption(model, features, tokenizer, max_length)
    print(y_pred)
    plt.imshow(image)


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

for key in test:
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()

    actual.append(actual_captions)
    predicted.append(y_pred)
print("BLEU: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))