import os
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input, Embedding, LSTM, Dropout, add
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import load_img, img_to_array, pad_sequences, to_categorical

image_path = 'Images'
working_path = 'Model Data'
logs_dir = 'Logs'

r'''.\venv\Scripts\activate'''
r'''tensorboard --logdir=.'''

image_model = VGG16()

transfer_layer = image_model.get_layer('fc2')

model = Model(inputs=image_model.input, outputs=transfer_layer.output)
print(model.summary())

print("Extracting features")

# features = {}
# count=0
# for image_name in os.listdir(image_path):
#     print(f"{count} / {len(os.listdir(image_path))}")
#     img_path = os.path.join(image_path, image_name)
#     image = load_img(img_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     feature = model.predict(image, verbose=0)
#     image_id = image_name.split('.')[0]
#     features[image_id] = feature
#     count+=1
#
# pickle.dump(features, open(os.path.join(working_path, 'features.pkl'), 'wb'))
#
# print("Features saved")

with open(os.path.join(working_path, 'features.pkl'), 'rb') as f:
    features = pickle.load(f, encoding='utf-8')

with open(os.path.join('captions.txt'), 'r') as f:
    next(f)
    caption_doc = f.read()

print("Captions loaded")

# mapping = {}
# for line in caption_doc.split('\n'):
#     tokens = line.split(',')
#     if len(line) < 2:
#         continue
#     image_id, caption = tokens[0].split('.')[0], tokens[1:]
#     caption = ' '.join(caption)
#     if image_id not in mapping:
#         mapping[image_id] = []
#     mapping[image_id].append(caption)
#
# pickle.dump(mapping, open(os.path.join(working_path, 'mapping.pkl'), 'wb'))
#
# print("Mapping saved")

with open(os.path.join(working_path, 'mapping.pkl'), 'rb') as f:
    mapping = pickle.load(f, encoding='utf-8')

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

print("Cleaned mapping")

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

print("All captions loaded")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    x1, x2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    x1.append(features[key][0])
                    x2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                x1 = np.array(x1)
                x2 = np.array(x2)
                y = np.array(y)
                print(x1.shape)
                print(x2.shape)
                yield [x1, x2], y
                x1, x2, y = list(), list(), list()
                n = 0



# ENCODER
# IMAGE FEATURE LAYERS
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# DECODER
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 20
batch_size = 64
steps = len(train) // batch_size

callback = TensorBoard(log_dir=logs_dir)
# print("Starting training")
# for i in range(epochs):
#     generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#
# model.save(os.path.join(working_path, 'Model Data/model.h5'))
