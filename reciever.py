from keras.models import load_model
import numpy as np

import pyAesCrypt

bufferSize = 64 * 1024
password = "hello"
pyAesCrypt.decryptFile("./weights/decoder_weights.txt.aes","./weights/decoder_weights.h5", password, bufferSize)
decoder = load_model(r'./weights/decoder_weights.h5')

inputs = np.array(np.load('encodedValues.npy'))
y = decoder.predict(inputs)
test="testing reciever";
print(test);
print('Decoded: {}'.format(y))
print(np.round(y))
