from keras.models import load_model
import numpy as np
import pyAesCrypt


bufferSize = 64 * 1024
password = "hello"
# decrypt
pyAesCrypt.decryptFile("./weights/encoder_weights.txt.aes","./weights/encoder_weights.h5", password, bufferSize)
encoder = load_model(r'./weights/encoder_weights.h5')
#decoder = load_model(r'./weights/coder_weights.h5')



inputs = np.array([[1,2,2,3,1]])

x = encoder.predict(inputs)
np.save('encodedValues',x)
#pyAesCrypt.encryptFile("encodedValues.txt","encodedValues.txt,aes", password, bufferSize)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(x))
#Sprint('Decoded: {}'.format(y))