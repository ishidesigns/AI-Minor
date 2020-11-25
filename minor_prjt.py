import numpy as np
# !pip install tensorflow
import tensorflow as tf
import cv2

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path = 'mnist.npz')
model = tf.keras.models.load_model('mnist.h5')

a = np.zeros([400, 400, 3], dtype='uint8') * 255
cv2.rectangle(a, (50, 50), (350, 350), (0, 255, 0), -5)

wname = 'Digit Recognition - Canvas'

state = False

cv2.namedWindow(wname)

def number(event, x, y, flags, param):
    global state
    
    if event == cv2.EVENT_LBUTTONDOWN:
        state = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if state == True:
            cv2.rectangle(a, (x, y), (x+10, y+10), (0, 0, 0), -5)
    
    elif event == cv2.EVENT_LBUTTONUP:
        state = False
        
cv2.setMouseCallback(wname, number)

while True:
    cv2.imshow(wname, a)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    elif key == ord('p'):
        num = a[50:350, 50:350]
        cv2.imshow("number", num)
        num = cv2.resize(num, (28, 28)).reshape(1, 28, 28)
        print(np.argmax(model.predict(num)))
                
    elif key == ord('c'):
        a[:, :] = 0
        cv2.rectangle(a, (50, 50), (350, 350), (0, 255, 0), -5)
        
cv2.destroyAllWindows()
        
