import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('mnist.h5')

a = np.ones([300, 300], dtype='uint8') * 0;

# cv2.rectangle(a, (50, 50), (350, 350), (0, 255, 0), -2)

wname = 'Digit Recognition - Canvas'

state = False

cv2.namedWindow(wname)

def number(event, x, y, state, param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        state = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if state == True:
            cv2.circle(a, (x, y), 5, (255, 255, 255), -5)
    
    elif event == cv2.EVENT_LBUTTONUP:
        state = False
        
cv2.setMouseCallback(wname, number)

while True:
    cv2.imshow(wname, a)
    key = cv2.waitKey(1)
    
    if key == 27:
        break
    
    elif key == ord('p'):
        num = a[0:300, 0:300]
        # cv2.imshow("number", num)
        num = cv2.resize(num, (28, 28)).reshape(1, 28, 28)
        num = num/255
        print(np.argmax(model.predict(num)))
                
    elif key == ord('c'):
        a[:, :] = 0
        # cv2.rectangle(a, (50, 50), (350, 350), (0, 255, 0), -2)
        
cv2.destroyAllWindows()
        
