from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import mediapipe as mp
import numpy as np
import keras
import cv2
from time import time

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

"""Functions that are not about Tkinter"""

def bound(x, mini, maxi):
    return min(max(x, mini), maxi)

def mean(x, y):
    return (x + y)//2

def if_float(x, n, *args):
    for i in range(n-1, 0, -1):
        if x > i/n:
            return args[i]
    return args[0]

def prediction_to_symbol(prediction):
    print('test', prediction.shape)
    symbol = []
    # Pinky
    symbol.append('pinky')
    symbol.append(if_float(prediction[0], 3, 'flat', 'curved', 'bent'))
    symbol.append(if_float(prediction[1], 3, 'open', 'semiclosed', 'closed'))
    # Ring finger
    symbol.append('ring finger')
    symbol.append(if_float(prediction[2], 3, 'flat', 'curved', 'bent'))
    symbol.append(if_float(prediction[3], 3, 'open', 'semiclosed', 'closed'))
    # Middle finger
    symbol.append('middle finger')
    symbol.append(if_float(prediction[4], 3, 'flat', 'curved', 'bent'))
    symbol.append(if_float(prediction[5], 3, 'open', 'semiclosed', 'closed'))
    # Index finger
    symbol.append('index finger')
    symbol.append(if_float(prediction[6], 3, 'flat', 'curved', 'bent'))
    symbol.append(if_float(prediction[7], 3, 'open', 'semiclosed', 'closed'))
    # Thumb
    symbol.append('thumb')
    symbol.append(if_float(prediction[8], 3, 'flat', 'curved', 'bent'))
    symbol.append(if_float(prediction[9], 3, 'open', 'semiclosed', 'closed'))
    symbol.append(if_float(prediction[10], 2, 'not opposed', 'opposed'))
    symbol.append(if_float(prediction[11], 2, 'does not touch pinky', 'touches pinky'))
    symbol.append(if_float(prediction[12], 2, 'does not touch ring finger', 'touches ring finger'))
    symbol.append(if_float(prediction[13], 2, 'does not touch middle finger', 'touches middle finger'))
    symbol.append(if_float(prediction[14], 2, 'does not touch index finger', 'touches index finger'))
    return symbol

# Takes an image (a 3D RGB array) and returns an OpenCV MNIST image centered on the hand (and None if there is no hand)
def img_to_mnist(img):
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    h, w, _ = img.shape
    hands = np.array([[[point.x*w, point.y*h, point.z*np.mean([w, h])] for point in hand] for hand in res.hand_landmarks])
    if len(hands):
        # We get the x and y min and max.
        min_x = max(0, int(np.min(hands[:, :, 0])) - 30)
        max_x = max(0, int(np.max(hands[:, :, 0])) + 30)
        min_y = max(0, int(np.min(hands[:, :, 1])) - 30)
        max_y = max(0, int(np.max(hands[:, :, 1])) + 70)
        # We compute the side that should have the square around the hand
        side = max(max_x - min_x, max_y - min_y)
        # We compute the center of the hand, and we move it if it is too close to the border
        center_x, center_y = mean(min_x, max_x), mean(min_y, max_y)
        center_x, center_y = bound(center_x, side//2, w - side//2), bound(center_y, side//2, h - side//2)
        # We crop the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[(center_y - side//2):(center_y + side//2), (center_x - side//2):(center_x + side//2)]
        # We reduce the image to a 28 * 28 pixel one
        return np.reshape(cv2.resize(img, (28, 28)), (1, 28, 28, 1))/255
    

"""Tkinter functions"""

# Returns a file name that is chosen by the user
def openfn():
    return filedialog.askopenfilename(title='open')

# Prints the result of the model on the given image (as an array)
def use_mnist(img):
    # We convert the image to MNIST
    mnist_img = img_to_mnist(img)
    # If a hand was detected
    if mnist_img is not None:
        # We apply our model to the image
        output = model(mnist_img)[0]
        # We convert our prediction to symbols
        print(prediction_to_symbol(np.round(output, 3)))

def recording_loop():
    global global_img
    # If the camera is open
    if capture:
        """About recording the video"""
        # We take a caption and save it
        _, img = capture.read()
        out.write(img)
        # We convert it to RGB and keep an array version of it
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # We convert it into an Image
        img = Image.fromarray(img_array)
        # We resize the image
        w, h = img.size
        new_w = 500
        new_h = int(h * (new_w/w))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # We convert it to an ImageTk and change the panel image
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        # We repeat
        panel.after(30, recording_loop)
        """About transcribing the video"""
        use_mnist(img_array)

def switch_recording():
    global capture, out
    if capture:
        # We stop the recording
        record_button.config(text='Start recording')
        capture.release()
        out.release()
        capture = None
    else:
        # We start the recording
        record_button.config(text='Stop recording')
        capture = cv2.VideoCapture(0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height))
        # We start the loop
        recording_loop()

# Global variables for the image that is on the screen, the camera and the video being recorded
global_img = None
capture = None
out = None

# We import the MNIST model
model = keras.models.load_model('model.keras')
# We import the MediaPipe model
base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


# We create the window
panel = Label(root); panel.pack()
record_button = Button(root, text='Start recording', command=switch_recording); record_button.pack()
root.mainloop()



