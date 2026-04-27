from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import mediapipe as mp
import numpy as np
import keras
import cv2
from time import time

"""Functions that are not about Tkinter"""

def bound(x, mini, maxi):
    return min(max(x, mini), maxi)

def mean(x, y):
    return (x + y)//2

def if_float(x, *args):
    n = len(args)
    for i in range(n-1, 0, -1):
        if x > i/n:
            return args[i]
    return args[0]

# Takes a float (0 for extension and 1 for flexion) and returns a string like 'flex4'
def get_extension_or_flexion_level(x):
    return if_float(x, 'extension4', 'extension3', 'extension2', 'extension1', 'reference', 'flexion1', 'flexion2', 'flexion3', 'flexion4')

# Takes a float (0 for abduction/open and 1 for adduction/closed) and returns a string like 'abd3'
def get_abduction_or_adduction_level(x):
    return if_float(x, 'abduction4', 'abduction3', 'abduction2', 'abduction1', 'reference', 'adduction1', 'adduction2', 'adduction3', 'adduction4')

# Tells how much it is opposed or not
def get_intern_or_extern_rotation_level(x):
    return if_float(x, 'intern4', 'intern3', 'intern2', 'intern1', 'reference', 'extern1', 'extern2', 'extern3', 'extern4')

def to_typannot(symbol):
    dico = {
        'reference': '',
        'flexion': '',
        'extension': '',
        'abduction': '',
        'adduction': '',
        'intern': '',
        'extern': '',
        'phalanx1': '',
        'phalanx2': '',
        'thumb': '',
        'index finger': '',
        'middle finger': '',
        'ring finger': '',
        'pinky': '',
        '1': '',
        '2': '',
        '3': '',
        '4': ''
    }

    symbol = ''.join(symbol)

    for key in dico:
        symbol = symbol.replace(key, dico[key])
    
    return symbol

def prediction_to_symbol(prediction):
    symbol = []
    """Thumb"""
    symbol.append('thumb')
    symbol.append('phalanx1')
    symbol.append(get_intern_or_extern_rotation_level(prediction[10]))
    symbol.append(get_abduction_or_adduction_level(prediction[9]))
    symbol.append('phalanx2')
    symbol.append(get_extension_or_flexion_level(prediction[8]))
    """Index finger"""
    symbol.append('index finger')
    symbol.append('phalanx1')
    symbol.append(get_extension_or_flexion_level(prediction[7]))
    symbol.append('phalanx2')
    symbol.append(get_extension_or_flexion_level(prediction[6]))
    """Middle finger"""
    symbol.append('middle finger')
    symbol.append('phalanx1')
    symbol.append(get_extension_or_flexion_level(prediction[5]))
    symbol.append('phalanx2')
    symbol.append(get_extension_or_flexion_level(prediction[4]))
    """Ring finger"""
    symbol.append('ring finger')
    symbol.append('phalanx1')
    symbol.append(get_extension_or_flexion_level(prediction[3]))
    symbol.append('phalanx2')
    symbol.append(get_extension_or_flexion_level(prediction[2]))
    """Pinky"""
    symbol.append('pinky')
    symbol.append('phalanx1')
    symbol.append(get_extension_or_flexion_level(prediction[1]))
    symbol.append('phalanx2')
    symbol.append(get_extension_or_flexion_level(prediction[0]))
    
    return to_typannot(symbol)

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

# Prints the result of the model on the given image (as an array)
def use_mnist(img):
    # We convert the image to MNIST
    mnist_img = img_to_mnist(img)
    # If a hand was detected
    if mnist_img is not None:
        # We apply our model to the image
        output = model(mnist_img)[0]
        # We convert our prediction to symbols
        symbols = prediction_to_symbol(output)
        previous_symbols = text_label.cget('text')
        if previous_symbols == '' or sum(s1 != s2 for s1, s2 in zip(symbols, previous_symbols)) >= 5:
            print(symbols, file=transcript_fd)
            text_label.config(text=symbols)


def tick():
    global capture, out, transcript_fd
    """About taking a caption and saving it"""
    # We take a caption
    ret, img = capture.read()
    # If this is the end of the video, we close the capture and transcript
    if not ret:
        capture.release()
        capture = None
        transcript_fd.close()
        transcript_fd = None
        return
    # We save the caption if necessary
    if out:
        out.write(img)
    """About showing the caption on screen"""
    # We convert it to RGB and keep an array version of it
    array_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We convert it into an Image
    img = Image.fromarray(array_img)
    # We resize the image
    w, h = img.size
    new_w = 500
    new_h = int(h * (new_w/w))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # We convert it to an ImageTk and change the panel image
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    """About transcribing the model"""
    use_mnist(array_img)

def loop():
    # If the camera is open, we tick and we loop
    if capture:
        panel.after(1, tick)
        panel.after(30, loop)

def switch_recording():
    global capture, out, transcript_fd
    if capture:
        # We stop the loop
        record_btn.config(text='Start recording')
        capture.release()
        capture = None
        out.release()
        out = None
        transcript_fd.close()
        transcript_fd = None
    else:
        # We start the camera, the recording and the transcript file
        record_btn.config(text='Stop recording')
        capture = cv2.VideoCapture(0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height))
        transcript_fd = open('output.txt', 'w', encoding='utf-8')
        # We start the loop
        loop()

def import_video():
    global capture, transcript_fd
    filename = filedialog.askopenfilename(title='open')
    if filename is not None:
        capture = cv2.VideoCapture(filename)
        transcript_fd = open('output.txt', 'w', encoding='utf-8')
        loop()
            
            


root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

# Global variables for the camera, the video being recorded and the transcript text file
capture = None
out = None
transcript_fd = None

# We import the MNIST model
model = keras.models.load_model('model.keras')
# We import the MediaPipe model
base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# We create the window
panel = Label(root); panel.pack()
text_label = Label(root, font=('TYPANNOT Beta Generics-Postural_Release_v3',25)); text_label.pack()
record_btn = Button(root, text='Start recording', command=switch_recording); record_btn.pack()
video_btn = Button(root, text='Import video', command=import_video); video_btn.pack()
root.mainloop()



