"""
import tkinter as tk

root = tk.Tk()
image = tk.PhotoImage(file="rain.gif")

def turn_tv_on():
    window = tk.Toplevel(root)
    window.title("TV")
    original_image = tk.Label(window, image=image)
    original_image.pack()

def volume_up():
    print("Volume Increase +1")

def volume_down():
    print("Volume Decrease -1")

turn_on = tk.Button(root, text="ON", command=turn_tv_on)
turn_on.pack()

turn_off = tk.Button(root, text="OFF", command=root.destroy)
turn_off.pack()

volume = tk.Label(root, text="VOLUME")
volume.pack()

vol_up = tk.Button(root, text="+", command=volume_up)
vol_up.pack()

vol_down = tk.Button(root, text="-", command=volume_down)
vol_down.pack()


root.mainloop()
"""




from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import mediapipe as mp
import numpy as np
import keras
import cv2

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

# Returns a file name that is chosen by the user
def openfn():
    return filedialog.askopenfilename(title='open')

# Action being made when clicking the button 'open image'
def open_img():
    global global_img
    # We select the filename
    filename = openfn()
    if filename:
        # We open the image and convert it to RGB
        img = Image.open(filename).convert("RGB")
        # We store its numpy array into a global variable
        global_img = np.array(img)
        # We resize the image
        w, h = img.size
        new_w = 500
        new_h = int(h * (new_w/w))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # We convert it to an ImageTk and change the panel image
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

def take_screenshot():
    global global_img
    # We take a caption, flip it and convert it to RGB
    capture = cv2.VideoCapture(0)
    _, img = capture.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We store its numpy array into a global variable
    global_img = img
    # We convert the caption into an Image
    img = Image.fromarray(img)
    # We resize the image
    w, h = img.size
    new_w = 500
    new_h = int(h * (new_w/w))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # We convert it to an ImageTk and change the panel image
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

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

def use_mnist():
    img = np.copy(global_img)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=global_img))
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
        mnist_img = np.reshape(cv2.resize(img, (28, 28)), (1, 28, 28, 1))/255
        # We apply our model to the image
        output = model(mnist_img)[0]
        output = np.round(output, 3)
        # We convert our prediction to symbols
        print(prediction_to_symbol(output))


# We create a global variable containing an array representing the picture we are using
global_img = None

# We import the MNIST model
model = keras.models.load_model('model.keras')
# We import the MediaPipe model
base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# We create the window
panel = Label(root); panel.pack()
open_img_btn = Button(root, text='Open image', command=open_img).pack()
screenshot_btn = Button(root, text='Screenshot from camera', command=take_screenshot).pack()
mnist_model = Button(root, text='Use MNIST model', command=use_mnist).pack()





root.mainloop()



