# python -m venv my-venv
# my-venv/Scripts/pip install mediapipe opencv-python matplotlib
# my-venv/Scripts/python main.py
# Download the file from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task


# Code inspired by the MediaPipe Task documentation:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# https://ai.google.dev/edge/api/mediapipe/python/mp/Image
# And also by this video:
# https://www.youtube.com/watch?v=RRBXVu5UE-U
# And this article:
# https://medium.com/@odil.tokhirov/how-i-built-a-hand-gesture-recognition-model-in-python-part-1-db378cf196e6


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import copy


# Takes an image and the hand landmarks (a numpy array of size nb_hands * 21 * 3), and draws the landmarks
def draw_hands(img, hands):
    neighbours = {
        0: [1, 5, 17],
        1: [2],
        2: [3],
        3: [4],
        4: [],
        5: [6, 9],
        6: [7],
        7: [8],
        8: [],
        9: [10, 13],
        10: [11],
        11: [12],
        12: [],
        13: [14, 17],
        14: [15],
        15: [16],
        16: [],
        17: [18],
        18: [19],
        19: [20],
        20: [],
    }
    
    # If there are hands being detected
    if hands.size != 0:
        # We drop the z axis and convert to integer
        hands2D = np.delete(hands, -1, axis=2).astype(int)
        # For each hand
        for hand in hands2D:
            # For each point
            for i, point in enumerate(hand):
                # We draw a line between the point and all its neighbours
                for j in neighbours[i]:
                    cv2.line(img, point, hand[j], color=(0, 0, 255), thickness=3)
                # We draw the point
                cv2.circle(img, center=point, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Result', img)


# Takes the name of an image, and shows the image with hands being detected
def from_image(filename):
    # We create a HandLandmarker object
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # We load the image as a numpy array containing the BGR values
    img = cv2.imread(filename)

    # We convert it to an mp.Image object containing the RGB values
    rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # We detect the points (landmarks) from the input image
    res = detector.detect(rgb)
    
    # We convert the result into a numpy array of size nb_hands * 21 * 3
    h, w, _ = img.shape
    hands = np.array([[[point.x*w, point.y*h, point.z] for point in hand] for hand in res.hand_landmarks]) 

    draw_hands(img, hands)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Shows the camera with hands being detected
def from_camera():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # We use the first camera
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        # We take a frame from the camera
        _, img = capture.read()
        # We flip the image so that it acts like a mirror
        img = cv2.flip(img, 1)
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = detector.detect(rgb)
        h, w, _ = img.shape
        hands = np.array([[[point.x*w, point.y*h, point.z] for point in hand] for hand in res.hand_landmarks])
        draw_hands(img, hands)
        cv2.waitKey(1)
    
    capture.release()
    cv2.destroyAllWindows()


# Returns an image corresponding to a grid
def get_grid_image(cell_size, number_cells, color=(0, 106, 0)):
    img = np.zeros((cell_size*number_cells, cell_size*number_cells, 3), np.uint8)
    for i in range(number_cells + 1):
        cv2.line(img, (cell_size*i, 0), (cell_size*i, cell_size*number_cells), color=color, thickness=1)
        cv2.line(img, (0, cell_size*i), (cell_size*number_cells, cell_size*i), color=color, thickness=1)
    return img


def draw_normalized_hands(hands, cell_size=128, number_cells=3):
    neighbours = {
        0: [1, 5, 17],
        1: [2],
        2: [3],
        3: [4],
        4: [],
        5: [6, 9],
        6: [7],
        7: [8],
        8: [],
        9: [10, 13],
        10: [11],
        11: [12],
        12: [],
        13: [14, 17],
        14: [15],
        15: [16],
        16: [],
        17: [18],
        18: [19],
        19: [20],
        20: [],
    }
    
    # We create a grid
    img = get_grid_image(cell_size, number_cells)
    if hands.size != 0:
        # We drop the z axis, center the hands on the grid, scale them so that each cell corresponds to a unit of 1, and convert to integer
        hands2D = np.array([[(point[0]*cell_size + cell_size*number_cells/2, point[1]*cell_size + cell_size*number_cells/2) for point in hand] for hand in hands]).astype(int)
        # We draw the lines and points
        for hand in hands2D:
            for i, point in enumerate(hand):
                for j in neighbours[i]:
                    cv2.line(img, point, hand[j], color=(0, 0, 255), thickness=3)
                cv2.circle(img, center=point, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Result', img)


# Normalizes the hand landmarks. "hands" is a nb_hands * 21 * 3 array, and handedness tells for each hand whether it is left or right.
def normalize(hands, handedness):
    if hands.size != 0:
        # We center each hand around 0 by subtracting each hand by their mean.
        hands -= hands.mean(axis=1, keepdims=True)
        # For each hand, we divide by its invariant (for instance the distance between point 0 and 5)
        for i in range(len(hands)):
            # We divide by the invariant
            invariant1 = np.linalg.norm(hands[i][5]-hands[i][0])
            invariant2 = np.linalg.norm(hands[i][17]-hands[i][0])
            hands[i] /= np.mean([invariant1, invariant2])
            # We transform the left hands into a right hands (we flip the x axis)
            if handedness[i]:
                hands[i, :, 0] *= -1
            # We compute the normal vector of the palm of the hand
            A, B, C = hands[i][0], hands[i][5], hands[i][17]
            normal_vector = np.cross((B - A), (C - A))
            target_vector = np.array([0, 0, -1])
            # We compute the cos(rotation angle) and the rotation axis
            cos_rotation_angle = np.arccos(normal_vector @ target_vector)
            rotation_axis = np.cross(normal_vector, target_vector)
            # We compute the skew-symmetric cross-product matrix
            v = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                          [rotation_axis[2], 0, -rotation_axis[0]],
                          [-rotation_axis[1], rotation_axis[0], 0]
                        ])
            # We compute the rotation matrix
            rotation_matrix = np.eye(3) + v + v @ v * (1 / (1 + cos_rotation_angle))
            print(rotation_matrix)
            # We apply the rotation matrix on all our points
            hands[i] = hands[i] @ rotation_matrix
    return hands


def from_camera_normalized():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        _, img = capture.read()
        img = cv2.flip(img, 1)
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = detector.detect(rgb)
        h, w, _ = img.shape
        hands = np.array([[[point.x*w, point.y*h, point.z] for point in hand] for hand in res.hand_landmarks])
        handedness = [hand[0].index for hand in res.handedness]
        # We normalize it
        hands = normalize(hands, handedness)
        # We draw the normalized result on a black screen
        draw_normalized_hands(hands)
        cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()


# Takes three points and returns the angle
def angle(a, b, c):
    # We compute the vectors ba and bc
    ba, bc = a - b, c - b
    # Remember that u · v = |u| * |v| * cos(θ)
    # Meaning that cos(theta) = (u · v) / (|u| * |v|)
    # Therefore, theta = arccos((u · v) / (|u| * |v|))
    return np.degrees(np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))

def from_camera_with_3D():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    capture = cv2.VideoCapture(0)
    
    x, y, z = [], [], []
    _, img = capture.read()
    h, w, _ = img.shape
    
    plt.ion()
    ax = plt.figure().add_subplot(projection="3d")
    scatter = ax.scatter(x, y, z, s=100)
    ax.set_box_aspect([w, np.mean([w, h]), h])
    ax.set_xlim(0, w)
    ax.set_ylim(-np.mean([w, h]), 0)
    ax.set_zlim(-h, 0)
    plt.show()
    
    
    while capture.isOpened():
        _, img = capture.read()
        #img = cv2.flip(img, 1)
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = detector.detect(rgb)
        hands = np.array([[[point.x*w, point.y*h, point.z*np.mean([w, h])] for point in hand] for hand in res.hand_landmarks])
        draw_hands(img, hands)
        
        
        cv2.waitKey(1)
        
        if hands.size != 0:
            hand = copy.deepcopy(hands[0])
            x, y, z = hand.T[0], hand.T[2], -hand.T[1]
            #x, y, z = np.random.rand(300), np.random.rand(300), np.random.rand(300) * 0.1
        scatter._offsets3d = (x, y, z)
        plt.pause(0.01)
        
    capture.release()
    cv2.destroyAllWindows()



#from_image("image.png")
#from_camera()
from_camera_normalized()
#from_camera_with_3D()

"""
for i in range(1, 6):
    from_image(f"black-hand-{i}.png")
for i in range(1, 4):
    from_image(f"white-hand-{i}.png")
from_camera()
"""