# python -m venv my-venv
# my-venv/Scripts/pip install mediapipe opencv-python matplotlib pandas torch tensorflow scikit-learn seaborn
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
import pandas as pd
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

# Takes a matrix
def from_matrix(X):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    print(type(X[0, 0, 0].astype(np.uint8)))
    # We repeat the last axis three times so that it becomes RGB
    img = X.repeat(3, -1).astype(np.uint8)
    rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    res = detector.detect(rgb)
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
            normal_vector /= np.linalg.norm(normal_vector)
            target_vector = np.array([0., 0., -1.])
            # We compute the cos(rotation angle) and the rotation axis
            cos_rotation_angle = normal_vector @ target_vector
            rotation_axis = np.cross(normal_vector, target_vector)
            # We compute the skew-symmetric cross-product matrix
            skew_symmetric_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                              [rotation_axis[2], 0, -rotation_axis[0]],
                                              [-rotation_axis[1], rotation_axis[0], 0]
            ])
            # We compute the rotation matrix
            rotation_matrix = np.eye(3) + skew_symmetric_matrix + skew_symmetric_matrix @ skew_symmetric_matrix / (1 + cos_rotation_angle)
            # We apply the rotation matrix on all our points
            hands[i] = hands[i] @ rotation_matrix.T
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
        hands = np.array([[[point.x*w, point.y*h, point.z*np.mean([w, h])] for point in hand] for hand in res.hand_landmarks])
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


from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


train, test = torch.tensor(pd.read_csv('sign_mnist_train.csv').values), torch.tensor(pd.read_csv('sign_mnist_test.csv').values)
x_train, y_train = train[:, 1:]/255, train[:, 0]
x_test, y_test = test[:, 1:]/255, test[:, 0]
x_train = x_train.view(-1, 1, 28, 28)
x_test = x_test.view(-1, 1, 28, 28)
y_train[y_train > 9] -= 1
y_test[y_test > 9] -= 1
#y_train = np.array([[int(i==y) for i in range(24)] for y in y_train])
#y_test = np.array([[int(i==y) for i in range(24)] for y in y_test])
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self, size_hidden_layer=200):
        super().__init__()
        self.conv_layers = [nn.Conv2d(1, 5, 3), nn.Conv2d(5, 10, 3), nn.Conv2d(10, 20, 3)] # in_channels, out_channels, kernel_size (stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20, size_hidden_layer)
        self.fc2 = nn.Linear(size_hidden_layer, 24)

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            test = self.conv_layers[i](x)
            x = self.pool(F.relu(self.conv_layers[i](x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

losses_train, accuracies_train, losses_test, accuracies_test = [], [], [], []
for epoch in range(10):
    print(f"{epoch=}")
    loss_train, accuracy_train = 0, 0
    for batch_train_x, batch_train_y in train_dataloader:
        # Train
        cnn.train()
        optimizer.zero_grad()
        # Calcul de la loss
        outputs = cnn(batch_train_x)
        loss = criterion(outputs, batch_train_y)
        loss_train += loss.item()
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)
        real = batch_train_y
        accuracy_train += (predicted == real).sum().item()
        # Descente de gradient
        loss.backward()
        optimizer.step()
    accuracies_train.append(accuracy_train/len(x_train))
    losses_train.append(loss_train/len(x_train))

    loss_test, accuracy_test = 0, 0
    for batch_test_x, batch_test_y in test_dataloader:
        # Test
        cnn.eval()
        outputs = cnn(batch_test_x)
        # Calcul de la loss
        loss = criterion(outputs, batch_test_y)
        loss_test += loss.item()
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)
        real = batch_test_y
        accuracy_test += (predicted == real).sum().item()
    accuracies_test.append(accuracy_test/len(x_test))
    losses_test.append(loss_test/len(x_test))

plt.plot(range(1, len(losses_train)+1), losses_train, label="train")
plt.plot(range(1, len(losses_test)+1), losses_test, label="test")
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.legend()
plt.show()

plt.plot(range(1, len(accuracies_train)+1), accuracies_train, label="train")
plt.plot(range(1, len(accuracies_test)+1), accuracies_test, label="test")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#for x in x_train[:10]:
#    from_matrix(x)
#from_image("mnist_example.png")

#from_image("image.png")
#from_camera()
#from_camera_normalized()
#from_camera_with_3D()

"""
for i in range(1, 6):
    from_image(f"black-hand-{i}.png")
for i in range(1, 4):
    from_image(f"white-hand-{i}.png")
from_camera()
"""