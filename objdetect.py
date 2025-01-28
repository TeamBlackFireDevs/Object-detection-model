import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import tensorflow_datasets as tfds

# Function to load and preprocess data
def load_data(data_dir):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            image = cv2.imread(os.path.join(data_dir, file))
            image = cv2.resize(image, (416, 416))  # Resize image to match YOLO input size
            images.append(image)
            label = file.split('_')[0]  # Extract label from filename
            labels.append(label)
    return np.array(images), np.array(labels)

# Load CIFAR-10 dataset from TensorFlow Datasets
(train_data, test_data), info = tfds.load('cifar10', split=['train', 'test'], with_info=True)

# Function to preprocess CIFAR-10 data
def preprocess_data(data):
    images = np.array([sample['image'].numpy() for sample in data])
    labels = np.array([sample['label'].numpy() for sample in data])
    return images, labels

# Preprocess training and testing data
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train = np.eye(num_classes)[y_train.astype(int)]
y_test = np.eye(num_classes)[y_test.astype(int)]


# Define YOLO model
def create_yolo_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Create YOLO model
input_shape = X_train.shape[1:]
model = create_yolo_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Calculate mean validation accuracy and validation loss
mean_val_accuracy = np.mean(history.history['val_accuracy'])
mean_val_loss = np.mean(history.history['val_loss'])

print("Mean Validation Accuracy: ", mean_val_accuracy)
print("Mean Validation Loss: ", mean_val_loss)


# Plot accuracy and loss over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load YOLOv3 model using OpenCV's DNN module
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Function to get the output layers of the YOLOv3 network
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        layer_index = i[0] - 1 if isinstance(i, list) else i - 1
        output_layers.append(layer_names[layer_index])
    return output_layers

# Define classes
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", 
           "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Function to perform object detection
def detect_objects(frame):
    # Convert frame to YOLO input size
    resized_frame = cv2.resize(frame, (416, 416))
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Convert normalized frame back to uint8 format
    uint8_frame = (normalized_frame * 255).astype(np.uint8)
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(uint8_frame, 0.00392, (416, 416), swapRB=True, crop=False)
    # Set input blob for the network
    net.setInput(blob)
    # Forward pass through the network to get predictions
    outs = net.forward(get_output_layers(net))
    
    # Process detections
    detected_classes = set()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Only consider the highest confidence detection for each class
                class_name = classes[class_id]
                if class_name not in detected_classes:
                    detected_classes.add(class_name)
                    # Object detected, get bounding box coordinates
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    # Calculate top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"{confidence:.2f}", (x + w - 60, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)

# Counter to skip frames
frame_counter = 0
skip_frames = 1  # Adjust this value based on your system's performance

# Infinite loop to process live video
while True:
    # Perform object detection
    if frame_counter % skip_frames == 0:
        ret, frame = cap.read()
        detected_frame = detect_objects(frame)
        # Display the resulting frame
        cv2.imshow('Object Detection', detected_frame)

    frame_counter += 1
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
