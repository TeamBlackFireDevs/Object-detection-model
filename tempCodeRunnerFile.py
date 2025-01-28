def detect_objects(frame):
    # Convert frame to YOLO input size
    resized_frame = cv2.resize(frame, (416, 416))
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    uint8_frame = (normalized_frame * 255).astype(np.uint8)
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(uint8_frame, 0.00392, (416, 416), swapRB=True, crop=False)
    # Set input blob for the network
    net.setInput(blob)
    # Forward pass through the network to get predictions
    outs = net.forward(get_output_layers(net))
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
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
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x + w - 60, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame