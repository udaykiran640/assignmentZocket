import cv2
import numpy as np
import glob
import time
import csv

# Load the yolo  weights and the configuration into the network
net = cv2.dnn.readNet('model/yolov3-prod.weights','model/yolov3-prod.cfg')

with open('model/classes.names', 'r') as f:
    classes = f.read().splitlines()
    
for file in glob.glob("./inputImages/*.png"):
    #_,  img = cap.read()
    img = cv2.imread(file)
    height, width, _ = img.shape # store the width and height of the original image

    # normalize the image using 1/255 and create an input to pass it on to setInput function
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop = False)

    # Set the input to the layer
    net.setInput(blob) 

    # Get the output layers names into the network
    output_layers_names = net.getUnconnectedOutLayersNames() 
    # Run the forward paths and get the outputs in the output layers for which we gave the output layer names
    layersOutputs = net.forward(output_layers_names) 

# Now there are results on detections already
# we need to visualize the results

# Initialize the lists
    boxes = []
    confidences = [] # probability
    class_ids = []

# Get output from each identified object
    for output in layersOutputs:
        for detection in output:
            # Store the classes predictions
            scores = detection[5:] 
            class_id = np.argmax(scores) # locations that contains the higher scores

            # extract the higher scores and assign into confidence variable
            confidence = scores[class_id] 
            if confidence > 0.5:
                 # get the original image size
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width) # per element width inside a detection
                h= int(detection[3]*height)# per element height inside a detection

    # yolo preditcs using the centers of the bounding boxes, so we need to extract the upperleft corner positions
                x = int(center_x - w/2)
                y = int(center_y - h/2)
            
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

#Non maximum suppressions: we will have more than 1 boxes for the same object
# So we need to keep their highest score boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
# identify each of the objects detected
    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (43, 75, 238), 2)
            cv2.putText(img, label + " " + confidence, (x, y-10), font, 0.8, (255,255,0), 2)
	            

    cv2.imshow('Object detector', img)
    cv2.imwrite("./detectedImages/" + str(time.strftime("%Y%m%d-%H%M%S")) + ".jpg", img)
    key = cv2.waitKey(1)
    if key == 27: # Break the while loop when pressed escape key
        break   
#cap.release()
cv2.destroyAllWindows()





