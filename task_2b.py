'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ 3356 ]
# Author List:		[ Akshit Jain, Kashish Kumawat, Khawahish Mathur, Khushi ]
# Filename:			task_2b.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
detected_list = []
numbering_list = []
img_name_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''
class CustomNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomNet, self).__init__()
        self.model = models.resnet18(weights=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the saved model
model = CustomNet(num_classes=5)
model.load_state_dict(torch.load('final_model.pth'))
model.eval()

# Define the transform for test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
###################################################################################################
###################################################################################################
''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''
def classify_event(image):
    '''
    ADD YOUR CODE HERE
    # '''
    test_image = Image.open(image).convert('RGB')
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0)  # Add batch dimension
    
    # Make a prediction
    with torch.no_grad():
        output = model(test_image)
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    
    # Map predicted class index to class name
    class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
    event = class_names[predicted_class]

    # print(f'The predicted event is: {event}')

    # print("\n*********************************************************************\n")


    # event = "variable to return the detected function"
    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''

###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list
    
def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()
###################################################################################################
###################################################################################################
