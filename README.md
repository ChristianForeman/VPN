# Training the model:
python3 train.py

# Testing the model on an input video
python3 visframetest.py

The test file opens the trained model made from train.py and outputs outputted.txt which is the probabability
of each viseme for each frame.

# Evaluate the model
python3 evaluateVisemes.txt

This file simply counts how many times it got the viseme correct for a range of frames