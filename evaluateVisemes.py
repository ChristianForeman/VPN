import numpy as np
import pandas as pd
from tqdm import tqdm

STARTFRAME = 1000
ENDFRAME = 35000

# Ground Truth Visemes
gtVisemesFile = open("./visframesChristian4.txt","r") 
gtVisemes = gtVisemesFile.read().split("\n")


# Predicted Visemes
predVisemes = np.loadtxt("./phoframe41Christian4/outputted.txt")
predVisemesDf = pd.DataFrame(data=predVisemes)


# Viseme Keys
visemeKeys = np.loadtxt("key.txt", dtype='str')
visemeKeysDf = pd.DataFrame(data = visemeKeys, columns = ['Viseme ID', 'Viseme Name'])

# Metrics
visemeKeysDf['Correct Classifications'] = 0
visemeKeysDf['Total Occurences'] = 0

for idx, row in tqdm(predVisemesDf.iterrows(), total=predVisemesDf.shape[0]):
    if idx < STARTFRAME or idx > ENDFRAME:
        continue

    # Find the max value in the row, aka the predicted viseme
    pred_viseme = int(np.argmax(row))

    gtViseme = int(gtVisemes[idx + 15])

    if pred_viseme == gtViseme:
        # This was a correct guess, add to the total correct for this viseme
        visemeKeysDf.loc[visemeKeysDf['Viseme ID'] == str(pred_viseme), 'Correct Classifications'] = visemeKeysDf['Correct Classifications'] + 1
        visemeKeysDf.loc[visemeKeysDf['Viseme ID'] == str(pred_viseme), 'Total Occurences'] = visemeKeysDf['Total Occurences'] + 1
        
    else:
        # This was an incorrect guess, add to total number of visemes
        visemeKeysDf.loc[visemeKeysDf['Viseme ID'] == str(gtViseme), 'Total Occurences'] = visemeKeysDf['Total Occurences'] + 1

visemeKeysDf['% Correct'] = visemeKeysDf['Correct Classifications'] / visemeKeysDf['Total Occurences']

print(visemeKeysDf)