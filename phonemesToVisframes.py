import json
import argparse
import cv2

phoneme_to_viseme_mapping = {
    'B':'P',
    'P':'P',
    'M':'P',

    'D':'T',
    'T':'T',
    'S':'T',
    'Z':'T',
    'TH':'T',
    'DH':'T',

    'G':'K',
    'K':'K',
    'N':'K',
    'NG':'K',
    'L':'K',
    'Y':'K',
    'HH':'K',

    'F':'F',
    'V':'F',

    'W':'W',
    'R':'W',

    'IY':'IY',
    'IH':'IY',

    'EH':'EY',
    'EY':'EY',
    'AE':'EY',

    'AA':'AA',
    'AW':'AA',
    'AY':'AA',

    'AH':'AH',

    'AO':'AO',
    'OY':'AO',
    'OW':'AO',

    'UH':'UH',
    'UW':'UH',

    'ER':'ER',

    'CH':'CH',
    'JH':'CH',
    'SH':'CH',
    'ZH':'CH',

    'OOV':'OOV' # For out of vocab phonemes
}

visemeToIndex = {'silence':0}

def get_video_info(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, num_frames

def main(json_file_path, fps, num_frames):
    with open(json_file_path, 'r') as json_file:
        parsed_json = json.loads(json_file.read())
    
    words = parsed_json["words"]

    visframes = [0]*num_frames

    for word in words:
        if word["case"] == "success":
            wordText = word["alignedWord"]
            
            wordPointer = word["start"]
            for phonemes in word["phones"]:
                start_s = wordPointer
                end_s = wordPointer+phonemes["duration"]

                startFrame = (int)(start_s*(fps))
                endFrame = (int)(end_s*(fps))

                for frame in range(startFrame,endFrame):
                    phoneme = phonemes["phone"].split("_")[0]
                    viseme = phoneme_to_viseme_mapping[phoneme.upper()]
                    if not viseme in visemeToIndex:
                        visemeToIndex[viseme] = len(visemeToIndex)
                    visframes[frame] = visemeToIndex[viseme] 
                
                wordPointer = end_s

    # Output the viseme -> index mapping
    f = open('./key.txt','w')
    for i in range(len(visemeToIndex)):
      for viseme in visemeToIndex:
        if(visemeToIndex[viseme] == i):
          f.write(str(visemeToIndex[viseme])+"\t"+str(viseme)+"\n")
    f.close()

    # Output the visframes
    f = open('./visframesChristian4.txt','w')
    for visframe in visframes:
        f.write(str(visframe)+"\n")
    f.close()
    print("Wrote to visframes.txt and key.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video MP4 file and a JSON file.")
    parser.add_argument("-v", "--video_file", default='/Users/christianforeman/Desktop/542_time/LipReading/cropped_video.mp4', type=str, help="Path to the video MP4 file.")
    parser.add_argument("-j", "--json_file", default='/Users/christianforeman/Desktop/VisemeClassifier/Transcript4Gentle/align.json', type=str, help="Path to the JSON file.")
    args = parser.parse_args()

    fps, num_frames = get_video_info(args.video_file)
    print(f"The video is {fps} fps and has {num_frames} frames")

    main(args.json_file, fps, num_frames)