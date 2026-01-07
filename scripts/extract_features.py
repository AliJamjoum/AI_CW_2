import os #lets us loop through files and folders
import cv2 #open cv reads images, mediapipe cant open images itself
import mediapipe as mp #gives access to media pipe
import pandas as pd #to save features into a csv file

os.makedirs("../outputs", exist_ok=True) #ensures output directory exists


DATA_DIR = "../data" #defines where the data lives .. means go up one level
mp_hands = mp.solutions.hands # hand landmark detector


hands = mp_hands.Hands(
    static_image_mode=True, #says they are indepedent images (not video)
    max_num_hands=1, #one hand, one feature vector
    min_detection_confidence=0.5 #default threshold, if confidence too low detection fails
)

rows = [] #each image is a row of data, put in list first then convert to df
failed_images = [] #track images where mediapipe fails

#lists everything inside data
for label in os.listdir(DATA_DIR):
    #builds full path - safer than string concatenation
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        #safety in case theres random things in data folder
        continue
    
    #goes through all files in data/A/ folder
    for filename in os.listdir(label_path):
        #to only process images
        if not filename.lower().endswith(".jpg"):
            continue
        #builds full image pathway
        image_path = os.path.join(label_path, filename)

        image = cv2.imread(image_path) #reads image into memory, returns NumPy array

        #in case files fail to load
        if image is None:
            failed_images.append(image_path)
            continue

        #opencv loads images in BGR, mediapipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #runs hand + landmark detection, returns results object
        results = hands.process(image_rgb)

        #if no landmarks detected, add to failed images (noise)
        if not results.multi_hand_landmarks:
            failed_images.append(image_path)
            continue

        #.landmark is list of 21 hand points for one hand (0)
        landmarks = results.multi_hand_landmarks[0].landmark

        features = [] #creates empty list for 63 numbers

        #there are 21 landmarks with x, y, z. order guarantees consistent feature ordering and 63 val
        for lm in landmarks:
            features.append(lm.x)
            features.append(lm.y)
            features.append(lm.z)

        #row label, can trace back to image
        instance_id = f"{label}/{filename}"

        #builds single row to store in csv
        row = [instance_id] + features + [label]
        #all data collected so far
        rows.append(row)

#generated column names, (instance_id, f1, f2... label) label is what letter it is
columns = ["instance_id"] + [f"f{i}" for i in range(63)] + ["label"]

#creates dataframe, index=False (prevents pandas adding extra 0..N index column)
df = pd.DataFrame(rows, columns=columns)
df.to_csv("../outputs/landmarks_raw.csv", index=False)

#writes file paths of images that didnt load or no landmarks. 
with open("../outputs/failed_images.txt", "w") as f:
    for img in failed_images:
        f.write(img + "\n")

#close mediapipe
hands.close()
print("Done.")
print("Rows saved:", len(df))
print("Failures:", len(failed_images))