
import cv2
import mediapipe as mp
import time

# Create a VideoCapture object cap to capture frames from the default camera (index 0).
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize variables pTime and cTime to keep track of the previous and current times for calculating the frames per second (FPS) of the video feed.
pTime = 0
cTime = 0

while True:
    # Read a frame from the video feed using cap.read(). The success variable indicates whether the frame was successfully read, and img stores the frame. Then, convert the frame color format from BGR to RGB using cv2.cvtColor().
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Pass the RGB image to the hands.process() function, which detects and tracks the hand landmarks in the frame. The resulting hand landmarks are stored in the results variable.
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        #Iterate over each detected hand (handLms) and its landmarks. For each landmark, retrieve the 2D coordinates (lm.x and lm.y), scale them according to the image size, and print the index and coordinates. Also, draw a filled circle at each landmark's position using cv2.circle().
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the current time and FPS (frames per second) based on the time difference between the previous and current frames.
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # Add the calculated FPS value to the frame using cv2.putText().
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    # Display the annotated frame using `cv2
    cv2.imshow("Image", img)
    cv2.waitKey(1)