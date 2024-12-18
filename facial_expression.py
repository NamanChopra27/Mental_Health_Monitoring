
import cv2
from fer import FER
import matplotlib.pyplot as plt

# Function to capture a photo using the webcam
def capture_photo():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'c' to capture a photo.")
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame in a window
        cv2.imshow('Webcam', frame)

        # Check for user input to capture the photo
        if cv2.waitKey(1) & 0xFF == ord('c'):
            filename = 'photo.jpg'
            cv2.imwrite(filename, frame)
            print(f'Photo saved to {filename}')
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    return filename

# Function to detect emotions from the captured image
def detect_emotions_in_photo(filename):
    try:
        if filename:
            # Load the image
            img = cv2.imread(filename)

            # Convert image from BGR (OpenCV format) to RGB (FER format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect emotions using the FER package
            emotion_detector = FER()
            emotions = emotion_detector.detect_emotions(img_rgb)

            # If emotions are detected, annotate and display the image
            if emotions:
                for face in emotions:
                    bounding_box = face["box"]
                    detected_emotions = face["emotions"]

                    # Draw bounding box around the detected face
                    cv2.rectangle(img_rgb, (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                  (0, 255, 0), 2)

                    # Display the emotion labels and scores on the image
                    for idx, (emotion, score) in enumerate(detected_emotions.items()):
                        cv2.putText(img_rgb, f'{emotion}: {score:.2f}',
                                    (bounding_box[0], bounding_box[1] - 10 - idx * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Print the detected emotions in the console (text output)
                    print(f"Detected emotions for face at {bounding_box}:")
                    for emotion, score in detected_emotions.items():
                        print(f'{emotion}: {score:.2f}')
                    print("\n")

                # Display the annotated image using Matplotlib
                plt.figure(figsize=(8, 6))
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.show()
            else:
                print("No face detected or unable to detect emotions.")
                plt.imshow(img_rgb)
                plt.title("Captured Image with No Face Detected")
                plt.axis('off')
                plt.show()
        else:
            print("No filename provided.")
    except Exception as e:
        print(f"Error in detecting emotions: {e}")

# Main function to execute the photo capture and emotion detection
if __name__ == "__main__":
    filename = capture_photo()
    detect_emotions_in_photo(filename)
