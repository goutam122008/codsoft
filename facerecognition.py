import os
import sys
import cv2
import face_recognition


def load_known_faces(known_faces_dir):
    """Loads images from a directory, extracts facial embeddings, and associates them with names."""
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Creating '{known_faces_dir}' directory. Please add target profile images there.")
        os.makedirs(known_faces_dir)
        return known_encodings, known_names

    print("Encoding known faces directory...")
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(known_faces_dir, filename)
            
            # Extract the person's name from the file name (e.g., "john_doe.jpg" -> "John Doe")
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            
            try:
                # Load image via face_recognition (RGB format)
                image = face_recognition.load_image_file(path)
                # Extract deep learning feature embeddings
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                    print(f" -> Enrolled: {name}")
                else:
                    print(f" -> Warning: No face found in {filename}. Skipped.")
            except Exception as e:
                print(f" -> Failed to process {filename}: {e}")
                
    return known_encodings, known_names


def process_image(input_image_path, known_encodings, known_names, output_path="output_result.jpg"):
    """Detects faces using Haar Cascades and recognizes them using deep learning embeddings."""
    if not os.path.exists(input_image_path):
        print(f"Error: Target input image '{input_image_path}' not found.")
        return

    # 1. Load image using OpenCV (BGR format)
    bgr_img = cv2.imread(input_image_path)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Convert to RGB format for the deep learning face_recognition module
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 2. Face Detection Stage (Using Haar Cascade)
    # Load the official Haar Cascade xml file pre-packaged within OpenCV
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    
    # Detect bounding coordinates: returns list of [x, y, width, height]
    detected_faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    print(f"\nDetected {len(detected_faces)} face(s) using Haar Cascade.")

    # 3. Face Recognition Stage
    # Convert OpenCV Haar bounding boxes to face_recognition's standard tuple format (top, right, bottom, left)
    face_locations = []
    for (x, y, w, h) in detected_faces:
        face_locations.append((y, x + w, y + h, x))

    # Compute feature vectors for the detected face regions
    current_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # Loop over each detected face and match it against our known database
    for (top, right, bottom, left), current_encoding in zip(face_locations, current_encodings):
        name = "Unknown"

        if known_encodings:
            # Compare current feature vector against all stored vectors (calculates Euclidean distance)
            matches = face_recognition.compare_faces(known_encodings, current_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, current_encoding)
            
            if True in matches:
                # Find the index of the face with the lowest vector variance distance
                best_match_idx = face_distances.argmin()
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]

        # 4. Rendering Annotations
        # Draw a bounding box around the detected face
        cv2.rectangle(bgr_img, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a text label with the recognized name below or above the box
        cv2.rectangle(bgr_img, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(bgr_img, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        print(f" -> Identified face at region [Top: {top}, Left: {left}] as: {name}")

    # Save the output visualization image to disk
    cv2.imwrite(output_path, bgr_img)
    print(f"\nProcessing complete! Annotated image saved as: '{output_path}'")


def main():
    print("====================================================")
    print("          Face Detection & Recognition AI           ")
    print("====================================================")

    # Setup directories
    known_dir = "known_profiles"
    
    # Load and encode system profiles
    known_encodings, known_names = load_known_faces(known_dir)
    
    if not known_encodings:
        print(f"\n[Notice]: The '{known_dir}' folder is empty.")
        print("Any detected faces will be flagged as 'Unknown' until profile images are added.")
    
    while True:
        print("-" * 50)
        input_path = input("Enter the path to an image file to analyze (or 'exit' to quit): ").strip()
        
        if input_path.lower() in ["exit", "quit", "q"]:
            print("System shutting down. Goodbye!")
            break
            
        input_path = input_path.strip("'\"") # Strip drag-and-drop structural symbols
        if not input_path:
            continue
            
        process_image(input_path, known_encodings, known_names)


if __name__ == "__main__":
    main()