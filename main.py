import cv2
from cv2 import VideoCapture, CascadeClassifier, Mat
import os
import zipfile
import numpy as np
from PIL import Image
from os.path import exists

SRC_PATH = os.getcwd()
ASSETS_PATH = os.path.join(SRC_PATH, 'Assets')

def main() -> None:
    """Main function. Executes the face detection on the webcam function.
    """
    face_recognizer = train_LBPH_Face_Recognizer()

    width, height = 220, 220
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL  

    face_detector_path = os.path.join(os.getcwd(), "Assets", "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(face_detector_path)

    camera = cv2.VideoCapture(0)

    detect_faces_on_camera(camera, face_detector, face_recognizer, width, height, font)

    camera.release()
    cv2.destroyAllWindows()


def train_LBPH_Face_Recognizer() -> None:
    """Trains the LBPHFaceRecognizer.
    """
    if exists(os.path.join(ASSETS_PATH, "rodrigo_lbph_classifier.yml")):
        lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
        lbph_classifier.read(os.path.join(ASSETS_PATH, "rodrigo_lbph_classifier.yml"))
    else:
        extract_zip_folder()
        ids, faces = pre_process_images()
        lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=4, neighbors=14, grid_x=9, grid_y=9)
        lbph_classifier.train(faces, ids)
        lbph_classifier.write(os.path.join(ASSETS_PATH, "rodrigo_lbph_classifier.yml"))
        
    return lbph_classifier


def extract_zip_folder() -> None:
    """Extracts the zip folder containing the images to be used in the LBPH classifier.
    """
    rodrigo_zip_path = os.path.join(ASSETS_PATH, "rodrigo.zip")
    zip = zipfile.ZipFile(file=rodrigo_zip_path, mode = 'r')
    zip.extractall(os.path.join(ASSETS_PATH, "Data"))
    zip.close()


def pre_process_images() -> tuple[np.ndarray, np.ndarray]:
    """Pre-process images from a folder to be used in the LBPH classifier.

    Args:
        images_folder_path (str): Path to the folder containing the images.
    """
    image_paths = [os.path.join(ASSETS_PATH, "Data", "rodrigo", img_path) for img_path in os.listdir(os.path.join(ASSETS_PATH, "Data", "rodrigo"))]
    faces = []
    ids = []
    for path in image_paths:
        image = Image.open(path).convert('L')
        imagem_np = np.array(image, 'uint8')
        id = 1
        ids.append(id)
        faces.append(imagem_np)

    return np.array(ids), np.array(faces)


def detect_faces_on_camera(camera:VideoCapture, face_detector:CascadeClassifier, face_recognizer, width, height, font) -> None:
    """Detects faces on camera and draws a rectangle around them. Quits when 'q' is pressed.
    
    Args:
        camera (VideoCapture): Webcam object.
        face_detector (CascadeClassifier): Face detector object.
    """
    # Capture frame-by-frame
    while True:
        ret, frame = camera.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = face_detector.detectMultiScale(gray_frame, minSize=(100, 100),
                                                    minNeighbors=5)
        
        draw_rectangle_around_faces(frame, detections)
        recognize_faces_on_camera(face_recognizer, detections, frame, gray_frame, width, height, font)

        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def draw_rectangle_around_faces(frame:Mat, faces:list[tuple]) -> None:
    """Draws a rectangle around the faces.
    Args:
        frame (Mat): Frame to draw on.
        faces (list[tuple]): List of faces.
    """
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def recognize_faces_on_camera(face_recognizer, detections:list, image:Mat, gray_image:Mat, width:int, height:int, font:int) -> None:
    for (x, y, w, h) in detections:
            image_face = cv2.resize(gray_image[y:y + w, x:x + h], (width, height))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
            id, confidence = face_recognizer.predict(image_face)         

            if id == 1 and confidence < 215:
                name = "Rodrigo"
            else:
                name = "Someone else"
            cv2.putText(image, name, (x,y +(w+30)), font, 2, (0,0,255))
            cv2.putText(image, str(confidence), (x,y + (h+50)), font, 1, (0,0,255))


if __name__ == "__main__":
    main()