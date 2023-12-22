import cv2 as cv
import matplotlib.pyplot as plt

# image path
imagePath = 'images/me.jpeg'


def detectFace():

    # read image using cv
    img = cv.imread(imagePath)

    # convert chosen image from colored to grayscale
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # load pre-trained cascade classifier to be used in the next step
    face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # using the loaded classifier, the gray scale image will be applied to
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # The face variable is an array with four values: the x and y-axis in which the faces were detected,
    # and their width and height. The above code iterates over the identified faces and creates a bounding box that
    # spans across these measurements. The parameter 0,255,0 represents the color of the bounding box,
    # which is green, and 4 indicates its thickness.
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # display the image with the detected faces, we first need to convert the image from the BGR format to RGB:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    detectFace()
