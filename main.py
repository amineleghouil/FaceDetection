import pathlib
import cv2
from pywin.Demos.cmdserver import flags

c_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(c_path))

cam = cv2.VideoCapture("guy.mp4")

while True :
    _, frame = cam.read()
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray ,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x ,y , width , height ) in faces :
        cv2.rectangle(frame , (x,y) , (x+width ,y+height),(255,255,0),2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1)==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

