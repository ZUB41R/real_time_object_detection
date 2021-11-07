from flask import Flask, render_template, Response
import cv2
# from yolov3_coco_body_detector import BodyDetector
from ssd_body_detector import BodyDetector


app = Flask(__name__)

camera = cv2.VideoCapture(0)
body_detector = BodyDetector()
def gen_frames():
    count = 0
    while True:
        count += 1
        print(f"Frame no: {count}")
        success, frame = camera.read()
        
        if not success:
            break
        else:
            # print(f"Frame_sahpe: {frame.shape}")
            # result_frame = body_detector.process(frame)
            # haarcascade_body_detector = cv2.CascadeClassifier("Haarcascades/haarcascade_fullbody.xml")
            haarcascade_face_detector = cv2.CascadeClassifier("/Users/zubairahmed/my_projects/real_time_object_detection/Haarcascades/haarcascade_frontalface_default.xml")
            
            # bodies = haarcascade_body_detector.detectMultiScale(frame, 1.1, 7)
            faces = haarcascade_face_detector.detectMultiScale(frame, 1.1, 7)
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # for (x, y, w, h) in bodies:
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # result_frame = frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # Concatenate frame and show result one by one

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vide_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)