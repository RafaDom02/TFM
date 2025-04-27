# detect_people_webcam.py
# Uso: Captura vídeo de la webcam, detecta personas con Google Cloud Vision API,
# aplica Non-Maximum Suppression (NMS) para filtrar detecciones superpuestas,
# y las diferencia mediante seguimiento de centroides con reidentificación por histograma de color
# más un filtro de Kalman para predecir posiciones en movimientos bruscos.
#
# Requisitos:
#  - Python 3.x
#  - Bibliotecas necesarias:
#      pip install opencv-python numpy google-cloud-vision
#
# Para mitigar pérdida de ID en movimientos rápidos:
# 1. Aumentar maxDisappeared para tolerar más frames sin detección.
# 2. Usar un filtro de Kalman por cada objeto para predecir posición cuando la visión falla.
# 3. Ajustar reidThreshold si la iluminación o ángulo cambian mucho.

import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\User\Desktop\Master\TFM\test reconocimiento de imagen\nimble-root-457808-r2-b639a6729402.json'

# detect_people_webcam.py
# Uso: Captura vídeo de la webcam, detecta personas con Google Cloud Vision API,
# aplica Non-Maximum Suppression (NMS) para filtrar detecciones superpuestas,
# realiza seguimiento con Kalman y reidentificación por reconocimiento facial
# para mantener IDs persistentes incluso al entrar y salir del plano.
#
# Requisitos:
#   pip install opencv-python numpy google-cloud-vision face_recognition

import cv2
import numpy as np
import face_recognition
from google.cloud import vision


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if not boxes:
        return []
    boxes_arr = np.array(boxes)
    scores_arr = np.array(scores)
    x1, y1 = boxes_arr[:,0], boxes_arr[:,1]
    x2, y2 = boxes_arr[:,2], boxes_arr[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_arr.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [boxes[k] for k in keep]

class PersonTracker:
    def __init__(self, maxDisappeared=80, faceThreshold=0.6):
        self.nextID = 0
        self.objects = {}         # id -> centroid
        self.boxes = {}           # id -> bbox
        self.disappeared = {}     # id -> frames missing
        self.kalman = {}          # id -> KalmanFilter
        self.known_encodings = {} # id -> face encoding
        self.maxDisappeared = maxDisappeared
        self.faceThreshold = faceThreshold

    def _create_kf(self, centroid):
        kf = cv2.KalmanFilter(4,2)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32)*1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*1e-1
        kf.statePre = np.array([centroid[0],centroid[1],0,0],dtype=np.float32)
        return kf

    def _get_face_encoding(self, frame, box):
        x1,y1,x2,y2 = box
        face = frame[max(0,y1):y2, max(0,x1):x2]
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        return encs[0] if encs else None

    def register(self, centroid, box, frame, reuseID=None):
        objID = reuseID if reuseID is not None else self.nextID
        self.objects[objID] = centroid
        self.boxes[objID] = box
        self.disappeared[objID] = 0
        # Initial Kalman filter
        self.kalman[objID] = self._create_kf(centroid)
        # Face encoding
        enc = self._get_face_encoding(frame, box)
        if enc is not None:
            self.known_encodings[objID] = enc
        if reuseID is None:
            self.nextID += 1

    def deregister(self, objID):
        for d in (self.objects, self.boxes, self.disappeared, self.kalman):
            d.pop(objID, None)
        # keep known_encodings for future re-id

    def update(self, rects, frame):
        h,w = frame.shape[:2]
        # Predict with Kalman and remove if predicted out of frame
        for objID, kf in list(self.kalman.items()):
            pred = kf.predict()
            cx,cy = int(pred[0]), int(pred[1])
            if cx<0 or cy<0 or cx>w or cy>h:
                self.deregister(objID)
            else:
                self.objects[objID]=(cx,cy)

        # If no detections
        if not rects:
            for objID in list(self.disappeared.keys()):
                self.disappeared[objID]+=1
                if self.disappeared[objID]>self.maxDisappeared:
                    self.deregister(objID)
            return self.boxes

        # If no active objects, register all
        if not self.objects:
            for box in rects:
                cx,cy=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                self.register((cx,cy), box, frame)
            return self.boxes

        # Compute centroids for detections
        inputCentroids = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in rects]
        objectIDs=list(self.objects.keys())
        objectCentroids=list(self.objects.values())
        D = np.linalg.norm(np.array(objectCentroids)[:,None]-np.array(inputCentroids)[None,:], axis=2)

        rows=D.min(axis=1).argsort()
        cols=D.argmin(axis=1)[rows]
        usedRows,usedCols=set(),set()

        # Match existing by spatial+face
        for (r,c) in zip(rows,cols):
            if r in usedRows or c in usedCols: continue
            if D[r,c] > max(w,h)*0.5: continue
            origID=objectIDs[r]
            enc_new=self._get_face_encoding(frame, rects[c])
            if enc_new is not None and origID in self.known_encodings:
                dist=np.linalg.norm(self.known_encodings[origID]-enc_new)
                if dist>self.faceThreshold: continue
            # assign
            self.objects[origID]=inputCentroids[c]
            self.boxes[origID]=rects[c]
            self.disappeared[origID]=0
            self.kalman[origID].correct(np.array(inputCentroids[c],dtype=np.float32))
            if enc_new is not None:
                self.known_encodings[origID]=enc_new
            usedRows.add(r); usedCols.add(c)

        # Determine unmatched detections
        unusedCols=set(range(len(rects)))-usedCols
        # For each unmatched, try re-id against known_encodings
        for c in unusedCols:
            enc_c=self._get_face_encoding(frame, rects[c])
            reuse=None
            if enc_c is not None:
                # compare to all known
                dists={objID:np.linalg.norm(enc-enc_c) for objID,enc in self.known_encodings.items()}
                bestID=min(dists, key=dists.get)
                if dists[bestID] <= self.faceThreshold and bestID not in self.objects:
                    reuse=bestID
            cx,cy=(int((rects[c][0]+rects[c][2])/2), int((rects[c][1]+rects[c][3])/2))
            self.register((cx,cy), rects[c], frame, reuseID=reuse)

        # Handle disappeared for existing unmatched
        unusedRows=set(range(len(objectIDs)))-usedRows
        if len(rects)<len(objectIDs):
            for r in unusedRows:
                objID=objectIDs[r]
                self.disappeared[objID]+=1
                if self.disappeared[objID]>self.maxDisappeared:
                    self.deregister(objID)

        return self.boxes


def main():
    # Credenciales GCP si no configurado externamente:
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\ruta\cred.json'
    client = vision.ImageAnnotatorClient()
    tracker = PersonTracker(maxDisappeared=80, faceThreshold=0.6)

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("No se abrió webcam"); return

    while True:
        ret,frame=cap.read()
        if not ret: break
        ret2,buf=cv2.imencode('.jpg',frame)
        if not ret2: continue
        image=vision.Image(content=buf.tobytes())
        resp=client.object_localization(image=image)
        anns=resp.localized_object_annotations

        h,w=frame.shape[:2]
        boxes_list, scores = [],[]
        for obj in anns:
            if obj.name.lower()=='person':
                verts=obj.bounding_poly.normalized_vertices
                pts=[(int(v.x*w),int(v.y*h)) for v in verts]
                x1,y1=min(p[0] for p in pts),min(p[1] for p in pts)
                x2,y2=max(p[0] for p in pts),max(p[1] for p in pts)
                boxes_list.append((x1,y1,x2,y2)); scores.append(obj.score)

        rects=non_max_suppression(boxes_list, scores, iou_threshold=0.5)
        tracked=tracker.update(rects, frame)

        for objID,(x1,y1,x2,y2) in tracked.items():
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID {objID}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow('Detección y Seguimiento',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__': main()
