#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    '/home/rafadom/2ºCuatrimestre/TFM/test reconocimiento de imagen/'
    'nimble-root-457808-r2-b639a6729402.json'
)

import cv2
import numpy as np
from google.cloud import vision
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=50, 
                 lost_track_timeout_frames=150,
                 hist_comparison_threshold=0.4):
        
        self.nextObjectID = 0
        self.objects = OrderedDict() 
        self.disappeared = OrderedDict()
        self.lost_tracks = OrderedDict() 

        self.maxDisappeared = maxDisappeared 
        self.lost_track_timeout = lost_track_timeout_frames
        self.hist_threshold = hist_comparison_threshold

    def _calculate_histogram(self, roi):
        if roi is None or roi.size == 0:
            return None 
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist.flatten()
        except cv2.error:
            return None

    def _purge_lost_tracks(self, current_frame_count):
        ids_to_purge = []
        for objectID, data in self.lost_tracks.items():
            if current_frame_count - data['deregistered_frame'] > self.lost_track_timeout:
                ids_to_purge.append(objectID)
        
        for objectID in ids_to_purge:
            del self.lost_tracks[objectID]

    def register(self, centroid, bbox, histogram):
        self.objects[self.nextObjectID] = {
            'centroid': centroid, 
            'bbox': bbox, 
            'histogram': histogram
        }
        self.disappeared[self.nextObjectID] = 0
        current_id = self.nextObjectID
        self.nextObjectID += 1
        return current_id

    def deregister(self, objectID, current_frame_count):
        if objectID in self.objects:
            self.lost_tracks[objectID] = {
                'centroid': self.objects[objectID]['centroid'],
                'bbox': self.objects[objectID]['bbox'],
                'histogram': self.objects[objectID]['histogram'],
                'deregistered_frame': current_frame_count 
            }
            del self.objects[objectID]
            if objectID in self.disappeared:
                del self.disappeared[objectID]

    def update(self, rects, rois, current_frame_count):
        self._purge_lost_tracks(current_frame_count)

        if len(rects) == 0:
            ids_a_eliminar = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    ids_a_eliminar.append(objectID)
            for objectID in ids_a_eliminar:
                 self.deregister(objectID, current_frame_count)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputBBoxes = [None] * len(rects)
        inputHistograms = [None] * len(rects)
        valid_indices = []
        
        for i in range(len(rects)):
             startX, startY, endX, endY = rects[i]
             if i < len(rois) and rois[i] is not None and rois[i].size > 0:
                hist = self._calculate_histogram(rois[i])
                if hist is not None:
                     inputHistograms[i] = hist
                     cX = int((startX + endX) / 2.0)
                     cY = int((startY + endY) / 2.0)
                     inputCentroids[i] = (cX, cY)
                     inputBBoxes[i] = rects[i]
                     valid_indices.append(i)

        if len(valid_indices) != len(rects):
             inputCentroids = inputCentroids[valid_indices]
             inputBBoxes = [inputBBoxes[i] for i in valid_indices]
             inputHistograms = [inputHistograms[i] for i in valid_indices]
             if not valid_indices:
                 return self.update([], [], current_frame_count)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                reidentified_id = self._attempt_reid(inputHistograms[i])
                if reidentified_id != -1:
                     self.objects[reidentified_id] = {
                        'centroid': inputCentroids[i], 
                        'bbox': inputBBoxes[i], 
                        'histogram': inputHistograms[i]
                     }
                     self.disappeared[reidentified_id] = 0
                     del self.lost_tracks[reidentified_id]
                else:
                     self.register(inputCentroids[i], inputBBoxes[i], inputHistograms[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = [data['centroid'] for data in self.objects.values()]
        
        D = dist.cdist(np.array(objectCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols: continue

            objectID = objectIDs[row]
            self.objects[objectID]['centroid'] = inputCentroids[col]
            self.objects[objectID]['bbox'] = inputBBoxes[col]
            self.objects[objectID]['histogram'] = inputHistograms[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(len(objectCentroids))).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID, current_frame_count)

        unusedCols = set(range(len(inputCentroids))).difference(usedCols)
        
        registered_new_ids = []
        for col in unusedCols:
             reidentified_id = self._attempt_reid(inputHistograms[col])
             if reidentified_id != -1:
                 self.objects[reidentified_id] = {
                    'centroid': inputCentroids[col], 
                    'bbox': inputBBoxes[col], 
                    'histogram': inputHistograms[col]
                 }
                 self.disappeared[reidentified_id] = 0
                 del self.lost_tracks[reidentified_id]
             else:
                 new_id = self.register(inputCentroids[col], inputBBoxes[col], inputHistograms[col])
                 registered_new_ids.append(new_id)

        return self.objects

    def _attempt_reid(self, new_histogram):
        best_match_id = -1
        min_dist = self.hist_threshold 

        if new_histogram is None:
             return -1

        for lost_id, lost_data in self.lost_tracks.items():
            lost_hist = lost_data.get('histogram')
            if lost_hist is None: continue

            try:
                dist = cv2.compareHist(new_histogram, lost_hist, cv2.HISTCMP_BHATTACHARYYA)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = lost_id
            except cv2.error:
                continue

        return best_match_id

def main():
    try:
        client = vision.ImageAnnotatorClient()
        print("Cliente de Vision API inicializado correctamente.")
    except Exception as e:
        print(f"Error al inicializar el cliente de Vision API: {e}")
        print("Asegúrate de que las credenciales de GCP estén configuradas.")
        return

    ct = CentroidTracker(maxDisappeared=50, 
                         lost_track_timeout_frames=150,
                         hist_comparison_threshold=0.4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    print("Iniciando detección. Presiona 'q' para salir.")
    frame_count = 0 

    nms_confidence_threshold = 0.4 
    nms_overlap_threshold = 0.3   

    cv2.namedWindow('Detección y Seguimiento', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        (h, w) = frame.shape[:2]
        
        boxes_for_nms = []      
        confidences_for_nms = [] 
        original_rects_pre_nms = []

        ret2, buf = cv2.imencode('.jpg', frame)
        if not ret2: continue 
        content = buf.tobytes()
        image = vision.Image(content=content)
        try:
            response = client.object_localization(image=image)
            objects_detected = response.localized_object_annotations
            if response.error.message: objects_detected = []
        except Exception as e: objects_detected = []

        for obj in objects_detected:
            if obj.name.lower() == 'person' and obj.score >= nms_confidence_threshold:
                verts = obj.bounding_poly.normalized_vertices
                pts_norm = [(max(0.0, min(1.0, v.x)), max(0.0, min(1.0, v.y))) for v in verts]
                pts = [(int(v[0] * w), int(v[1] * h)) for v in pts_norm]
                x_coords = [p[0] for p in pts]; y_coords = [p[1] for p in pts]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

                if x_max > x_min and y_max > y_min:
                    boxes_for_nms.append([x_min, y_min, x_max - x_min, y_max - y_min]) 
                    confidences_for_nms.append(float(obj.score))
                    original_rects_pre_nms.append((x_min, y_min, x_max, y_max)) 

        indices_to_keep = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, 
                                             nms_confidence_threshold, nms_overlap_threshold)

        rects_after_nms = []
        rois_after_nms = []
        if len(indices_to_keep) > 0:
            if isinstance(indices_to_keep, np.ndarray) and indices_to_keep.ndim > 1:
                 indices_to_keep = indices_to_keep.flatten()
            
            for i in indices_to_keep:
                 if 0 <= i < len(original_rects_pre_nms):
                    (x_min, y_min, x_max, y_max) = original_rects_pre_nms[i]
                    roi_y_min, roi_y_max = max(0, y_min), min(h, y_max)
                    roi_x_min, roi_x_max = max(0, x_min), min(w, x_max)
                    
                    if roi_y_max > roi_y_min and roi_x_max > roi_x_min:
                        roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                        if roi.size > 0:
                             rects_after_nms.append(original_rects_pre_nms[i])
                             rois_after_nms.append(roi)

        tracked_people = ct.update(rects_after_nms, rois_after_nms, frame_count) 

        for objectID, data in tracked_people.items():
            centroid = data['centroid']
            (startX, startY, endX, endY) = data['bbox'] 
            # Dibujar el bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Dibujar el centroide como un punto
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            
            # Mostrar ID y coordenadas del centroide
            text = f"ID {objectID} ({centroid[0]}, {centroid[1]})"
            text_y = startY - 10 if startY - 10 > 10 else startY + 15 
            cv2.putText(frame, text, (startX, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 

        cv2.imshow('Detección y Seguimiento', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    # Forzar el cierre de todas las ventanas en Linux
    for i in range(5):
        cv2.waitKey(1)

if __name__ == '__main__':
    main() 