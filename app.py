#!/usr/bin/env python
from flask import Flask, render_template, Response, make_response
import cv2
import sys
import yaml
import numpy as np



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    fn = r"datasets/parkinglot_1_480p.mp4"
    fn_yaml = r"datasets/parking2.yml"
    config = {'save_video': False,
              'text_overlay': True,
              'parking_overlay': True,
              'parking_id_overlay': False,
              'parking_detection': True,
              'min_area_motion_contour': 60,
              'park_sec_to_wait': 0,
              'start_frame': 0}
    
    cap = cv2.VideoCapture('https://videos3.earthcam.com/fecnetwork/9974.flv/playlist.m3u8')
    
    video_info = {'fps':    cap.get(cv2.CAP_PROP_FPS),
                  'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                  'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.set(cv2.CAP_PROP_POS_FRAMES, config['start_frame'])

    with open(fn_yaml, 'r') as stream:
        parking_data = yaml.load(stream)
    parking_contours = []
    parking_bounding_rects = []
    parking_mask = []
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:,0] = points[:,0] - rect[0] # shift contour to roi
        points_shifted[:,1] = points[:,1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)

    parking_status = [False]*len(parking_data)
    parking_buffer = [None]*len(parking_data)


    while(cap.isOpened()):   
        spot = 0
        occupied = 0 
        # Read frame-by-frame    
        video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current position of the video file in seconds
        video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) # Index of the frame to be decoded/captured next
        ret, frame = cap.read()    
        if ret == False:
            print("Capture Error")
            break
    
        # frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        # Background Subtraction
        frame_blur = cv2.GaussianBlur(frame.copy(), (5,5), 3)
        frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()

        if config['parking_detection']:        
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                rect = parking_bounding_rects[ind]
                roi_gray = frame_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calculation   
                # print np.std(roi_gray)

                points[:,0] = points[:,0] - rect[0] # shift contour to roi
                points[:,1] = points[:,1] - rect[1]
                # print np.std(roi_gray), np.mean(roi_gray)
                status = np.std(roi_gray) < 22 and np.mean(roi_gray) > 53
                # If detected a change in parking status, save the current time
                if status != parking_status[ind] and parking_buffer[ind]==None:
                    parking_buffer[ind] = video_cur_pos
                # If status is still different than the one saved and counter is open
                elif status != parking_status[ind] and parking_buffer[ind]!=None:
                    if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                        parking_status[ind] = status
                        parking_buffer[ind] = None
                # If status is still same and counter is open                    
                elif status == parking_status[ind] and parking_buffer[ind]!=None:
                    #if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                    parking_buffer[ind] = None                    
                # print(parking_status)
   
        if config['parking_overlay']:                    
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                if parking_status[ind]: 
                    color = (0,255,0)
                    spot = spot+1
                else: 
                    color = (0,0,255)
                    occupied = occupied+1
                cv2.drawContours(frame_out, [points], contourIdx=-1,
                                 color=color, thickness=2, lineType=cv2.LINE_8)            
                moments = cv2.moments(points)        
                centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
                cv2.putText(frame_out, str(park['id']), (centroid[0]+1, centroid[1]+1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame_out, str(park['id']), (centroid[0]-1, centroid[1]-1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame_out, str(park['id']), (centroid[0]+1, centroid[1]-1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame_out, str(park['id']), (centroid[0]-1, centroid[1]+1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame_out, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                # print 'occupied: ', occupied
                # print 'spot: ', spot

        # Draw Overlay
        if config['text_overlay']:
            cv2.rectangle(frame_out, (1, 5), (280, 90),(255,255,255), 85) 
            str_on_frame = "Frames: %d/%d" % (video_cur_frame, video_info['num_of_frames'])
            cv2.putText(frame_out, str_on_frame, (5,30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        0.7, (0,128,255), 2, cv2.LINE_AA)
            str_on_frame = "Spot: %d Occupied: %d" % (spot, occupied)
            cv2.putText(frame_out, str_on_frame, (5,90), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        0.7, (0,128,255), 2, cv2.LINE_AA)  
    
        # Display
        retval, buffer = cv2.imencode('.jpg', frame_out)
        response = buffer.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+response+b'\r\n')
        
    del(cap)

@app.route('/calc')
def calc():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    
#if config['save_video']: out.release()
#cv2.destroyAllWindows()    