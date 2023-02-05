import cv2
import numpy as np
from o3d_utils import Visu3D
import mediapipe_utils as mpu



# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINES_BODY = [[9,10],[4,6],[1,3],
            [12,14],[14,16],[16,20],[20,18],[18,16],
            [12,11],[11,23],[23,24],[24,12],
            [11,13],[13,15],[15,19],[19,17],[17,15],
            [24,26],[26,28],[32,30],
            [23,25],[25,27],[29,31]]

COLORS_BODY = ["middle","right","left",
                "right","right","right","right","right",
                "middle","middle","middle","middle",
                "left","left","left","left","left",
                "right","right","right","left","left","left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]





class BlazeposeRenderer:
    def __init__(self,
                tracker,
                show_3d=None,
                output=None):
        self.tracker = tracker
        self.show_3d = show_3d
        self.fram = None
        self.pause = False

        # Rendering flags
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_score = False
        self.show_fps = False

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz

        if self.show_3d == "mixed" and not self.tracker.xyz:
            print("'mixed' 3d visualization needs the tracker to be in 'xyz' mode !")
            print("3d visualization falling back to 'world' mode.")
            self.show_3d = 'world'
        if self.show_3d == "image":
            self.vis3d = Visu3D(zoom=0.7, segment_radius=3)
            z = min(tracker.img_h, tracker.img_w)/3
            self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
            self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
            self.vis3d.init_view()
        elif self.show_3d == "world":
            self.vis3d = Visu3D(bg_color=(0.2, 0.2, 0.2), zoom=1.1, segment_radius=0.01)
            self.vis3d.create_grid([-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1],2,2) # Floor
            self.vis3d.create_grid([-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],2,2) # Wall
            self.vis3d.init_view()
        elif self.show_3d == "mixed":
            self.vis3d = Visu3D(bg_color=(0.4, 0.4, 0.4), zoom=0.7, segment_radius=0.01)
            half_length = 3
            grid_depth = 5
            self.vis3d.create_grid([-half_length,1,0],[half_length,1,0],[half_length,1,grid_depth],[-half_length,1,grid_depth],2*half_length,grid_depth) # Floor
            self.vis3d.create_grid([-half_length,1,grid_depth],[half_length,1,grid_depth],[half_length,-1,grid_depth],[-half_length,-1,grid_depth],2*half_length,2) # Wall
            self.vis3d.create_camera()
            self.vis3d.init_view()

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,tracker.video_fps,(tracker.img_w, tracker.img_h)) 

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.tracker.presence_threshold

    def draw_landmarks(self, body, my_keypoints=None):
        if self.show_rot_rect:
            cv2.polylines(self.frame, [np.array(body.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
        if self.show_landmarks:                
            list_connections = LINES_BODY
            lines = [np.array([body.landmarks[point,:2] for point in line]) for line in list_connections if self.is_present(body, line[0]) and self.is_present(body, line[1]) and (my_keypoints is None or (line[0] in my_keypoints and line[1] in my_keypoints))]
            cv2.polylines(self.frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            
            # for i,x_y in enumerate(body.landmarks_padded[:,:2]):
            for i,x_y in enumerate(body.landmarks[:self.tracker.nb_kps,:2]):
                if self.is_present(body, i) and (my_keypoints is None or i in my_keypoints):
                    if i > 10:
                        color = (0,255,0) if i%2==0 else (0,0,255)
                    elif i == 0:
                        color = (0,255,255)
                    elif i in [4,5,6,8,10]:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    cv2.circle(self.frame, (x_y[0], x_y[1]), 4, color, -11)
            
            # init variables
            if body:
                LEFT_SHOULDER = body.landmarks[mpu.KEYPOINT_DICT['left_shoulder']] if self.is_present(body, mpu.KEYPOINT_DICT['left_shoulder']) else None
                RIGHT_SHOULDER = body.landmarks[mpu.KEYPOINT_DICT['right_shoulder']] if self.is_present(body, mpu.KEYPOINT_DICT['right_shoulder']) else None
                LEFT_ELBOW = body.landmarks[mpu.KEYPOINT_DICT['left_elbow']] if self.is_present(body, mpu.KEYPOINT_DICT['left_elbow']) else None
                RIGHT_ELBOW = body.landmarks[mpu.KEYPOINT_DICT['right_elbow']] if self.is_present(body, mpu.KEYPOINT_DICT['right_elbow']) else None
                LEFT_WRIST = body.landmarks[mpu.KEYPOINT_DICT['left_wrist']] if self.is_present(body, mpu.KEYPOINT_DICT['left_wrist']) else None
                RIGHT_WRIST = body.landmarks[mpu.KEYPOINT_DICT['right_wrist']] if self.is_present(body, mpu.KEYPOINT_DICT['right_wrist']) else None
                LEFT_HIP = body.landmarks[mpu.KEYPOINT_DICT['left_hip']] if self.is_present(body, mpu.KEYPOINT_DICT['left_hip']) else None
                RIGHT_HIP = body.landmarks[mpu.KEYPOINT_DICT['right_hip']] if self.is_present(body, mpu.KEYPOINT_DICT['right_hip']) else None
                LEFT_KNEE = body.landmarks[mpu.KEYPOINT_DICT['left_knee']] if self.is_present(body, mpu.KEYPOINT_DICT['left_knee']) else None
                RIGHT_KNEE = body.landmarks[mpu.KEYPOINT_DICT['right_knee']] if self.is_present(body, mpu.KEYPOINT_DICT['right_knee']) else None
                LEFT_ANKLE = body.landmarks[mpu.KEYPOINT_DICT['left_ankle']] if self.is_present(body, mpu.KEYPOINT_DICT['left_ankle']) else None
                RIGHT_ANKLE = body.landmarks[mpu.KEYPOINT_DICT['right_ankle']] if self.is_present(body, mpu.KEYPOINT_DICT['right_ankle']) else None
                LEFT_HEEL = body.landmarks[mpu.KEYPOINT_DICT['left_heel']] if self.is_present(body, mpu.KEYPOINT_DICT['left_heel']) else None
                RIGHT_HEEL = body.landmarks[mpu.KEYPOINT_DICT['right_heel']] if self.is_present(body, mpu.KEYPOINT_DICT['right_heel']) else None
                LEFT_FOOT_INDEX = body.landmarks[mpu.KEYPOINT_DICT['left_foot_index']] if self.is_present(body, mpu.KEYPOINT_DICT['left_foot_index']) else None
                RIGHT_FOOT_INDEX = body.landmarks[mpu.KEYPOINT_DICT['right_foot_index']] if self.is_present(body, mpu.KEYPOINT_DICT['right_foot_index']) else None

                # draw angles
                if LEFT_SHOULDER is not None and LEFT_ELBOW is not None and LEFT_WRIST is not None:
                    cv2.putText(self.frame, f"{mpu.angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST):.2f}", (LEFT_ELBOW[0]+20, LEFT_ELBOW[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if RIGHT_SHOULDER is not None and RIGHT_ELBOW is not None and RIGHT_WRIST is not None:
                    cv2.putText(self.frame, f"{mpu.angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST):.2f}", (RIGHT_ELBOW[0]+20, RIGHT_ELBOW[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if LEFT_HIP is not None and LEFT_KNEE is not None and LEFT_ANKLE is not None:
                    cv2.putText(self.frame, f"{mpu.angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE):.2f}", (LEFT_KNEE[0]+20, LEFT_KNEE[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if RIGHT_HIP is not None and RIGHT_KNEE is not None and RIGHT_ANKLE is not None:
                    cv2.putText(self.frame, f"{mpu.angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE):.2f}", (RIGHT_KNEE[0]+20, RIGHT_KNEE[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)



        if self.show_score:
            h, w = self.frame.shape[:2]
            cv2.putText(self.frame, f"Landmark score: {body.lm_score:.2f}", 
                        (20, h-60), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

        if self.show_xyz and body.xyz_ref:
            x0, y0 = body.xyz_ref_coords_pixel.astype(np.int)
            x0 -= 50
            y0 += 40
            cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
            cv2.putText(self.frame, f"X:{body.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
            cv2.putText(self.frame, f"Y:{body.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            cv2.putText(self.frame, f"Z:{body.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        if self.show_xyz_zone and body.xyz_ref:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(body.xyz_zone[0:2]), tuple(body.xyz_zone[2:4]), (180,0,180), 2)

    def draw_3d(self, body, my_keypoints=None):
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        if body is not None:
            points = body.landmarks if self.show_3d == "image" else body.landmarks_world
            draw_skeleton = True
            if self.show_3d == "mixed":  
                if body.xyz_ref:
                    """
                    Beware, the y value of landmarks_world coordinates is negative for landmarks 
                    above the mid hips (like shoulders) and negative for landmarks below (like feet).
                    The y value of (x,y,z) coordinates given by depth sensor is negative in the lower part
                    of the image and positive in the upper part.
                    """
                    translation = body.xyz / 1000
                    translation[1] = -translation[1]
                    if body.xyz_ref == "mid_hips":                   
                        points = points + translation
                    elif body.xyz_ref == "mid_shoulders":
                        mid_hips_to_mid_shoulders = np.mean([
                            points[mpu.KEYPOINT_DICT['right_shoulder']],
                            points[mpu.KEYPOINT_DICT['left_shoulder']]],
                            axis=0) 
                        points = points + translation - mid_hips_to_mid_shoulders   
                else: 
                    draw_skeleton = False
            if draw_skeleton:
                lines = LINES_BODY
                colors = COLORS_BODY
                for i,a_b in enumerate(lines):
                    a, b = a_b
                    if self.is_present(body, a) and self.is_present(body, b) and (my_keypoints is None or a in my_keypoints or b in my_keypoints):
                            self.vis3d.add_segment(points[a], points[b], color=colors[i])
        self.vis3d.render()
                
        
    def draw(self, frame, body, my_keypoints=None):
        if not self.pause:
            self.frame = frame
            if body:
                self.draw_landmarks(body, my_keypoints)
            self.body = body
        elif self.frame is None:
            self.frame = frame
            self.body = None
        # else: self.frame points to previous frame
        if self.show_3d:
            self.draw_3d(self.body, my_keypoints)
        return self.frame
    
    def exit(self):
        if self.output:
            self.output.release()

    def waitKey(self, delay=1):
        if self.show_fps:
            self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Blazepose", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            self.pause = not self.pause
        elif key == ord('r'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('s'):
            self.show_score = not self.show_score
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('x'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('z'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        return key
        
            