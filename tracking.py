import cv2
import numpy as np

def track_with_coordinates():
    template = cv2.imread('ref-point.jpg')
    
    h, w = template.shape[:2]

    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    
    fly_h, fly_w = fly.shape[:2]
    
    fly_b, fly_g, fly_r, fly_alpha = cv2.split(fly)
    fly_mask = fly_alpha / 255.0
    
    cam = None
    for i in range(3): 
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            break
        else:
            cam.release()
    
    if not cam or not cam.isOpened():
        return
    
    while True:
        flag, frame = cam.read()
        if not flag:
            break
        
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.7)
        
        found = False
        x, y = 0, 0
        center_x, center_y = 0, 0
        
        for xy in zip(*loc[::-1]):
            found = True
            x, y = xy[0], xy[1]
            center_x = x + w // 2
            center_y = y + h // 2
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (56, 25, 34), 2)
            
            break
        
        if found:
            text = f"X={x}, Y={y}"
            color = (0, 255, 0)
        else:
            text = "Poisk metki"
            color = (0, 0, 255)
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if found:
            fly_x = center_x - fly_w // 2
            fly_y = center_y - fly_h // 2
            
            fly_place = frame[fly_y:fly_y + fly_h, fly_x:fly_x + fly_w]
            
            for c in range(3):
                fly_place[:, :, c] = (fly_place[:, :, c] * (1 - fly_mask) + 
                               fly[:, :, c] * fly_mask).astype(np.uint8)
            
            frame[fly_y:fly_y + fly_h, fly_x:fly_x + fly_w] = fly_place

        
        cv2.imshow('Poisk metki s muxoi', frame)

        if cv2.waitKey(1) == ord('q'):
            break        
    
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_with_coordinates()