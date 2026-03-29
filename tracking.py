import cv2
import numpy as np

def track_with_coordinates():
    template = cv2.imread('ref-point.jpg')
    
    h, w = template.shape[:2]

    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    
    fly_h, fly_w = fly.shape[:2]
    
    fly_b, fly_g, fly_r, fly_alpha = cv2.split(fly)
    fly_mask = fly_alpha / 255.0
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.7)
        
        found = False
        x, y = 0, 0
        center_x, center_y = 0, 0
        
        for pt in zip(*loc[::-1]):
            found = True
            x, y = pt[0], pt[1]
            center_x = x + w // 2
            center_y = y + h // 2
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            break
        
        if found:
            text = f"X={x}, Y={y}"
            color = (0, 255, 0)
        else:
            text = "Метка не найдена"
            color = (0, 0, 255)
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if found:
            fly_x = center_x - fly_w // 2
            fly_y = center_y - fly_h // 2
            
            roi = frame[fly_y:fly_y + fly_h, fly_x:fly_x + fly_w]
            
            for c in range(3):
                roi[:, :, c] = (roi[:, :, c] * (1 - fly_mask) + 
                               fly[:, :, c] * fly_mask).astype(np.uint8)
            
            frame[fly_y:fly_y + fly_h, fly_x:fly_x + fly_w] = roi
            
            cv2.putText(frame, f"Fly at ({center_x}, {center_y})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow('Поиск метки с мухой', frame)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_with_coordinates()