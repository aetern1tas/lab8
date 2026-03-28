import cv2

def convert_to_grayscale():
    img_path = 'variant-1.jpg' 
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original', img)
    cv2.imshow('Grayscale (Halftone)', gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('result.jpg', gray)

if __name__ == "__main__":
    convert_to_grayscale()