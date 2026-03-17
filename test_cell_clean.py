import cv2
import numpy as np

def clean_cell_image(cell_bgr):
    """
    Takes a raw BGR cell image extracted from the warped grid,
    crops out the borders, and centers the digit.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Crop 10% from each side to remove grid lines
    h, w = gray.shape
    crop_h = int(h * 0.15)
    crop_w = int(w * 0.15)
    cropped = gray[crop_h:h-crop_h, crop_w:w-crop_w]
    
    # 2. Thresholding
    _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find if the cell is empty or has a digit
    # If the cell is mostly black (few white pixels), it's empty
    white_pixels = cv2.countNonZero(binary)
    if white_pixels < (binary.size * 0.05): # Less than 5% white pixels
        return np.zeros((28, 28), dtype=np.uint8) # Return empty black image
        
    # 4. Find bounding box of the digit
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add a tiny padding around the digit bounding box
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2*pad)
        h = min(binary.shape[0] - y, h + 2*pad)
        
        digit_crop = binary[y:y+h, x:x+w]
        
        # 5. Resize keeping aspect ratio, and place in center of 28x28
        # MNIST digits are roughly 20x20 pixels centered in a 28x28 box
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        # Prevent zero dimension crash
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        centered = np.zeros((28, 28), dtype=np.uint8)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        
        centered[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return centered
        
    return np.zeros((28, 28), dtype=np.uint8)

# Dummy test
img = np.zeros((50, 50, 3), dtype=np.uint8)
img = cv2.circle(img, (25, 25), 10, (255, 255, 255), -1)
res = clean_cell_image(img)
print("Centered shape:", res.shape)
print("Success")
