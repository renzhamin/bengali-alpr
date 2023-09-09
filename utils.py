import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"load_image(): {path} not found")
    return img


import imutils
def resize_image(img, max_width = 500):
    if img is None:
        print(f'resize_image(): img is null')
        return
    if img.shape[0] > maxwidth:
        img = imutils.resize(img, maxwidth)
    return img


from matplotlib import pyplot as plt
def show_image(img):
    plt.axis("off")
    if isinstance(img, str):
        img = cv2.imread(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



from ultralytics.models import YOLO
model = YOLO('models/yolo.pt')
def detect_license_plate(img):
    detection = model.predict(img, conf=0.5, verbose=False)
    if detection is None:
        print(f"detect_license_plate(): img is null")
        return
    return detection[0]



from extract_license_text import extract_license_text
def detect_and_extract_lp_text(path, show_cropped_image = True):
    img = load_image(path)
    
    detection_result = detect_license_plate(img)
    bbox = detection_result.boxes.data.numpy()
    xmin, ymin = bbox[0][:2].astype(int)
    xmax, ymax = bbox[0][2:4].astype(int)
    cropped_img = img[ymin:ymax, xmin:xmax]
    if show_cropped_image:
        show_image(cropped_img)
        
    lp_text = extract_license_text(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY))
    print(lp_text)