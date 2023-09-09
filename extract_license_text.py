from easyocr import Reader 
import cv2
import difflib
import matplotlib.pyplot as plt


f = open('areas.txt', 'r')
words = f.read().splitlines()

vclass = [
    'গ','হ','ল','ঘ','চ','ট','থ','এ',
    'ক','খ','ভ','প','ছ','জ','ঝ','ব',
    'স','ত','দ','ফ','ঠ','ম','ন','অ',
    'ড','উ','ঢ','শ','ই','য','র'
]

dict = []

for w in words:
    for c in vclass:
        dict.append(f'{w}-{c}')



reader = Reader(['bn'], verbose = False, recog_network = 'bn_license_tps', model_storage_directory = "./models/EasyOCR/models",user_network_directory="./models/EasyOCR/user_network", download_enabled = False)

nums = set('০১২৩৪৫৬৭৮৯')

def extract_license_text(path, reader = reader):
    img = path
    if isinstance(path, str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
    
    result = reader.readtext(img, detail = False, paragraph = True)
    area = ""
    number = ""

    for c in "".join(result)[::-1]:
        if c == "-":
            if len(number) <= 4:
                number += "-"
            else:
                area += "-"
        elif c in nums:
            number += c
        else:
            area += c
    
    area = area[::-1]
    match = difflib.get_close_matches(area, dict, n = 1, cutoff = 0.5)

    if match:
        area = match[0]

    number = number[::-1]

    if number.find("-") == -1 and len(number) == 6:
        number = number[:2] + "-" + number[2:]
    
    return area.strip(), number.strip()