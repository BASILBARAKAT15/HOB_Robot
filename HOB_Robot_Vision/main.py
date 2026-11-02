from pathlib import Path
from collections import Counter
import re
import os
import argparse
import cv2
import pytesseract

ROOM = re.compile(r'(?:B[12]|G0|[1-4])[0-9]{3}')

if os.name == 'nt':
    tesseract = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    if tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract)

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    default=str(BASE / 'images' / 'WhatsApp Image 2026-10-20 at 12.17.51 PM.jpeg')
)
args = parser.parse_args()

img = cv2.imread(args.image)
if img is None:
    raise SystemExit('Cannot open image: '+str(args.image))

output = BASE / 'output'
output.mkdir(exist_ok=True)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = gray.shape

candidates=[]
for t in (95, 75, 115):
    mask=cv2.inRange(gray,0,t)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
             cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        x,y,w,h=cv2.boundingRect(cont)
        if not (max(30,W*0.012) <= w <= W*0.3 and
                max(12,H*0.012) <= h <= H*0.13 and
                1.3 <= w/max(h,1) <= 7.5):
            continue
        box=(x,y,w,h)
        # Remove nearly duplicate threshold detections
        is_duplicate = any(
            abs(x-bx)<=8 and abs(y-by)<=8 and
            abs(w-bw)<=12 and abs(h-bh)<=12
            for bx,by,bw,bh in candidates
        )
        if not is_duplicate:
            candidates.append(box)

candidates.sort(key=lambda b:b[1])
print('Possible plaques:',len(candidates))
marked=img.copy()
found=None
for i,(x,y,w,h) in enumerate(candidates[:25],1):
    cv2.rectangle(marked,(x,y),(x+w,y+h),(0,180,255),2)
    # Remove a thin dark outer border and include side lettering
    x0=max(0,x-max(4,round(w*.06)))
    y0=max(0,y+max(1,round(h*.03)))
    x1=min(W,x+w+max(4,round(w*.06)))
    y1=min(H,y+h+max(2,round(h*.08)))
    crop=gray[y0:y1,x0:x1]
    up=cv2.resize(crop,None,fx=3,fy=3,
                   interpolation=cv2.INTER_CUBIC)
    _,bw=cv2.threshold(up,0,255,
                      cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    attempts=[]
    for processed in (up,bw):
        for psm in (7,6):
            text=pytesseract.image_to_string(
                processed, lang='eng',
                config=f'--psm {psm} -c tessedit_char_whitelist=BG0123456789'
            )
            clean=re.sub(r'[^A-Z0-9]','',text.upper())
            if clean:
                attempts.append(clean)
    print('Region',i, 'at', (x,y,w,h), 'OCR:', attempts)
    valid=[value for value in attempts if ROOM.fullmatch(value)]
    if len(valid)>=2:
        code=Counter(valid).most_common(1)[0][0]
        found=code
        cv2.imwrite(str(output/'sign_crop.png'),up)
        cv2.rectangle(marked,(x,y),(x+w,y+h),(0,255,0),3)
        print('ROOM:',code)
        print('FLOOR:', 'G' if code.startswith('G0') else code[:-3])
        print('NUMBER:',code[-3:])
        break
cv2.imwrite(str(output/'candidates.jpg'),marked)
if not found:
    print('No reliable room code found; check output/candidates.jpg')
