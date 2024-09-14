import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import requests
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
df = pd.read_csv('dataset/test.csv')
df['detected_text'] = ''
def download_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img
for i in range(len(df)):
    image_url = df['image_link'][i]
    if i%10==0:
    	print(i) 
    try:
        img = download_image_from_url(image_url)
        img_np = np.array(img)
        result = ocr.ocr(img_np)
        full_text = ' '.join([line[1][0] for line in result[0]])
        df.at[i, 'detected_text'] = full_text
    except Exception as e:
        df.at[i, 'detected_text'] = f"Error: {str(e)}"
        print(f"Failed to process image at index {i}: {e}")

df.to_csv('updated_image_ocr_results.csv', index=False)
