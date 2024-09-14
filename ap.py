import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import requests
from io import BytesIO
from PIL import Image

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the dataset
df = pd.read_csv('dataset/test.csv')
df['detected_text'] = ''

def download_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Process the images
start_index = 0
batch_size = 1000

# Determine the starting index from the checkpoint if resuming
checkpoint_file = 'checkpoint.txt'
try:
    with open(checkpoint_file, 'r') as f:
        start_index = int(f.read().strip())
except FileNotFoundError:
    start_index = 0

# Process in batches
for i in range(start_index, len(df)):
    image_url = df['image_link'][i]
    if i % 10 == 0:
        print(f"Processing image {i}")
    try:
        img = download_image_from_url(image_url)
        img_np = np.array(img)
        result = ocr.ocr(img_np)
        full_text = ' '.join([line[1][0] for line in result[0]])
        df.at[i, 'detected_text'] = full_text
    except Exception as e:
        df.at[i, 'detected_text'] = f"Error: {str(e)}"
        print(f"Failed to process image at index {i}: {e}")
    
    # Save progress after every 5000 rows
    if (i + 1) % batch_size == 0:
        checkpoint_index = i + 1
        df.iloc[start_index:i + 1].to_csv(f'updated_image_ocr_results_{checkpoint_index}.csv', index=False)
        with open(checkpoint_file, 'w') as f:
            f.write(str(checkpoint_index))
        print(f"Saved checkpoint at index {checkpoint_index}")
        start_index = i + 1

# Save the final batch
if start_index < len(df):
    df.iloc[start_index:].to_csv(f'updated_image_ocr_results_{len(df)}.csv', index=False)
    print(f"Saved final batch from index {start_index} to {len(df)}")

# Clean up checkpoint file
import os
os.remove(checkpoint_file)

