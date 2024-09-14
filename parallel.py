import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import aiohttp
import asyncio
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os

# Initialize OCR with GPU acceleration (if available)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Set use_gpu=True if GPU is available

# Load the dataset
df = pd.read_csv('dataset/test.csv')
df['detected_text'] = ''

# Asynchronous function to download image from URL
async def download_image_from_url(session, image_url):
    async with session.get(image_url) as response:
        if response.status == 200:
            img = Image.open(BytesIO(await response.read()))
            return img
        else:
            return None

# Function to process OCR on the image
def perform_ocr(img_np):
    result = ocr.ocr(img_np)
    full_text = ' '.join([line[1][0] for line in result[0]]) if result else ""
    return full_text

# Asynchronous image processing
async def process_image(i, image_url, session):
    try:
        img = await download_image_from_url(session, image_url)
        if img is not None:
            img_np = np.array(img)
            full_text = perform_ocr(img_np)
            df.at[i, 'detected_text'] = full_text
        else:
            df.at[i, 'detected_text'] = "Error: Image download failed"
    except Exception as e:
        df.at[i, 'detected_text'] = f"Error: {str(e)}"
        print(f"Failed to process image at index {i}: {e}")

# Function to handle batch processing and saving
async def process_batch(start_index, batch_size, max_workers=10):
    # Asynchronous session
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Multi-threaded OCR execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(start_index, min(start_index + batch_size, len(df))):
                image_url = df['image_link'][i]
                tasks.append(asyncio.ensure_future(process_image(i, image_url, session)))

            # Wait for all tasks to complete
            await asyncio.gather(*tasks)

# Main function
async def main():
    start_index = 0
    batch_size = 1000
    max_workers = 10  # Number of threads for OCR processing

    # Determine the starting index from the checkpoint if resuming
    checkpoint_file = 'checkpoint.txt'
    try:
        with open(checkpoint_file, 'r') as f:
            start_index = int(f.read().strip())
    except FileNotFoundError:
        start_index = 0

    # Process in batches
    for i in range(start_index, len(df), batch_size):
        print(f"Processing batch starting from {i}")
        await process_batch(i, batch_size, max_workers)

        # Save the DataFrame incrementally
        checkpoint_index = i + batch_size
        df.iloc[start_index:i + batch_size].to_csv(f'updated_image_ocr_results_{checkpoint_index}.csv', index=False)
        with open(checkpoint_file, 'w') as f:
            f.write(str(checkpoint_index))
        print(f"Saved checkpoint at index {checkpoint_index}")

    # Save the final batch
    if start_index < len(df):
        df.iloc[start_index:].to_csv(f'updated_image_ocr_results_{len(df)}.csv', index=False)
        print(f"Saved final batch from index {start_index} to {len(df)}")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
