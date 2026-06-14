import os
import sys
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

def main():
    print("--- Loading BLIP Image Captioning Model ---")
    
    # Set device dynamically (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and feature processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    print(f"Model ready on {device}.\n")

    while True:
        img_path = input("Enter image file path (or 'exit'): ").strip().strip("'\"")
        
        if img_path.lower() in ['exit', 'quit']:
            break
            
        if not os.path.exists(img_path):
            print("File not found. Check the path and try again.\n")
            continue

        try:
            # Open image and convert to standard RGB channel format
            raw_img = Image.open(img_path).convert('RGB')
            
            # Step 1: Preprocess image pixels into tensors
            inputs = processor(raw_img, return_tensors="pt").to(device)
            
            # Step 2: Autoregressively generate token IDs 
            with torch.no_grad():
                output_tokens = model.generate(**inputs, max_new_tokens=40)
            
            # Step 3: Decode text tokens to natural language
            caption = processor.decode(output_tokens[0], skip_special_tokens=True)
            print(f"Caption: {caption.capitalize()}\n")
            
        except Exception as e:
            print(f"Error processing image: {e}\n")

if __name__ == "__main__":
    main()