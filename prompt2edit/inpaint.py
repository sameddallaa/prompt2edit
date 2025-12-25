import numpy as np
from PIL import Image

def inpaint_image(inpainting_model, image, mask, prompt):
    mask = (mask[0].cpu().numpy().astype(np.uint8) * 255)
    mask_image = Image.fromarray(mask)
    
    output = inpainting_model(
        prompt=prompt, 
        image=image, 
        mask_image=mask_image
    ).images[0]
    
    return output