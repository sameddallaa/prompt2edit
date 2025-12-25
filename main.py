from prompt2edit.detect import detect_object
from prompt2edit.segment import segment_object
from prompt2edit.inpaint import inpaint_image
from prompt2edit.load_models import load_object_detection_model, load_segmentation_model, load_inpainting_model
from PIL import Image
import argparse

object_detector = load_object_detection_model()
segmentation_model = load_segmentation_model()
inpainting_model = load_inpainting_model()

def replace(image_path, detect_prompt, replace_prompt):
    raw_image = Image.open(image_path).convert("RGB")
    bbox = detect_object(object_detector, raw_image, detect_prompt)
    
    if bbox is None:
        return
    
    mask = segment_object(segmentation_model, raw_image, bbox)
    
    
    final_image = inpaint_image(inpainting_model, raw_image, mask, replace_prompt)
    
    return final_image

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    detect_prompt = "a red apple"
    replace_prompt = "a green apple"
    
    edited_image = replace(image_path, detect_prompt, replace_prompt)
    if edited_image:
        edited_image.show()