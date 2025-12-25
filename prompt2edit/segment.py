import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def segment_object(segmentation_model, image, bbox):
    model, processor = segmentation_model
    input_boxes = [[[bbox]]]
    inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    
    return masks[0][0]