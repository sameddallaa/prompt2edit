import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_object(object_detector, image, text_prompt):
    model, processor = object_detector
    
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    
    if len(results["boxes"]) > 0:
        best_box = results["boxes"][0]
        return best_box.cpu().numpy()
    else:
        print("No object found!")
        return None