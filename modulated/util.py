from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from diffusion_fusion.util import to_numpy_image
import matplotlib.pyplot as plt
import math
def load_owlvit(checkpoint_path, device):
    """
    Return: model, processor (for text inputs)
    """
    processor = OwlViTProcessor.from_pretrained(checkpoint_path)
    model = OwlViTForObjectDetection.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model, processor
    
def prepare_image(batch, input):
    output = to_numpy_image(torch.cat(
        (batch, input[2].to(batch.device), input[3].to(batch.device)), dim=1))
    image = cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(image)
    
def detect_objects(image, text_prompt, processor, model,device, threshold=0.0):
    texts = [text_prompt.split(",")]
    with torch.no_grad():
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes.to(device))
    scores = torch.sigmoid(outputs.logits)
    return results, scores, texts
    
def filter_detections(results, scores, texts, get_topk=False, topk_k=1, min_ratio=0.1, min_score=0.005):
    i = 0
    text = texts[i]
    if get_topk:
        topk_scores, topk_idxs = torch.topk(scores, k=topk_k, dim=1)
        topk_idxs = topk_idxs.squeeze(1).tolist()
        boxes = results[i]['boxes'][topk_idxs]
        scores = topk_scores.view(len(text), -1)
        labels = results[i]["labels"][topk_idxs]
    else:
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    score_max = max([round(score.item(), 4) for score in scores])
    
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        s = round(score.item(), 4)
        if s >= score_max * min_ratio and s >= min_score:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)
            #print(f"Detected {text[label]} with confidence {s} at location {[round(i,2) for i in box.tolist()]}")
    
    return text, filtered_boxes, filtered_scores, filtered_labels
    
    
def run_sam_segmentation(image, boxes, predictor,device):
    size = image.size
    image_np = np.asarray(image)
    predictor.set_image(image_np)
    
    if len(boxes) == 0:
        return torch.full((1, image_np.shape[0], image_np.shape[1]), 0).to(device)

    boxes_tensor = torch.tensor([box.tolist() for box in boxes], device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, image_np.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    mask_end = masks[0]
    for mask in masks[1:]:
        mask_end = mask_end | mask


    #plt.figure(figsize=(10, 10))
    #plt.imshow(image_np)
    #show_mask(mask_end.cpu().numpy(), plt.gca(), random_color=True)
    #for box in boxes_tensor:
       # show_box(box.cpu().numpy(), plt.gca())
    #plt.axis('off')
    #plt.savefig(f"owlvit_segment_anything_output.jpg")

    return mask_end

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def get_mask(image, boxes):  
    image_np = np.array(image)  
    height, width = image_np.shape[:2]  
    boxes = np.array(boxes).reshape(-1, 4) 
    boxes_normalized = np.clip(boxes, [0, 0, 0, 0], [width, height, width, height]).astype(int)  
    mask = np.zeros((height, width), dtype=np.uint8)  
    for box in boxes_normalized:  
        x1, y1, x2, y2 = box      
        roi = image_np[y1:y2, x1:x2] 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
        edges = cv2.Canny(blurred, 50, 150)  
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        if not contours:  
            continue  
        largest_contour = max(contours, key=cv2.contourArea)  
        roi_mask = np.zeros_like(roi[:,:,0])  
        cv2.drawContours(roi_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)  
        mask[y1:y2, x1:x2] = roi_mask  
    return mask  

def resize_and_align16_batch(images):
    
    assert isinstance(images, torch.Tensor)
    assert images.dim() == 4

    resized_images = []
    for img in images:
        C, H, W = img.shape

        scale = 256 / min(H, W)
        H_new = int(H * scale)
        W_new = int(W * scale)

       
        H_new = math.ceil(H_new / 16) * 16
        W_new = math.ceil(W_new / 16) * 16

        img_resized = TF.resize(img, [H_new, W_new], antialias=True)
        resized_images.append(img_resized)

    return torch.stack(resized_images)  

def OWL_VIT_SAM(batch,batch1, input, text_prompt, processor, model, predictor):
    image = prepare_image(batch, input)
    results, scores, texts = detect_objects(image, text_prompt, processor, model,device=batch.device)
    text, boxes, scores, labels = filter_detections(results, scores, texts)

  
    boxes_np = np.array([box.cpu().detach().numpy() for box in boxes])
    mask = get_mask(image, boxes_np)

  
    final_mask1 = run_sam_segmentation(image, boxes, predictor,device=batch.device)
    
    image = prepare_image(batch1, input)
    results, scores, texts = detect_objects(image, text_prompt, processor, model,device=batch.device)
    text, boxes, scores, labels = filter_detections(results, scores, texts)

   
    boxes_np = np.array([box.cpu().detach().numpy() for box in boxes])
    mask = get_mask(image, boxes_np)

   
    final_mask2 = run_sam_segmentation(image, boxes, predictor,device=batch.device)
    final_mask=(final_mask1|final_mask2).long().unsqueeze(0)
    
    return {'condition': resize_and_align16_batch(batch)},{'condition': resize_and_align16_batch(batch1)},resize_and_align16_batch(final_mask),final_mask