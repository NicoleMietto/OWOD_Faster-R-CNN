import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from OWOD_detector import OWODFasterRCNN
import json
import random
import torchvision.transforms as T

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 1. Initialize model
    print("Loading model...")
    num_known_classes = 20
    model = OWODFasterRCNN(num_known_classes=num_known_classes)
    
    checkpoint_path = "/kaggle/working/OWOD_Faster-R-CNN/best_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Se ha 'model_state_dict', è il file "last". Altrimenti è il file "best" (che è solo i pesi)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from Epoch {checkpoint.get('epoch', 0)+1} (Last Checkpoint)")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully from Best Checkpoint format")
            
    except Exception as e:
        print(f"Error loading checkpoint. Did you put the weights in the right place? {e}")
        return
        
    model.to(device)
    model.eval() # Imposta la modalità di inferenza
    
    # 2. Pick a random image from TEST set
    test_json_path = "/kaggle/working/task1_uu_test.json"
    with open(test_json_path, 'r') as f:
        coco_data = json.load(f)
        
    img_info = random.choice(coco_data['images'])
    # Le immagini del test set (Task 1) provengono dalla cartella val2017 originale di COCO
    img_path = f"/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017/{img_info['file_name']}"
    
    print(f"Testing on image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image).to(device)
    
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        # model.eval() restituisce le predizioni finali
        predictions = model([img_tensor])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()
    
    # Filtriamo per confidenza (mostriamo solo roba sicura al > 50%)
    keep = scores > 0.5
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # --- AGGIUNTA: NMS Agnostico per pulire i box sovrapposti ---
    # Applichiamo un NMS globale (ignora le classi) per rimuovere box che si 
    # sovrappongono per più del 40% (IoU > 0.4)
    if len(boxes) > 0:
        nms_keep_idx = torchvision.ops.nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[nms_keep_idx]
        labels = labels[nms_keep_idx]
        scores = scores[nms_keep_idx]
    
    # Se ci sono troppi unknown, limitiamo la visualizzazione ai top 5 per pulizia
    final_boxes, final_labels, final_scores = [], [], []
    unk_count = 0
    for b, l, s in zip(boxes, labels, scores):
        if l.item() == 21:
            unk_count += 1
            if unk_count > 5:
                continue
        final_boxes.append(b)
        final_labels.append(l)
        final_scores.append(s)
        
    print(f"Mostrando {len(final_boxes)} oggetti dopo pulizia (NMS e Top-5 Unknown)")
    
    # 4. Plotting
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    for box, label, score in zip(final_boxes, final_labels, final_scores):
        x1, y1, x2, y2 = box.numpy()
        
        # L'etichetta 21 è "Unknown"
        is_unknown = (label.item() == 21)
        color = 'red' if is_unknown else 'green'
        text = f"UNKNOWN: {score:.2f}" if is_unknown else f"Known ({label.item()}): {score:.2f}"
        
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, text, bbox=dict(facecolor=color, alpha=0.7), fontsize=12, color='white', weight='bold')
        
    plt.axis('off')
    save_path = "/kaggle/working/visualized_output.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {save_path}! Apri il file per vedere il risultato.")

if __name__ == '__main__':
    main()
