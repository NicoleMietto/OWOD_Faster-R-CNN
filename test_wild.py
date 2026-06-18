import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from OWOD_detector import OWODFasterRCNN
import os
import requests
from io import BytesIO

def download_image(url):
    print(f"Scaricamento immagine da: {url} ...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Avvio test IN THE WILD su {device}...")

    # 1. Carica il modello (usiamo il best_model.pth dell'Epoca 8)
    model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True)
    model_path = "/kaggle/working/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Errore: {model_path} non trovato!")
        return

    print("Caricamento pesi...")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()

    # 2. Immagini prese da Internet (Mucche, Cani, Persone, e un Treno per sicurezza)
    test_urls = {
        "cavallo_e_cane": "https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "gatto_sul_divano": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "treno_e_macchine": "https://images.unsplash.com/photo-1496262967815-132206202600?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "tavolo_con_cibo": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
    }

    output_dir = "/kaggle/working/wild_test_results"
    os.makedirs(output_dir, exist_ok=True)

    for name, url in test_urls.items():
        try:
            image = download_image(url)
            image_tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                detections = model([image_tensor])

            pred = detections[0]
            boxes = pred['boxes'].cpu()
            labels = pred['labels'].cpu()
            scores = pred['scores'].cpu()

            image_to_draw = (image_tensor.cpu() * 255).to(torch.uint8)
            colors, text_labels, keep_boxes, keep_scores_list = [], [], [], []

            for box, label, score in zip(boxes, labels, scores):
                if score > 0.65: # Soglia un po' più alta per immagini wild
                    keep_boxes.append(box)
                    keep_scores_list.append(score)
                    if label.item() == 81:
                        colors.append("red")
                        text_labels.append(f"UNK: {score:.2f}")
                    else:
                        colors.append("green")
                        text_labels.append(f"KN {label.item()}: {score:.2f}")

            if len(keep_boxes) > 0:
                keep_boxes = torch.stack(keep_boxes)
                keep_scores_tensor = torch.stack(keep_scores_list)
                
                # Visual NMS
                final_keep = torchvision.ops.nms(keep_boxes, keep_scores_tensor, iou_threshold=0.3)
                
                keep_boxes = keep_boxes[final_keep]
                colors = [colors[i] for i in final_keep]
                text_labels = [text_labels[i] for i in final_keep]

                drawn_image = torchvision.utils.draw_bounding_boxes(
                    image_to_draw, keep_boxes, labels=text_labels, 
                    colors=colors, width=4, font_size=15
                )
            else:
                drawn_image = image_to_draw

            save_path = os.path.join(output_dir, f"{name}.jpg")
            img_pil = F.to_pil_image(drawn_image)
            img_pil.save(save_path)
            print(f"✅ Salvato: {save_path} ({len(keep_boxes)} oggetti trovati)")
            
        except Exception as e:
            print(f"❌ Errore con l'immagine {name}: {e}")

    print(f"\n🎉 Finito! Trovi le immagini analizzate in '{output_dir}'.")

if __name__ == "__main__":
    main()
