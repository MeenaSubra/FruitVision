import cv2
import argparse
import numpy as np
import json
import os

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = tuple(map(int, COLORS[class_id]))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def load_classes(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, target_size=(416, 416)):
    return cv2.dnn.blobFromImage(image, 1/255.0, target_size, swapRB=True, crop=False)

def load_model(config_path, weights_path):
    return cv2.dnn.readNet(weights_path, config_path)

def detect_objects(image, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    Height, Width = image.shape[:2]
    blob = preprocess_image(image)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)
            
def save_image_with_boxes(image, output_dir, output_file):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    cv2.imwrite(output_path, image)
    print(f"Image with detected objects saved at: {output_path}")

# Load ground truth annotations
def load_annotations(annotation_dir):
    annotations = {}
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".json"):
            with open(os.path.join(annotation_dir, filename)) as f:
                data = json.load(f)
                image_name = filename.split(".")[0]  # Extract image name without extension
                annotations[image_name] = data["objects"]
    return annotations

# Calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

# Evaluate model
def evaluate(gt_annotations, pred_annotations):
    total_iou = 0
    total_objects = 0

    for image_name, gt_objects in gt_annotations.items():
        if image_name in pred_annotations:
            pred_objects = pred_annotations[image_name]

            for gt_obj in gt_objects:
                gt_box = gt_obj["points"]["exterior"]
                gt_class = gt_obj["classTitle"]
                gt_bbox = [gt_box[0][0], gt_box[0][1], gt_box[1][0], gt_box[1][1]]

                for pred_obj in pred_objects:
                    pred_box = pred_obj["bbox"]  # Assuming bbox is already extracted during inference
                    pred_class = pred_obj["class"]
                    pred_bbox = [pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]]

                    # Calculate IoU between ground truth and predicted bounding boxes
                    iou = calculate_iou(gt_bbox, pred_bbox)

                    # If IoU is above a threshold, consider it a correct detection
                    if iou > 0.5 and gt_class == pred_class:
                        total_iou += iou
                        total_objects += 1
                        break  # Only count each predicted object once

    # Calculate mean IoU
    mean_iou = total_iou / total_objects if total_objects > 0 else 0

    if total_objects == 0:
        print("No objects detected by the model.")

    return mean_iou

def main(args):
    global classes, COLORS
    classes = load_classes(args.classes)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    model = load_model(args.config, args.weights)
    image = cv2.imread(args.image)
    detect_objects(image, model, classes)
    
    # Output directory and file name
    output_dir = 'output'
    output_file = 'detected_objects.jpg'

    # Save image with bounding boxes
    save_image_with_boxes(image, output_dir, output_file)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load ground truth annotations
    annotation_dir = "ann"  # Directory containing annotation files
    gt_annotations = load_annotations(annotation_dir)

    # Run inference on evaluation images (Assuming you already have this part implemented)
    pred_annotations = {}  # Dictionary mapping image names to predicted objects

    # Evaluate model
    mean_iou = evaluate(gt_annotations, pred_annotations)
    print("Mean IoU:", mean_iou)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Input image path')
    ap.add_argument('-c', '--config', required=True, help='YOLO config file path')
    ap.add_argument('-w', '--weights', required=True, help='YOLO weights file path')
    ap.add_argument('-cl', '--classes', required=True, help='File containing class names')
    args = ap.parse_args()
    main(args)
