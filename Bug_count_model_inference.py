import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp6/weights/best.pt')

# Inference
img_path = "C:/Users/dell/Downloads/IMG_4081.jpg"  
results = model(img_path)

# Results
results.print()       # Print results to console
results.show()        # Display image with predictions
results.save()        # Save image with predictions to 'runs/detect/exp'
