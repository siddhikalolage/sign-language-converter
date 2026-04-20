# Final Project

This folder is the final review-ready project package for the image-based phrase recognition system.

It contains the final two approaches we agreed on:

- Landmark-based prediction
  Main final approach
- YOLO-based prediction
  Backup and custom-dataset-friendly approach

## Final Main Files

- `predict_single_gesture.py`
  Final landmark live predictor
- `predict_phrase_yolo.py`
  Final YOLO live predictor
- `create_text_gesture_data_from_images.py`
  Creates landmark CSV dataset from phrase image folders
- `train_text_gesture_model.py`
  Trains the landmark classifier
- `test_text_gesture_model.py`
  Tests the landmark classifier
- `prepare_yolo_classification_dataset.py`
  Creates YOLO train/val/test image splits
- `train_yolo_phrase_model.py`
  Trains the YOLO image classifier

## Core Helper Files

- `gesture_pipeline.py`
  Landmark extraction and quality scoring
- `gesture_features.py`
  Landmark feature engineering
- `gesture_dataset.py`
  Dataset loading and splitting
- `torch_gesture_model.py`
  Helper module imported by the training script

## Included Final Models

- `text_phrase_image_model.pkl`
- `text_phrase_image_label_encoder.pkl`
- `text_phrase_yolo_cls.pt`
- `text_phrase_yolo_cls.metrics.json`
- `holistic_landmarker.task`

## Included Final Data Folders

- `images for phrases`
  Original phrase image dataset
- `text_phrase_image_data`
  Extracted landmark CSV dataset
- `yolo_phrase_dataset`
  YOLO-ready split dataset

## Recommended Demo Run

Run the main landmark predictor:

```powershell
python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl
```

Run the YOLO backup predictor:

```powershell
python predict_phrase_yolo.py --model-path text_phrase_yolo_cls.pt --class-map yolo_phrase_dataset/class_name_map.json
```

## Recreate Landmark Dataset

```powershell
python create_text_gesture_data_from_images.py
```

## Train Landmark Model

```powershell
python train_text_gesture_model.py --data-folder text_phrase_image_data --backend sklearn --split-strategy frame --model-output text_phrase_image_model.pkl --encoder-output text_phrase_image_label_encoder.pkl
```

## Test Landmark Model

```powershell
python test_text_gesture_model.py --data-folder text_phrase_image_data --backend sklearn --split-strategy frame
```

## Recreate YOLO Dataset

```powershell
python prepare_yolo_classification_dataset.py --source-folder "images for phrases" --output-folder yolo_phrase_dataset --overwrite
```

## Train YOLO Model

```powershell
python train_yolo_phrase_model.py --dataset-folder yolo_phrase_dataset --model yolov8n-cls.yaml --epochs 25 --imgsz 224 --batch 16 --device cpu --save-model-to text_phrase_yolo_cls.pt
```
