# Brain Tumor Segmentation Using YOLO




This project implements **brain tumor segmentation** using **YOLOv11**. The goal is to detect and segment brain tumors from MRI images, utilizing a custom-trained 
YOLOv11 model. The dataset is sourced from **Roboflow** for easy management and version control.



![image](https://github.com/user-attachments/assets/a4dec2a2-1a51-4593-9c5a-8655b712b299)


## Table of Contents

1. [Project Overview](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#project-overview)

2. [Dataset](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#dataset)

3. [Setup and Installation](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#setup-and-installation)

4. [Model Training](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#model-training)

5. [Validation](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#validation)

6. [Prediction](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#prediction)

7. [Results](https://github.com/elnemr19/Brain-Tumor-Segmentation-Using-YOLO/blob/main/README.md#results)


## Project Overview

This project aims to create a robust YOLOv11 model capable of detect and segment brain tumer in images. By leveraging Roboflow for dataset preparation and YOLOv11 for training, 
the project achieves high accuracy in detecting and segmenting brain tumor in images.



## Dataset

The dataset is sourced from Roboflow, containing labeled images of brain tumor. The dataset includes 
training, validation, and test sets for optimal model performance.

**Key Details:**

* **Number of Classes:** Multiple beverage container types.

* **Image Resolution:** 640x640 (optimized for YOLOv11).

* **Format:** YOLO-specific annotations.


To download the dataset, we used the following Roboflow API integration:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="QuwzZuvxyCQ2yTPnfDV9")
project = rf.workspace("iotseecs").project("brain-tumor-yzzav")
version = project.version(1)
dataset = version.download("yolov11")

```



## Setup and Installation

1. Install the necessary dependencies:
```python
  pip install roboflow ultralytics
```
2. Integrate the dataset using the Roboflow API and download the YOLOv11 configuration file.

3. Ensure the YOLOv11 environment is ready:
 ```python
  yolo check
 ```



## Model Training

To train the YOLOv11 model, execute the following command:

```python
!yolo task=segment mode=train model=yolo11s-seg.pt data={dataset.location}/data.yaml  epochs=60 imgsz=640 plots=True
```

**Training Parameters**:

* **Model:** YOLOv11 small (yolo11s-seg.pt)

* **Epochs:** 50

* **Image Size:** 640x640

* **Plots:** Enabled for visualization of training progress.




## Validation

After training, validate the model to evaluate its performance:

```python
!yolo task=segment mode=val model=/{HOME}/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml
```

**Validation Metrics**:

* **mAP (mean Average Precision)**: Indicates detection accuracy.

* **Precision and Recall**: Evaluate the balance between true positives and false negatives.



## Prediction

Use the trained model to detect and segment tumor in new images:

```python
!yolo task=segment mode=predict model=/{HOME}/runs/segment/train/weights/best.pt conf=.3 source={dataset.location}/test/images
```

**Parameters**:

* **Confidence Threshold:** 0.3

* **Source**: Test images from the dataset.

The output will include:

* Bounding boxes around detected objects.

* Class labels and confidence scores.

* Annotated images saved in the designated output directory.

## Evaluation

**Confusion Matrix**

![image](https://github.com/user-attachments/assets/f8a6d391-0604-4c70-8b13-303f8c333df2)




![image](https://github.com/user-attachments/assets/ce7c26ed-66e8-4517-9aa0-2b6d0463a9bc)





## Results

The trained YOLOv11 model achieves high accuracy in detecting and segment on new images. Below are sample detections from the test set:


![image](https://github.com/user-attachments/assets/fa856235-2bfc-4034-8ddc-df7a060e073b)






