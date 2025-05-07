# Multi-View Plant Phenotyping with Vision Transformers & SAM
Plant Growth Modelling using age estimation (in days) and leaf counting

This repository contains model for estimation of **leaf count** and **plant age** of a plant using multi-view image data. The model is trained using a dataset containing images of plants at different growth stages.

The model builds on the GroMo Challenge MVVT model by integrating:

1. **SAM-generated leaf instance masks** (via the Leaf Only SAM pipeline)  
2. A **lightweight CNN backbone** for local feature extraction  
3. **Cross‐view transformer attention** to fuse information across perspectives  

Results on the mustard subset of the GroMo25 dataset show dramatic error reductions (MAE ↓57–81%) over the original MVVT.

## Important Parameters
- **Number of Plants (`n_plants`)**: Defines how many different plants are included in the dataset.
- **Maximum Days of Crop (`max_days`)**: The maximum number of days considered for plant growth.
- **Number of Multi-View Images (`n_images`)**: The number of images selected from the total 24 available multi-view images per plant.

## GroMo Dataset Structure
The dataset consists of images of multiple plants (`p1`, `p2`, ..., `pn`) captured over different days (`d1`, `d2`, ..., `dm`) and categorized into five levels (`L1`, `L2`, `L3`, `L4`, `L5`). Each plant has **24 images** per growth cycle, representing different angles with a **15-degree gap** between consecutive images.

### Naming Convention
Each image follows the format:
```
radish_pX_dY_LZ_A.jpg
```
where:
- `X` represents the plant ID (`p1, p2, ...`)
- `Y` represents the day (`d1, d2, ...`)
- `Z` represents the level (`L1, L2, L3, L4, L5`)
- `A` represents the angle (ranging from `0` to `345` degrees in 15-degree increments)

### Directory Structure
```
/dataset/
    ├── train/
    │   ├── p1/
    │   │   ├── d1/
    │   │   │   ├── L1/
    │   │   │   │   ├── radish_p1_d1_L1_0.png
    │   │   │   │   ├── radish_p1_d1_L1_15.png
    │   │   │   │   ├── ...
    │   │   │   │   ├── radish_p1_d1_L1_345.png
    │   │   │   ├── L2/
    │   │   │   ├── L3/
    │   │   │   ├── L4/
    │   │   │   ├── L5/
    │   │   ├── d2/
    │   │   ├── ...
    │   ├── p2/
    │   ├── ...

```
Each plant has images captured at different time points (`d1`, `d2`, ...), categorized into five levels (`L1` to `L5`), with a total of 24 images per level taken from different angles.



## Folder Structure
```
/dlproject/
    |──README.md
    |──compute masks.ipynb
    |──model.ipynb
    |──Mustard_Masks/
    |──|──masks_p1.zip
    |──|──masks_p2.zip
    |──|──masks_p3.zip
```
---

## Usage

### 1. Dataset Preparation
Download the GroMo25 dataset. Modify the dataset path in `CropDataset` to point to your dataset location.
 * Mustard Dataset at https://www.kaggle.com/datasets/hamzakhn/gromo-mustard-dataset/settings
 * Groud Truths at https://www.kaggle.com/datasets/hamzakhn/gromo-ground-truths
 * Masks for Mustard Plants at /dlproject/Mustard_Masks/

### 2. Generating Leaf Masks
Open and run compute_masks.ipynb:
* Loads each image (resized to 512×512).
* Runs SAM to produce initial masks.
* Filters leaf masks by color, shape, and overlap (Leaf Only SAM).
* Saves compressed .npz stacks per plant/day/level/angle.


### 3. Training & Evaluation
Open and run train_and_evaluate.ipynb:
* Dataset
    * Loads raw images + precomputed masks.
    * Constructs 6-channel per-view inputs: RGB + mask + leaf count + leaf area.
* Model
    * VisionTransformerCrossViewSAM: model is used for two task separetly:
        * for Leaf Count Prediction (model[0])
        * for Plant Age Estimation (model[1])
    * During training, the loss function minimizes the RMSE loss for both leaf count and plant age. The model prints training losses per epoch

### 4. Result Metrics
- **Leaf Count RMSE**: Measures the root mean squared error in predicting the number of leaves.
-  **Leaf Count MAE**: Measures the mean absolute error in predicting the number of leaves.
- **Age RMSE**: Measures the root mean squared error  in predicting the plant’s age.
- **Age MAE**: Measures the mean absolute error in predicting the plant’s age.

## 5. Outputs
* Training curves 
* Checkpoints