# Document Image Indexing and Visual Similarity Search

## Project Description

This academic mini-project focuses on document image indexing and visual similarity search.

The objective is to retrieve documents that are visually similar to a given query image based on their visual characteristics, without relying on OCR or textual content.

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-image
- Jupyter Notebook

## Methodology

### 1. Image Preprocessing
- Grayscale conversion
- Image resizing (256 × 192)
- Otsu thresholding

### 2. Feature Extraction

The project implements six visual descriptors:

- Layout
- Density
- Projection Profiles
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- Edge Histogram

### 3. Similarity Search

- Feature normalization
- Feature combination
- Cosine distance
- Euclidean distance
- Top-K similar document retrieval

### 4. Evaluation

The retrieval system is evaluated using:

- Precision@1
- Precision@5
- Precision@10
- Mean Average Precision (mAP)

## Dataset

The notebook uses the Tobacco3482 document image dataset.

The dataset is not included in this repository. You must place the extracted `Tobacco3482-jpg` folder next to the notebook before running it.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Execution

1. Clone or download this repository.
2. Prepare the Tobacco3482-jpg dataset.
3. Place the dataset folder in the project directory.
4. Open `indexation.ipynb` in Jupyter Notebook.
5. Execute the cells sequentially.

## Project Type

Academic Mini-Project – Image Processing and Computer Vision.