# SPIPose
SPIPose uses single-pixel sampling values or aliasing imaging results, 
combined with machine learning algorithms (such as deep neural networks), 
to infer the pose of a human or other target.

## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n animalrtpose python=3.10
conda activate animalrtpose
```

2. Install PyTorch (rqeuired versio >= 1.8) and torchvision following 
the [official instructions](https://pytorch.org/). We use PyTorch 2.5.1+cu118.

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

3. Clone the ultralytics repository and install the required dependencies.

```
git clone https://github.com/wux024/ultralytics.git
cd ultralytics
git checkout animalrtpose
pip install -v -e .
```

## Data Preparation

The construction of the dataset primarily relies on two methods: One is based on the single-pixel 
imaging principle to transform ordinary images to obtain single-pixel sampling values or aliased 
imaging results, with related methods detailed in papers [1][2]. 
The other involves using a single-pixel imaging system to directly image real 
objects, followed by annotation on the reconstructed images, for which the setup 
of the single-pixel imaging system and associated reconstruction algorithms can 
be found in papers [3][4]. This study employed both methods to generate the dataset, 
with the former considered as simulation experiments and the latter as field tests.

The naming convention for the dataset is as follows:
- `images-`: This prefix indicates that the dataset consists of image files.
- `[optical-field-size]x[optical-field-size]`: This segment specifies the dimensions of the 
original imaging area, or the optical field, from which the single-pixel samples are taken. 
For example, `128x128` indicates that the original imaging area is 128 pixels wide and 128 pixels 
high. **(Required)**
- `[sub-optical-field-size]x[sub-optical-field-size]`: This specifies the dimensions of the 
sub-optical fields used in the imaging process. For instance, `64x64` indicates that each 
sub-optical field is 64 pixels wide and 64 pixels high. **(Optional)**
- `[window_size]x[window_size]`: This denotes the size of the window used for sampling or 
reconstruction. For example, `2x2` indicates a 2x2 pixel window. **(Optional)**
- `[inverse or aliasing]`: This indicates whether the method used is inverse or aliasing. 
For example, `inverse` or `aliasing`. **(Optional)**
- `[imgsz]`: This specifies the normalsized size of the images in the dataset. For example, `256` indicates 
that the normalsized images size are 256x256 pixels. **(Optional)**
- `[seed]`: Indicates that a random sampling method was used, ensuring reproducibility. **(Optional)**

Example: `images-128x128-64x64-inverse-256-1234`

Note: All parts except `[optical-field-size]x[optical-field-size]` can be omitted.

## Training , Evaluation and Prediction

1. The datasets (e.g. Mouse) used SPIPose could be downloaded by contacting the corresponding author and me (<EMAIL>wux024@nenu.edu.cn). Extract the dataset to the `datasets` folder.
```text
ultralytics
├── ultralytics
├── docs
├── tests
├── tools
├── configs
├── weights
├── datasets
    │── mouse
        ├── images_
            |── train
            |── val
            |── test
        |── images-128x128-256
            |── train
            |── val
            |── test
        |── images-128x128-128x128-256
        |── images-128x128-64x64-aliasing-256
        |── images-128x128-64x64-inverse-256
        │── labels
        │── annotations
    |—— other datasets
```
The rationale behind renaming the 'images' directory to 'images_' is to accommodate the project's expansion into a broader array of scenarios. Specifically, at the start of the process, the training, evaluation, and inference scripts will temporarily rename this directory back to 'images'. Once these operations are completed, the directory name will revert to 'images_'. This practice helps to streamline the workflow and ensure that each phase of the project can be executed seamlessly without conflicts.

2. Run the following command to train the model:
```
python tools/pose_train.py --dataset mouse --device 0,1 --batch 256 --model-type spipose --models n --imgsz 640 --optical-field-sizes 128 --sub-optical-field-sizes 64 --imgsz-hadamard 256 --inverse --seed 1234
```
The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/15L-q6CETD3dC8aOknamEunZaDEVlA1na?usp=drive_link). You should move the downloaded model to the `weights` folder.

3. The training log and checkpoints will be saved in the `runs/spipose/train/mouse/spipose-n-128x128-64x64-inverse-256-1234` folder. If '--seed' is not specified, a random seed will be [started train time], e.g. 20241114.

4. Evaluate the trained model:
``` 
python tools/pose_val.py --dataset mouse --model-type spipose --models n --imgsz 640 --optical-field-sizes 128 --sub-optical-field-sizes 128 --imgsz-hadamard 256 --inverse --seed 1234
python tools/pose_modify_categories_id.py --dataset mouse --model-type spipose --models n --optical-field-sizes 128 --sub-optical-field-sizes 128 --imgsz-hadamard 256 --inverse --seed 1234
python tools/pose_coco_eval.py --dataset mouse --model-type spipose --models n --optical-field-sizes 128 --sub-optical-field-sizes 128 --imgsz-hadamard 256 --inverse --seed 1234
```

5. The evaluation results will be saved in the `runs/spipose/eval/mouse/spipose-n-128x128-64x64-inverse-256-1234` folder. If '--seed' is not specified, a random seed will be [started train time], e.g. 20241114.

6. Use the trained model to predict the pose of animals, and we provide two ways to do this:
```
python tools/pose_stream_inference.py --dataset mouse --model-type spipose --models n --imgsz 640 --optical-field-sizes 128 --sub-optical-field-sizes 128 --imgsz-hadamard 256 --inverse --seed 1234
```
or 
```
python tools/pose_normal_inference.py --source path/to/image/or/video --dataset mouse 
--model path/to/best.pt --project path/to/save_folder
```
`pose_stream_inference.py` is used for datasets, the predicted pose will be saved in the `runs/spipose/predict/mouse/spipose-n-128x128-64x64-inverse-256-1234` folder. While `pose_normal_inference.py` is used for single image or video pose estimation. The predicted pose will be saved in the `path/to/save_folder` folder.


## SPIPose Results on Benchmark Datasets

### Mouse

### Fly

### Anipose Fly

### Horse10

### Two person

## Real-world Demo



