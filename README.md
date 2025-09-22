<p align="center">
  <h1 align="center">[Reimplementation] HRAvatar: High-Quality and Relightable Gaussian Head Avatar</h1>
<p align="center">
  
## 📌 Overview
This repository contains my reimplementation of the CVPR 2025 paper "HRAvatar: High-Quality and Relightable Gaussian Head Avatar" by Zhang et al. The original implementation is available at Pixel-Talk/HRAvatar. HRAvatar uses 3D Gaussian Splatting to reconstruct high-fidelity, relightable 3D head avatars, achieving real-time rendering and realistic visual effects under varying lighting conditions. I tested this implementation with the HDTF dataset and a custom video.

| <img src="assets/docs/readme_figs/pipeline.png" alt="Pipeline" style="background:white; padding:10px; border-radius:10px;" /> |
| :----------------------------------------------------------: |
| Pipeline of HRAvatar |

## 📸 Sample Results

<p align="center">
  <img src="assets/results/custom_video_front.gif" width="300"/>
  <img src="assets/results/HDTF_katie_side.gif" width="300"/>
</p>

## 🖥️ Cloning the Repository
```shell
# HTTPS
git clone https://github.com/iraheers/HRAvatar-Reimplementation.git
```

The implementation was tested on Ubuntu 20.04 with an NVIDIA GPU (24 GB VRAM recommended).

## 📂 Datasets preparation

- HDTF Dataset: Download videos from [here](https://drive.google.com/drive/folders/1lJMrNuvCSCDwMsd6Pz7W3cH_jXPt_fKv?usp=sharing).
- Custom Video: Ensure your video is preprocessed (e.g., face detection, mask extraction) as described below.


## 🛠️ Setup

#### Optimizer
The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

#### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

#### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.7)
- C++ Compiler and CUDA SDK must be compatible

### Environment Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate HRAvatar
cd submodules
git clone https://github.com/NVlabs/nvdiffrast.git
pip install nvdiffrast
pip install diff-gaussian-rasterization_c10
pip install simple-knn
```
Note: If you encounter dependency issues, ensure fsspec[http]>=2023.5.0 is installed, as it resolved an issue during my implementation.

## 🔧 Data Preprocessing

#### Frame Cropping and Matting
For HDTF or custom videos:
1. Install dependencies:
  - Download face-parsing model (79999_iter.pth) and place in preprocess/submodules/face-parsing.PyTorch/res/cp/.[Download Here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=drive_open)
  -  Download RobustVideoMatting model (rvm_resnet50.pth) and place in preprocess/submodules/RobustVideoMatting. [Download Here](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth)


2. Run
```shell
# example script
fps=30  # Adjust to video frame rate
resize=512  # Desired image size
python preprocess/crop_and_matting.py \
    --source /path/to/data \
    --name subject_name \
    --fps $fps \
    --image_size $resize $resize \
    --matting \
    --crop_image \
    --mask_clothes True
```

3. Alternatively, edit and run shell scripts in preprocess/preprocess_shell/HDTF/ or preprocess/preprocess_shell/custom/ for HDTF or custom data.

#### Albedo Extraction
Required for HDTF or videos with local lighting:
1. Download pretrained weights from [Hugging Face](https://huggingface.co/LittleFrog/IntrinsicAnything) and place in assets/intrinsic_anything/albedo.

2. Run
```shell
# example script
base_dir=/path/to/subject
python preprocess/submodules/IntrinsicAnything/inference.py \
    --input_dir $base_dir/image \
    --model_dir assets/intrinsic_anything/albedo \
    --output_dir $base_dir/albedo \
    --ddim 100 --batch_size 10 --image_interval 3
```
Note: The original albedo URL was invalid; I used the Hugging Face link above.

#### Facial Tracking
1. Install dependencies:
  - Download [deca_model.tar](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view) and generic_model.pkl from [FLAME2020](https://flame.is.tue.mpg.de/download.php) (rename to generic_model2020.pkl).
  - Place in preprocess/submodules/DECA/data.
  - Note: I used FLAME2023.
  - If assets/flame_model/flame_2020.pkl is missing, copy preprocess/submodules/DECA/data/generic_model2020.pkl to assets/flame_model/ and rename it to flame_2020.pkl.

2. Run:
```shell
# example script
cd preprocess/submodules/DECA

base_dir=/path/to/subject
python demos/demo_reconstruct.py \
    -i $base_dir/image \
    --savefolder $base_dir/deca \
    --saveCode True \
    --saveVis False \
    --sample_step 1 \
    --render_orig False

cd ../..
python keypoint_detector.py --path $base_dir
python iris.py --path $base_dir
fx=1539.67462
fy=1508.93280
cx=261.442628
cy=253.231895
resize=512

cd preprocess/submodules/DECA
python optimize.py --path $base_dir \
    --cx $cx --cy $cy --fx $fx --fy $fy \
    --size $resize --n_shape 100 \
    --n_expr 100 --with_translation
```

For more details on data preprocessing, refer to [Data_Preprocessing](assets/docs/Data_Preprocessing.md)

Environment map filtering is described in [Filter_Envmap](assets/docs/Filter_Envmap.md)


## 🎯 Traning

For HDTF
```shell
# example script
CUDA_VISIBLE_DEVICES=0  python train.py --source_path /path/to/marcia \
  --model_path outputs/HDTF/marcia  --eval  --test_set_num 500  --epochs 15 \
  --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist
```

For Custom DATASET
```shell
# example script
# Note: Lower learning rates can lead to better geometry 
#       but may degrade quantitative metrics (e.g., PSNR, SSIM)
CUDA_VISIBLE_DEVICES=0  python train.py --source_path /path/to/subject \
  --model_path outputs/custom/subject  --eval  --test_set_num 500  --epochs 15 \
  --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist \
  --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 \
  --position_lr_init 5e-5 --position_lr_final 5e-7
```


## 🎨 Rendering

Render the training and testing results  
(This is automatically done after training by default)
```shell
# example script
CUDA_VISIBLE_DEVICES=0 python render.py  --model_path outputs/insta/bala
```

Render others
Add arguments in render.py
```shell
--skip_test # Skip rendering self-reenactment test set results
--skip_train # Skip rendering self-reenactment training set results
--render_albedo # Render albedo component
--render_normal # Render normal component
--render_irradiance # Render irradiance component
--render_specular # Render specular component
--render_roughness  # Render roughness component
--render_reflectance # Render reflectance component
--render_depth  # Render depth map
--render_envmap # Visualize optimized environment map
--render_relighting # Perform relighting render
--with_relight_background # Use input environment map as background during relighting
--envmap_path assets/envmaps/cobblestone_street  # Filtered environment map for relighting
--render_material_editing # Render material editing results (gradually increase reflectance)
--corss_source_path  # Render cross-reenactment results (specify the processed data path of another subject)
--test_static_material_edting_idxs 100 # Apply material editing to a specific image
--test_static_relight_idxs 100  # Apply relighting to a specific image
```

### Evaluation
(This is automatically done after training by default)
```shell
# example script
python metrics.py --model_path outputs/insta/bala
```

## 📖 Citation
If you use this reimplementation, please cite the original paper:
```bibtex
@InProceedings{HRAvatar,
    author    = {Zhang, Dongbin and Liu, Yunfei and Lin, Lijian and Zhu, Ye and Chen, Kangjie and Qin, Minghan and Li, Yu and Wang, Haoqian},
    title     = {HRAvatar: High-Quality and Relightable Gaussian Head Avatar},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26285-26296}
}
```

## 🙏 Acknowledgement
- The original authors of HRAvatar (Pixel-Talk/HRAvatar) for their open-source implementation.
- GraphDeco-Inria for the 3D Gaussian Splatting framework.
- INSTA and HDTF for providing datasets.
- IntrinsicAnything, DECA, FLAME, RobustVideoMatting, and face-parsing.PyTorch for preprocessing tools. 
