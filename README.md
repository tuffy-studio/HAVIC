<div align="center">
  <h2><b> Leave No Stone Unturned: Uncovering Holistic Audio-Visual Intrinsic Coherence for Deepfake Detection </b></h2>
</div>

<div align="center">

</div>

<p align="center">
  <a href="https://github.com/tuffy-studio/HAVIC/">
    <img src="https://img.shields.io/badge/Github-HAVIC-black?logo=github">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="">
    <img src="https://img.shields.io/badge/CVF Open Access-HAVIC-blue">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/JielunPeng/HAVIC/">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-HAVIC-ffd21e">
  </a>
</p>



## Overview
This repository contains the official implementation of [Leave No Stone Unturned: Uncovering Holistic Audio-Visual Intrinsic Coherence for Deepfake Detection](#). The proposed **HiFi-AVDF** dataset is available at [here](https://huggingface.co/datasets/JielunPeng/HiFi-AVDF).

![alt text](assets/ptnft.png)

## Requirements

### 1. Create a conda environment and activate it

```bash
conda create -n HAVIC python=3.10 ffmpeg
conda activate HAVIC
```
### 2. Install Python ependencies

```bash
pip install -r requirements.txt
```


## Dataset Download

We pre-train the HAVIC on the **LRS2** dataset, which contains only real videos. Please download LRS2 at [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). Due to copyright we cannot release the data. 



We use the **FakeAVCeleb** dataset to finetune the model and evaluate intra-dataset performance. Please follow the instructions on their [official site](https://sites.google.com/view/fakeavcelebdash-lab/) to request and download the dataset.

To further evaluate the cross-dataset generalization of the model, we use **KoDF** and **HiFi-AVDF** dataset:

* **KoDF official page**: [KoDF](https://deepbrainai-research.github.io/kodf/)
* **HiFi-AVDF download page**: [HiFi-AVDF](#)





## Training

### Pretraining
We pretrain HAVIC using large-scale real videos dataset LRS2 to learn intrinsic audio–visual coherence.

Following [AVFF](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html), We initialize the audio and visual encoder–decoders separately with the pretrained weights of **AudioMAE** and **MARLIN**. 
Download the official pretrained [AudioMAE]() and [MARLIN]() model weights and place them in the `weights/` folder. Then Run the following to get init weights:

```bash
cd weights/
python initialize_model.py
```
"./weights/model_to_be_ft.pth"

 
**The pretrained model weights is provided at [here](https://huggingface.co/JielunPeng/HAVIC/).**

### Finetuning

#### Step 1: Data Preprocessing

First, we randomly split the FakeAVCeleb dataset into **70% training** and **30% testing** sets by running the following command:

```bash
# Make sure you are in the project root directory.
cd ./video_data_engine/
python split_FakeAVCeleb_dataset.py \
    --dataset_root <path to dataset root, e.g., /data/FakeAVCeleb_v1.2> \
    --training_set_ratio 0.7
```
This will produce two CSV files: `training_set.csv` for the training split and `validation_set.csv` for the validation split under the `video_data_engine` directory. Each file contains a single column: `video_path`.


Then, we perform preprocessing on the two splits separately to crop the face regions and prepare the corresponding data labels：

```bash
# Make sure you are in the project root directory.
cd ./video_data_engine/
python preprocess_ft_dataset.py \
    --training_set_csv <path to the training set csv file> \
    --validation_set_csv <path to the validation set csv file>
```
This step produces two new CSV files containing the preprocessed data for training and validation: `processed_training_set.csv` and `processed_training_set.csv`. Each file contains five columns: `video_path,face_crop_folder,audio_label,visual_label,overall_label`.

#### Step 2: Initialize Weights for Finetuning

After pretraining, the pretrained weights need to be transferred to be loaded into the model for finetuning. Please put the pre-trained weights or the weights downloaded from our release into the `./weights` directory, then run the following command to obtain the initial weights for finetuning:

```bash
# Make sure you are in the project root directory.
cd ./weights/
python pt2ft.py
```
This step will generate a `model_to_be_ft.pth` file, which contains the initialized model weights for finetuning, and a `newly_added_modules.txt` file, which is a list recording the newly added modules that will be trained at a larger learning rate during the finetuning stage.

#### Step 3: Start Finetuning

After the above steps, you can start the finetuning process by running the following command. Note that you need to configure several settings in the shell script `finetune.sh` in advance, including the path of  **model saving directory**, etc.
```bash
# Make sure you are in the project root directory.
cd ./scripts/
bash finetune.sh
```


**The finetuned model weights is also provided at [here](https://huggingface.co/JielunPeng/HAVIC/).** 

## Evaluation and Inference
Before evaluation or inference, please prepare your fine-tuned model, or download the model provided by us.

To evaluate or run inference on videos, please first organize the input videos into a CSV file. You may use [our provided code](evaluation/make_label_csv.py) or prepare the CSV using your own method.

For evaluation, the CSV file should contain two columns: `video_path, overall_label`, where `video_path` is the absolute path to the video file, and `overall_label` indicates the ground-truth label of the sample. For inference, the CSV file should contain a single column: `video_path`.


>❗**Note:** No additional video pre-processing is required, as the entire video will be automatically processed using a sliding-window strategy during evaluation and inference, and the face detection module from FaceX-Zoo is integrated into the pipeline. During execution, a temporary directory named sliding_window_inference_tmp will be created in the current working directory to store intermediate files.

Then you can run evaluation or inference using the following commands:

```bash
# Make sure you are in the project root directory.
cd ./evaluation/
python sliding_window_infer.py \
    --csv_file_path <path_to_input_csv> \
    --save_csv_path <path_to_output_csv> \
    --finetune_path <path_to_finetune_weight> \
    --mode <evalution or inference>
```

For each input video, the model outputs a deepfake probability score, indicating the likelihood that the video is manipulated (1:fake; 0:real). The prediction results will be saved to  `save_csv_path`, where each row contains a video path, the ground-truth label (if in evaluation mode), and predicted probability.


## Acknowledgement

We appreciate the following github repos for their valuable code and contributions:

- MARLIN (https://github.com/ControlNet/MARLIN)
- AudioMAE (https://github.com/facebookresearch/AudioMAE)
- OpenAVFF (https://github.com/JoeLeelyf/OpenAVFF)
- FaceX-Zoo (https://github.com/JDAI-CV/faceX-Zoo)


## Contact

If you have any questions or concerns, please contact:

📧 **[jielunpeng.hit@gmail.com](mailto:jielunpeng.hit@gmail.com)**

📧 **[25s003052@stu.hit.edu.cn](25s003052@stu.hit.edu.cn)**

or feel free to submit an issue in this repository.
