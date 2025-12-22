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
### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```


## Dataset preparation

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


 

### Finetuning

After completing pretraining, the pretrained weights should be transferred to the model for finetuning. Run the following to get init weights:

```bash
cd weights/
python pt2ft.py
```



The pretrained and finetuned model weights are provided at [here](https://huggingface.co/JielunPeng/HAVIC/) for convenience and reproducibility. 

## Inference
For inference the finetuned model. Please put your finetuned model under ./weights, or please download our finetuned weight. Then:

```bash
cd evaluation
bash swi.sh
```

The model outputs a deepfake probability score for each input video.



## Acknowledgement

We appreciate the following github repos for their valuable code and contributions:

- MARLIN (https://github.com/ControlNet/MARLIN)
- AudioMAE (https://github.com/facebookresearch/AudioMAE)
- OpenAVFF (https://github.com/JoeLeelyf/OpenAVFF)



## Contact

If you have any questions or concerns, please contact:

📧 **[jielunpeng.hit@gmail.com](mailto:jielunpeng.hit@gmail.com)**

📧 **[25s003052@stu.hit.edu.cn](25s003052@stu.hit.edu.cn)**

or feel free to submit an issue in this repository.
