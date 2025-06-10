# [NeurIPS 2024 spotlight] Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Model
This repository is the official implementation of the **NeurIPS 2024** paper:
_"Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Model"_ 
### [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/45e409b46bebd648e9041a628a1a9964-Abstract-Conference.html) | [Arxiv](https://arxiv.org/abs/2410.23905) | [Poster](https://neurips.cc/virtual/2024/poster/93032) 


![PAMI](https://github.com/user-attachments/assets/f3c2efa1-2590-4be0-8a4b-ce82c676b02e)

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{zhang2024text,
  title={Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Model}, 
  author={Zhang, Hao and Cao, Lei and Ma, Jiayi},
  journal={Advances in Neural Information Processing Systems},
  volume={37}, 
  pages={39552--39572},
  year={2024}
  url={https://proceedings.neurips.cc/paper_files/paper/2024/hash/45e409b46bebd648e9041a628a1a9964-Abstract-Conference.html}
}
```
## Contact me
If you have any questions or discussions, please send me an email:
```
whu.caolei@whu.edu.cn
```


## Environmental Installation

```
conda create -n Text-DiFuse python==3.9
conda activate Text-DiFuse
```
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install -r requirements.txt
```
## ❄️ Test Code
### Prepare Your Dataset
Download the public datasets [MSRS](https://github.com/Linfeng-Tang/MSRS), [RoadScene](https://github.com/jiayi-ma/RoadScene), [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029), [LLVIP](https://github.com/bupt-ai-cz/LLVIP), [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), and [Harvard](https://www.med.harvard.edu/AANLIB/home.html), and place them in the following directory: 
```
./data/test/
```
### Pretrained weights

You can download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1LIcehq772Qd-3_OnaKmHWGGwkArN4MYg) and place them in the following directory: 
```
./pretrained/
```
### Run 
After modifying configurable parameters such as **task_type** and **timestep**, you can directly run the code:
```
python test.py
```
### Run the modulation mode code
If you want to test the modulation mode, please first download the pretrained model weights for [OWL-ViT](https://huggingface.co/google/owlvit-large-patch14) and [SAM](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth), and place them at the following path:
```
./modulated/checkpoint/
```
You can modify parameter **text_prompt**, and then run the code:
```
python test_modulated.py
```
## 🔥 Train code
### Train the diffusion model
Place your own training data in the directory:
```
./data/train_diffusion/
```
And then run the code:
```
python train_diffusion.py
```
### Train the FCM model
Place your own training data in the directory:
```
./data/train_FCM/
```
And then run the code:
```
python train_FCM.py
```
