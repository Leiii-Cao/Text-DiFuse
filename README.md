# Text-DiFuse
This is the official code of the NeurIPS 2024 paper "Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Model"
# Environmental Installation

conda create -n Text-DiFuse python==3.9.0

conda activate Text-DiFuse

Install the appropriate torch, we recommend the CUDA Version 11.8 environment torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

pip install -r requirements.txt
# Pre-trained models and data

Pre-trained diffusion model parameters and Fusion Control Module (FCM) parameters can be downloaded from https://drive.google.com/drive/folders/1LIcehq772Qd-3_OnaKmHWGGwkArN4MYg and placed in the "./pretained"

Download the infrared-visible light images and medical multimodal images that need to be inferred and place them in "./data/",We also provide some example pairing data, which can be tested directly

# Test
Select "task_type"0(#Modify parameters to achieve different fusion tasks: VIS-IR, MRI-CT, MRI-PET, MRI-SPECT#)

python test.py


