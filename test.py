import copy
import argparse
import os
import torch.nn.functional as F
import random
import torch
import cv2
from torch.optim import AdamW
from PIL import Image
from torch.utils.data import DataLoader
from diffusion_fusion.unet import Get_Fusion_Control_Model
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TestDataset, to_numpy_image
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,
                              create_model_and_diffusion,
                              model_and_diffusion_defaults)

def main(): 

    """Setup""" 
    torch.manual_seed(0)
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--Task_type', type=str, default='VIS-IR')#Modify parameters to achieve different fusion tasks: VIS-IR, MRI-CT, MRI-PET, MRI-SPECT#
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.set_defaults(timestep_respacing="25")
    args = parser.parse_args()
    
    """TestData Dataloader"""
    
    diffusion_stage1_path = "./pretrained/diffusion_stage1.pth"
    diffusion_stage2_path = "./pretrained/diffusion_stage2.pth"
    if args.Task_type == 'VIS-IR':
       VIS_path = "./data/test/VIS-IR/VIS/"
       IR_path = "./data/test/VIS-IR/IR/"
       Test_Dataset = GET_TestDataset(VIS_path,IR_path,MAX_SIZE=640)
       FCM_path = "./pretrained/FCM-VIS-IR.pt"
    elif args.Task_type == 'MRI-CT':
       MRI_path = "./data/test/MRI-CT/MRI/"
       CT_path = "./data/test/MRI-CT/CT" 
       Test_Dataset = GET_TestDataset(MRI_path,CT_path,MAX_SIZE=640)
       FCM_path = "./pretrained/FCM-Medi.pt"
    elif args.Task_type == 'MRI-PET':
       MRI_path = "./data/test/MRI-PET/MRI/"
       PET_path = "./data/test/MRI-PET/PET" 
       Test_Dataset = GET_TestDataset(MRI_path,PET_path,MAX_SIZE=640)
       FCM_path = "./pretrained/FCM-Medi.pt"
    elif args.Task_type == 'MRI-SPECT':
       MRI_path = "./data/test/MRI-SPECT/MRI/"
       SPECT_path = "./data/test/MRI-SPECT/SPECT" 
       Test_Dataset = GET_TestDataset(MRI_path,SPECT_path,MAX_SIZE=640)
       FCM_path = "./pretrained/FCM-Medi.pt"
    test_loader = DataLoader(
        Test_Dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False
    )
    os.makedirs(args.save_dir, exist_ok = True)
    
    """Model loader"""
    
    diffusion_stage1,diffusion_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    Fusion_Control_Model=Get_Fusion_Control_Model()
    diffusion_stage1.load_state_dict(torch.load(diffusion_stage1_path))
    diffusion_stage2.load_state_dict(torch.load(diffusion_stage2_path))
    Fusion_Control_Model.load_state_dict(torch.load(FCM_path),strict=False)
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    Fusion_Control_Model=Fusion_Control_Model.to(args.device)
    diffusion_stage1.eval()
    diffusion_stage2.eval()
    Fusion_Control_Model.eval()
 
    

    """Begin Test Fusion"""
    print("Text-DiFuse....Begin Test Fusion.....")
    for i, input in enumerate(test_loader):
        cond={'condition': input[0].to(args.device)}
        cond1={'condition': input[1].to(args.device)}
        filename = os.path.basename(str(input[4]))[:-3]
        output = diffusion.p_sample_loop(
            diffusion_stage1, 
            diffusion_stage2,
            Fusion_Control_Model,
            input[0].shape,
            couple_single=True,
            model_kwargs=cond,
            model_kwargs1=cond1,
            progress=True,
        )  
        output = to_numpy_image(torch.cat((output,input[2].to(args.device),input[3].to(args.device)),dim=1))
        output5= cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
        output_name = filename
        Image.fromarray(output5).save(os.path.join(args.save_dir, output_name))     


if __name__ == "__main__":
    main()