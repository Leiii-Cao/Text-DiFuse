import copy
import argparse
import os
import torch.nn.functional as F
import random
import torch
import cv2
import random
from torch.optim import AdamW
from PIL import Image
from torch.utils.data import DataLoader
from diffusion_fusion.unet import Get_Fusion_Control_Model
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TrainDataset, to_numpy_image,GET_TestDataset
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
    parser.add_argument('--save_dir', type=str, default='./log_FCM')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.set_defaults(timestep_respacing="25")
    args = parser.parse_args()
    
    """TrainData Dataloader"""
    diffusion_stage1_path = "./pretrained/diffusion_stage1.pth"
    diffusion_stage2_path = "./pretrained/diffusion_stage2.pth"
    if args.Task_type == 'VIS-IR':
       VIS_path = "./data/train_FCM/VIS-IR/VIS/"
       IR_path = "./data/train_FCM/VIS-IR/IR/"
       VIS_path1 = "./data/test/VIS-IR/VIS/"
       IR_path1 = "./data/test/VIS-IR/IR/"
       Train_Dataset = GET_TrainDataset(VIS_path,IR_path,MAX_SIZE=320,CROP_SIZE=320)
       Test_Dataset = GET_TestDataset(VIS_path1,IR_path1,MAX_SIZE=640)
    elif args.Task_type == 'Medi':
       MRI_path = "./data/train_FCM/MRI-CT/MRI/"
       CT_path = "./data/train_FCM/MRI-CT/CT" 
       MRI_path1 = "./data/test/MRI-CT/MRI/"
       CT_path1 = "./data/test/MRI-CT/CT"
       Train_Dataset = GET_TrainDataset(MRI_path,CT_path,MAX_SIZE=320,CROP_SIZE=320)
       Test_Dataset = GET_TestDataset(MRI_path1,CT_path1,MAX_SIZE=640)

    train_loader = DataLoader(
        Train_Dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False
    )
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
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    Fusion_Control_Model=Fusion_Control_Model.to(args.device)
    optimizer = AdamW(Fusion_Control_Model.parameters(), lr=2e-5, weight_decay=0.0)
    diffusion_stage1.eval()
    diffusion_stage2.eval()
    Fusion_Control_Model.train()

    """Begin Train FCM"""
    for epoch in range(args.epoch):
        print("Text-DiFuse....Begin Train Fusion.....")
        for i, input in enumerate(train_loader):
            cond={'condition': input[0].to(args.device)}
            cond1={'condition': input[1].to(args.device)}
            filename = os.path.basename(str(input[4]))[:-3]
            """GET GT"""
            output_GT1 = diffusion.p_sample_loop(
                diffusion_stage1, 
                diffusion_stage2,
                Fusion_Control_Model,
                input[0].shape,
                couple_single=False,
                model_kwargs=cond,
                model_kwargs1=cond,
                progress=True,
            )  
        
            output_GT2 = diffusion.p_sample_loop(
                diffusion_stage1, 
                diffusion_stage2,
                Fusion_Control_Model,
                input[0].shape,
                couple_single=False,
                model_kwargs=cond1,
                model_kwargs1=cond1,
                progress=True,
            )  
       
            """Begin Train FCM"""

            indices = list(range(diffusion.num_timesteps))[::-1]
            torch.manual_seed(0)
            x_F_t=torch.randn(input[0].shape, device=args.device)
            number = random.randint(0, 24)
            for timestep in indices:
                t = torch.tensor([timestep] * x_F_t.shape[0], device=args.device)
                if timestep>number:
                    out =diffusion.p_mean_variance(
                        diffusion_stage1, 
                        diffusion_stage2,
                        Fusion_Control_Model,
                        x_F_t, 
                        t, 
                        model_kwargs=cond,
                        model_kwargs1=cond1,
                    )
                    x_F_t=x_F_t.detach()
                    sample = out["mean"] 
                    x_F_t=sample.detach()
                elif timestep==number:
                    optimizer.zero_grad()
                    loss,out =diffusion.train_FCM_loss(
                        diffusion_stage1, 
                        diffusion_stage2,
                        Fusion_Control_Model,
                        x_F_t, 
                        t, 
                        output_GT1,
                        output_GT2,
                        model_kwargs=cond,
                        model_kwargs1=cond1,
                    )
                    loss.backward()
                    optimizer.step()
                else:
                    pass

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
            out_dir = os.path.join(args.save_dir, f'{epoch}')
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(output5).save(os.path.join(out_dir,filename))    
           
    ckpt_dir = os.path.join(args.save_dir, args.Task_type+'_checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)
    FCM_save_path = os.path.join(ckpt_dir, f'FCM.pt')
    torch.save(Fusion_Control_Model.state_dict(), FCM_save_path)

if __name__ == "__main__":
    main()