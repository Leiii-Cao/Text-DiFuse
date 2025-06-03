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

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from modulated.segment_anything.segment_anything import build_sam, SamPredictor
from modulated.util import load_owlvit,OWL_VIT_SAM,resize_and_align16_batch

def main(): 

    """Setup""" 
    torch.manual_seed(0)
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./result_modulated')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.set_defaults(timestep_respacing="25")
    args = parser.parse_args()
    
    """TestData Dataloader"""
    
    diffusion_stage1_path = "./pretained/diffusion_stage1.pth"
    diffusion_stage2_path = "./pretained/diffusion_stage2.pth"
    VIS_path = "./data/test/modulated/VIS/"
    IR_path = "./data/test/modulated/IR/"
    Test_Dataset = GET_TestDataset(VIS_path,IR_path,MAX_SIZE=640)
    FCM_path = "./pretained/FCM-VIS-IR.pt"

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
    Fusion_Control_Model_modulated=Get_Fusion_Control_Model()
    diffusion_stage1.load_state_dict(torch.load(diffusion_stage1_path))
    diffusion_stage2.load_state_dict(torch.load(diffusion_stage2_path))
    Fusion_Control_Model.load_state_dict(torch.load(FCM_path),strict=False)
    Fusion_Control_Model_modulated.load_state_dict(torch.load(FCM_path),strict=False)
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    Fusion_Control_Model=Fusion_Control_Model.to(args.device)
    Fusion_Control_Model_modulated=Fusion_Control_Model_modulated.to(args.device)
    optimizer = AdamW(Fusion_Control_Model_modulated.parameters(), lr=2e-5, weight_decay=0.0)
    diffusion_stage1.eval()
    diffusion_stage2.eval()
    Fusion_Control_Model.eval()
 
    """OWT-VIT-SAM Model loader"""
    OWL_VIT_model, processor = load_owlvit(checkpoint_path="./modulated/checkpoint/owlvit-base-patch32/", device=args.device)
    predictor = SamPredictor(build_sam(checkpoint="./modulated/checkpoint/sam_vit_h_4b8939.pth").to(args.device))
    

    """Begin Test Fusion"""
    print("Text-DiFuse....Begin Test Fusion.....")
    for i, input in enumerate(test_loader):
        cond={'condition': input[0].to(args.device)}
        cond1={'condition': input[1].to(args.device)}
        cond_modulated,cond1_modulated,target_modulated,target=OWL_VIT_SAM(batch=input[0].to(args.device),batch1=input[1].to(args.device),input=input,text_prompt="people,car,bike",processor=processor,model=OWL_VIT_model,predictor=predictor)
        torch.manual_seed(0)
        x_F_t=torch.randn(input[0].shape, device=args.device)
        torch.manual_seed(0)
        x_F_t_m=torch.randn(input[0].shape, device=args.device)
        torch.manual_seed(0)
        x_F_t_modulated=torch.randn(target_modulated.shape, device=args.device)
        filename = os.path.basename(str(input[4]))[:-3]
        indices = list(range(diffusion.num_timesteps))[::-1]
        for timestep in indices:
            t = torch.tensor([timestep] * x_F_t.shape[0], device=args.device)
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
            output_xt=out['mean']
            output_GT=resize_and_align16_batch(out['pred_xstart'])
            optimizer.zero_grad()
            loss,out =diffusion.modulated_loss(
                diffusion_stage1, 
                diffusion_stage2,
                Fusion_Control_Model_modulated,
                x_F_t_modulated, 
                t, 
                output_GT,
                target_modulated,
                model_kwargs=cond_modulated,
                model_kwargs1=cond1_modulated,
            )
            loss.backward()
            optimizer.step()
            x_F_t_modulated=x_F_t_modulated.detach()
            output_xt_modulated_resize16=out['mean']
            sample_m=output_xt_modulated_resize16
            x_F_t_modulated=sample_m.detach()
            out =diffusion.p_mean_variance(
                diffusion_stage1, 
                diffusion_stage2,
                Fusion_Control_Model_modulated,
                x_F_t_m, 
                t, 
                model_kwargs=cond,
                model_kwargs1=cond1,
            )
            x_F_t_m=x_F_t_m.detach()
            output_xt_m=out['mean']
            sample_m = output_xt_m*target+output_xt*(1-target)
            sample=output_xt
            x_F_t_m=sample_m.detach()
            x_F_t=sample.detach()
        output = to_numpy_image(torch.cat((sample_m,input[2].to(args.device),input[3].to(args.device)),dim=1))
        output5= cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
        output_name = filename[:-4]+'_modulated.png'
        Image.fromarray(output5).save(os.path.join(args.save_dir, output_name))  
        output = to_numpy_image(torch.cat((sample,input[2].to(args.device),input[3].to(args.device)),dim=1))
        output5= cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
        output_name = filename
        Image.fromarray(output5).save(os.path.join(args.save_dir, output_name)) 
            


if __name__ == "__main__":
    main()