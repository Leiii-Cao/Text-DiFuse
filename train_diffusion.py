import argparse
import copy
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import itertools
from diffusion_fusion.nn_util import EMA
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TrainDataset, to_numpy_image
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,
                              create_model_and_diffusion,
                              model_and_diffusion_defaults)
                              

def main():
    """Setup"""
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./log_diffusion')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ema_rate', type=float, default=0.9999)
    parser.add_argument('--schedule_sampler', type=str, default="uniform")
    parser.add_argument('--max_iterations', type=int, default=1500000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.set_defaults(timestep_respacing="1000")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    """TrainData Dataloader"""
    LR_path = "./data/train_diffusion/LQ/"
    HR_path = "./data/train_diffusion/HQ/"
    Train_Dataset = GET_TrainDataset(LR_path,HR_path,MAX_SIZE=512,CROP_SIZE=256)
    train_loader = DataLoader(
        Train_Dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=False,
    )
    """Model loader"""
    diffusion_stage1,diffusion_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    diffusion_stage1.train()
    diffusion_stage2.train()
    optimizer = AdamW(itertools.chain(diffusion_stage1.parameters(),diffusion_stage2.parameters()), lr=2e-5, weight_decay=0.0)
    ema = EMA(args.ema_rate)
    diffusion_stage1_ema = copy.deepcopy(diffusion_stage1)
    diffusion_stage2_ema = copy.deepcopy(diffusion_stage2)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) 
 
    """Begin Train diffusion"""
    step = 0
    it = iter(train_loader)
    while step < args.max_iterations:
        try:
            batch, batch1, *rest = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch, batch1, *rest = next(it)
        batch=batch.to(args.device)
        cond={'condition': batch1.to(args.device)}
        optimizer.zero_grad()
        t, weights = schedule_sampler.sample(batch.shape[0], args.device)
        losses = diffusion.training_losses(diffusion_stage1,diffusion_stage2, batch, t, model_kwargs=cond)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        optimizer.step()
        ema.update_model_average(diffusion_stage1_ema, diffusion_stage1)
        ema.update_model_average(diffusion_stage2_ema, diffusion_stage2)
        

        if (step + 1) % args.log_interval == 0:
            print({"loss": loss.detach().item()})
                  
        # save model
        if (step) % args.save_interval == 0:
            ckpt_dir = os.path.join(args.save_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_dir_ema = os.path.join(args.save_dir, "checkpoint_ema")
            os.makedirs(ckpt_dir_ema, exist_ok=True)
            model_save_path1 = os.path.join(ckpt_dir, f'diffusion_stage1_iter_{step}.pt')
            model_save_path2 = os.path.join(ckpt_dir, f'diffusion_stage2_iter_{step}.pt')
            model_save_path1_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage1_iter_{step}.pt')
            model_save_path2_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage2_iter_{step}.pt')
            torch.save(diffusion_stage1.state_dict(), model_save_path1)
            torch.save(diffusion_stage2.state_dict(), model_save_path2)
            torch.save(diffusion_stage1_ema.state_dict(), model_save_path1_ema)
            torch.save(diffusion_stage2_ema.state_dict(), model_save_path2_ema)
        
        step += 1

    print("End of training")




if __name__ == "__main__":
    main()