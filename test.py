import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re
from torchvision import transforms

from test_dataset_for_testing import dehaze_test_dataset
from model_convnext import fusion_net
from collections import OrderedDict ########## ---- CH code ---- ##########
from tqdm import tqdm ########## ---- CH code ---- ##########

parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--test_dir', type=str, default='./input_data/')
parser.add_argument('--output_dir', type=str, default='./output_result/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()
output_dir =args.output_dir
if not os.path.exists(output_dir + '/'):
    os.makedirs(output_dir + '/', exist_ok=True)
test_dir = args.test_dir
test_batch_size = args.test_batch_size

test_dataset = dehaze_test_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

########## ---- Start of CH code: Run tests for multiples weights ---- ##########
list_weight_files = ["./weights/validation_best.pkl"]   ## original weights
# list_weight_files = ["2025-03-07_17-22-01_NH2_epoch07675.pkl", 
#                      "2025-03-07_17-22-01_NH2_epoch03880.pkl", 
#                      "2025-03-07_17-22-01_NH2_epoch01940.pkl"]  ## SET VARIABLE

for weight_file in list_weight_files:
  ########## ---- End of CH code ---- ##########

  multiple_gpus = True  ## SET VARIABLE (Code by Caitlin)
  print(f"------- starting ---------")
  if multiple_gpus:
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(device_ids[1])
    print(device)

    # --- Define the network --- #
    MyEnsembleNet = fusion_net()

    # --- Multi-GPU --- #
    MyEnsembleNet = MyEnsembleNet.to(device)

    ########## ---- Start of CH code: To use given pkl file with parallel GPUs ---- ##########
    # Load checkpoint first, without wrapping in DataParallel
    checkpoint = torch.load(weight_file, map_location=device)
    print(f"Using weights from: {weight_file}")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]  # Extract the actual state dict
    else:
        state_dict = checkpoint  # Direct state_dict case

    # Remove "module." prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    # Load state dict into model
    MyEnsembleNet.load_state_dict(new_state_dict)

    # Now wrap in DataParallel
    MyEnsembleNet = nn.DataParallel(MyEnsembleNet, device_ids=device_ids)
    MyEnsembleNet = MyEnsembleNet.to(device) ## ch add  ## CHECK - MIGHT BE REMOVABLE NOW
    ########## ---- End of CH code ---- ##########
    # MyEnsembleNet.load_state_dict(torch.load(weight_file), strict=True)
  else:
    ########## ---- Start of CH code: To get around parallel GPU requirement ---- ##########
    # Load checkpoint
    print("Using single GPU only")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weight_file, map_location=device)  # Ensure checkpoint is loaded on correct device
    MyEnsembleNet = fusion_net().to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]  # Extract the actual state dict
    else:
        state_dict = checkpoint  # Direct state_dict case

    # Remove "module." prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    # Load state dict into model
    MyEnsembleNet.load_state_dict(new_state_dict)
    ########## ---- End of CH code ---- ##########

  with torch.no_grad(): ########## ---- CH code ---- ##########
    start_time = time.time() ########## ---- CH code ---- ##########

    ########## ---- CH code: adjustment to allow smaller images ---- ##########
    for batch_idx, image_sections in enumerate(tqdm(test_loader)):
      if len(image_sections) == 2:
         # Process whole image instead of cropping if image is exactly 1200x1600
         hazy_up_left, name = image_sections
         torch.cuda.empty_cache()
         hazy_up_left = hazy_up_left.to(device)
         frame_out_up_left = MyEnsembleNet(hazy_up_left)
         frame_out_up_left = frame_out_up_left.to(device)

      else:
        hazy_up_left, hazy_up_middle, hazy_up_right, hazy_middle_left, hazy_middle_middle, hazy_middle_right, hazy_down_left, hazy_down_middle, hazy_down_right, name = image_sections
      ############ End of adjustment CH code

        torch.cuda.empty_cache() ########## ---- CH code ---- ##########  ## CHECK IF AFFECTS QUALITY
        hazy_up_left = hazy_up_left.to(device)
        hazy_up_middle = hazy_up_middle.to(device)
        hazy_up_right = hazy_up_right.to(device)                
        hazy_middle_left = hazy_middle_left.to(device)
        hazy_middle_middle = hazy_middle_middle.to(device)
        hazy_middle_right = hazy_middle_right.to(device)

        hazy_down_left = hazy_down_left.to(device)
        hazy_down_middle = hazy_down_middle.to(device)
        hazy_down_right = hazy_down_right.to(device)

        frame_out_up_left = MyEnsembleNet(hazy_up_left)
        frame_out_middle_left = MyEnsembleNet(hazy_middle_left)
        frame_out_down_left = MyEnsembleNet(hazy_down_left)

        frame_out_up_middle = MyEnsembleNet(hazy_up_middle)
        frame_out_middle_middle = MyEnsembleNet(hazy_middle_middle)
        frame_out_down_middle = MyEnsembleNet(hazy_down_middle)

        frame_out_up_right = MyEnsembleNet(hazy_up_right)
        frame_out_middle_right = MyEnsembleNet(hazy_middle_right)
        frame_out_down_right = MyEnsembleNet(hazy_down_right)

        frame_out_up_left = frame_out_up_left.to(device)
        frame_out_middle_left = frame_out_middle_left.to(device)
        frame_out_down_left = frame_out_down_left.to(device)
        frame_out_up_middle = frame_out_up_middle.to(device)
        frame_out_middle_middle = frame_out_middle_middle.to(device)
        frame_out_down_middle = frame_out_down_middle.to(device)
        frame_out_up_right = frame_out_up_right.to(device)
        frame_out_middle_right = frame_out_middle_right.to(device)
        frame_out_down_right = frame_out_down_right.to(device)

      
      #### CH code
      if frame_out_up_left.shape[2]==1200:
         ## Note currently very inefficient, model processes image 9 times for no reason
         frame_out = frame_out_up_left

      if frame_out_up_left.shape[2]==1600:
        frame_out_up_left_middle=(frame_out_up_left[:,:,:,1800:2432]+frame_out_up_middle[:,:,:,0:632])/2
        frame_out_up_middle_right=(frame_out_up_middle[:,:,:,1768:2432]+frame_out_up_right[:,:,:,0:664])/2

        frame_out_middle_left_middle=(frame_out_middle_left[:,:,:,1800:2432]+frame_out_middle_middle[:,:,:,0:632])/2
        frame_out_middle_middle_right=(frame_out_middle_middle[:,:,:,1768:2432]+frame_out_middle_right[:,:,:,0:664])/2

        frame_out_down_left_middle=(frame_out_down_left[:,:,:,1800:2432]+frame_out_down_middle[:,:,:,0:632])/2
        frame_out_down_middle_right=(frame_out_down_middle[:,:,:,1768:2432]+frame_out_down_right[:,:,:,0:664])/2

              

        frame_out_left_up_middle=(frame_out_up_left[:,:,1200:1600,0:1800]+frame_out_middle_left[:,:,0:400,0:1800])/2
        frame_out_left_middle_down=(frame_out_middle_left[:,:,1200:1600,0:1800]+frame_out_down_left[:,:,0:400,0:1800])/2

        frame_out_left = (torch.cat([frame_out_up_left[:, :, 0:1200, 0:1800].permute(0, 2, 3, 1),frame_out_left_up_middle.permute(0, 2, 3, 1), frame_out_middle_left[:, :, 400:1200, 0:1800].permute(0, 2, 3, 1), frame_out_left_middle_down.permute(0, 2, 3, 1), frame_out_down_left[:, :, 400:, 0:1800].permute(0, 2, 3, 1)],1))
              
              
        frame_out_leftmiddle_up_middle=(frame_out_up_left_middle[:,:,1200:1600,:]+frame_out_middle_left_middle[:,:,0:400,:])/2
        frame_out_leftmiddle_middle_down=(frame_out_middle_left_middle[:,:,1200:1600,:]+frame_out_down_left_middle[:,:,0:400,:])/2


        frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1200, :].permute(0, 2, 3, 1),frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1), frame_out_middle_left_middle[:, :, 400:1200, :].permute(0, 2, 3, 1), frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1), frame_out_down_left_middle[:, :, 400:, :].permute(0, 2, 3, 1)],1))
      
        frame_out_middle_up_middle=(frame_out_up_middle[:,:,1200:1600,632:1768]+frame_out_middle_middle[:,:,0:400,632:1768])/2
        frame_out_middle_middle_down=(frame_out_middle_middle[:,:,1200:1600,632:1768]+frame_out_down_middle[:,:,0:400,632:1768])/2
              
        frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1200, 632:1768].permute(0, 2, 3, 1),frame_out_middle_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle[:, :, 400:1200, 632:1768].permute(0, 2, 3, 1), frame_out_middle_middle_down.permute(0, 2, 3, 1), frame_out_down_middle[:, :, 400:, 632:1768].permute(0, 2, 3, 1)],1))
              
        frame_out_middleright_up_middle=(frame_out_up_middle_right[:,:,1200:1600,:]+frame_out_middle_middle_right[:,:,0:400,:])/2
        frame_out_middleright_middle_down=(frame_out_middle_middle_right[:,:,1200:1600,:]+frame_out_down_middle_right[:,:,0:400,:])/2

        frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1200, :].permute(0, 2, 3, 1),frame_out_middleright_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle_right[:, :, 400:1200, :].permute(0, 2, 3, 1), frame_out_middleright_middle_down.permute(0, 2, 3, 1), frame_out_down_middle_right[:, :, 400:, :].permute(0, 2, 3, 1)],1))

        frame_out_right_up_middle=(frame_out_up_right[:,:,1200:1600,664:]+frame_out_middle_right[:,:,0:400,664:])/2
        frame_out_right_middle_down=(frame_out_middle_right[:,:,1200:1600,664:]+frame_out_down_right[:,:,0:400,664:])/2

        frame_out_right = (torch.cat([frame_out_up_right[:, :, 0:1200, 664:].permute(0, 2, 3, 1),frame_out_right_up_middle.permute(0, 2, 3, 1), frame_out_middle_right[:, :, 400:1200, 664:].permute(0, 2, 3, 1), frame_out_right_middle_down.permute(0, 2, 3, 1), frame_out_down_right[:, :, 400:, 664:].permute(0, 2, 3, 1)],1))

        frame_out=torch.cat([frame_out_left, frame_out_leftmiddle, frame_out_middle, frame_out_middleright, frame_out_right],2).permute(0, 3, 1, 2)

              
              
      if frame_out_up_left.shape[2]==2432:
        frame_out_up_left_middle=(frame_out_up_left[:,:,:,1200:1600]+frame_out_up_middle[:,:,:,0:400])/2
        frame_out_up_middle_right=(frame_out_up_middle[:,:,:,1200:1600]+frame_out_up_right[:,:,:,0:400])/2

        frame_out_middle_left_middle=(frame_out_middle_left[:,:,:,1200:1600]+frame_out_middle_middle[:,:,:,0:400])/2
        frame_out_middle_middle_right=(frame_out_middle_middle[:,:,:,1200:1600]+frame_out_middle_right[:,:,:,0:400])/2

        frame_out_down_left_middle=(frame_out_down_left[:,:,:,1200:1600]+frame_out_down_middle[:,:,:,0:400])/2
        frame_out_down_middle_right=(frame_out_down_middle[:,:,:,1200:1600]+frame_out_down_right[:,:,:,0:400])/2

              

        frame_out_left_up_middle=(frame_out_up_left[:,:,1800:2432,0:1200]+frame_out_middle_left[:,:,0:632,0:1200])/2
        frame_out_left_middle_down=(frame_out_middle_left[:,:,1768:2432,0:1200]+frame_out_down_left[:,:,0:664,0:1200])/2

        frame_out_left = (torch.cat([frame_out_up_left[:, :, 0:1800, 0:1200].permute(0, 2, 3, 1),frame_out_left_up_middle.permute(0, 2, 3, 1), frame_out_middle_left[:, :, 632:1768, 0:1200].permute(0, 2, 3, 1), frame_out_left_middle_down.permute(0, 2, 3, 1), frame_out_down_left[:, :, 664:, 0:1200].permute(0, 2, 3, 1)],1))
              
              
        frame_out_leftmiddle_up_middle=(frame_out_up_left_middle[:,:,1800:2432,:]+frame_out_middle_left_middle[:,:,0:632,:])/2
        frame_out_leftmiddle_middle_down=(frame_out_middle_left_middle[:,:,1768:2432,:]+frame_out_down_left_middle[:,:,0:664,:])/2


        frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1800, :].permute(0, 2, 3, 1),frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1), frame_out_middle_left_middle[:, :, 632:1768, :].permute(0, 2, 3, 1), frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1), frame_out_down_left_middle[:, :, 664:, :].permute(0, 2, 3, 1)],1))
              
              
        frame_out_middle_up_middle=(frame_out_up_middle[:,:,1800:2432,400:1200]+frame_out_middle_middle[:,:,0:632,400:1200])/2
        frame_out_middle_middle_down=(frame_out_middle_middle[:,:,1768:2432,400:1200]+frame_out_down_middle[:,:,0:664,400:1200])/2
              
        frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1800, 400:1200].permute(0, 2, 3, 1),frame_out_middle_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle[:, :, 632:1768, 400:1200].permute(0, 2, 3, 1), frame_out_middle_middle_down.permute(0, 2, 3, 1), frame_out_down_middle[:, :, 664:, 400:1200].permute(0, 2, 3, 1)],1))
              
              
              
        frame_out_middleright_up_middle=(frame_out_up_middle_right[:,:,1800:2432,:]+frame_out_middle_middle_right[:,:,0:632,:])/2
        frame_out_middleright_middle_down=(frame_out_middle_middle_right[:,:,1768:2432,:]+frame_out_down_middle_right[:,:,0:664,:])/2

        frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1800, :].permute(0, 2, 3, 1),frame_out_middleright_up_middle.permute(0, 2, 3, 1), frame_out_middle_middle_right[:, :, 632:1768, :].permute(0, 2, 3, 1), frame_out_middleright_middle_down.permute(0, 2, 3, 1), frame_out_down_middle_right[:, :, 664:, :].permute(0, 2, 3, 1)],1))

              


        frame_out_right_up_middle=(frame_out_up_right[:,:,1800:2432,400:]+frame_out_middle_right[:,:,0:632,400:])/2
        frame_out_right_middle_down=(frame_out_middle_right[:,:,1768:2432,400:]+frame_out_down_right[:,:,0:664,400:])/2

        frame_out_right = (torch.cat([frame_out_up_right[:, :, 0:1800, 400:].permute(0, 2, 3, 1),frame_out_right_up_middle.permute(0, 2, 3, 1), frame_out_middle_right[:, :, 632:1768, 400:].permute(0, 2, 3, 1), frame_out_right_middle_down.permute(0, 2, 3, 1), frame_out_down_right[:, :, 664:, 400:].permute(0, 2, 3, 1)],1))

            
            
        frame_out=torch.cat([frame_out_left, frame_out_leftmiddle, frame_out_middle, frame_out_middleright, frame_out_right],2).permute(0, 3, 1, 2)
      # frame_out=torch.cat([frame_out_left, frame_out_leftmiddle, frame_out_middle, frame_out_middleright, frame_out_right],2).permute(0, 3, 1, 2)
      frame_out=frame_out.to(device)
          
      # fourth_channel=torch.ones([frame_out.shape[0],1,frame_out.shape[2],frame_out.shape[3]],device='cuda:1')
      fourth_channel = torch.ones([frame_out.shape[0],1,frame_out.shape[2],frame_out.shape[3]], device=device) ########## ---- CH code ---- ##########

      frame_out_rgba=torch.cat([frame_out,fourth_channel],1)
      # print(frame_out_rgba.shape)
            
      # name= re.findall("\d+",str(name))
      # imwrite(frame_out_rgba, output_dir + '/' + str(name[0])+'.png', range=(0, 1))

      ########## ---- Start of CH code: Output meaningful filenames ---- ##########
      imwrite(frame_out_rgba, os.path.join(output_dir, str(name[0])), range=(0, 1))
    
  test_time = time.time() - start_time
  print(f"Time taken: {test_time}")
      ########## ---- End of CH code ---- ##########