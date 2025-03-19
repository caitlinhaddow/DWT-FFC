import torch
import time
import argparse
import datetime

######### ch added
# from model import fusion_net, Discriminator
from model_convnext import fusion_net, Discriminator

from train_dataset import dehaze_train_dataset  ## added file from dwgan (originally from ITB-Dehaze)

from test_val_dataset import dehaze_test_dataset
# from test_dataset_for_testing import dehaze_test_dataset  ## if doesn't work, try adding from DW-GAN

###########################

from torch.utils.data import DataLoader
import os
from utils_test import to_psnr, to_ssim_skimage ##, to_rmse
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from perceptual import LossNetwork

######### ch added
save_prefix = "NHNH2RB10" ## SET VARIABLE WHEN TRAINING DATASET CHANGES
script_start_time = datetime.datetime.now()
print(f"--- Start time: {script_start_time.strftime('%Y-%m-%d_%H-%M-%S')} ---")
torch.cuda.empty_cache()
###########################

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='DWT-FFC Dehaze')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)  ## originally 25, paper used 16
parser.add_argument('-train_epoch', help='Set the training epoch', default=10005, type=int) ## paper used 10000 epochs
parser.add_argument('--train_dataset', type=str, default='./training_data/')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./check_points')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='./test_data/')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
# train_dataset = os.path.join('/your/path/')
train_dataset = args.train_dataset
# --- test --- #
# test_dataset = os.path.join('/your/path/')
test_dataset = args.test_dataset
predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join('check_points/')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = fusion_net()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()

########### CH New code

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids).to(device)
    DNet = torch.nn.DataParallel(DNet, device_ids=device_ids).to(device)
else:
    print(f"Using {torch.cuda.device_count()} GPUs")
    MyEnsembleNet = MyEnsembleNet.to(device)
    DNet = DNet.to(device)

###################

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)

# --- Load training data --- #
dataset = dehaze_train_dataset(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, num_workers=10, pin_memory=True)  # CH change: pin memory to preload directly to GPU

# --- Load testing data --- #
test_dataset = dehaze_test_dataset(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, num_workers=10, pin_memory=True) # CH change: pin memory to preload directly to GPU
MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)
writer = SummaryWriter()

# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(os.path.join(args.teacher_model, 'best.pkl')))  ## CHECK WHETHER I SHOULD BE CHANGING THIS
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16].to(device)  ## CH change Moved VGG to device earlier
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model).to(device)  ## CH change: moved to same device as VGG
loss_network.eval()
msssim_loss = msssim

# --- Strat training --- #
print("strat training")
iteration = 0
best_psnr = 0
for epoch in range(train_epoch):
    start_time = time.time()
    scheduler_G.step()
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    
    for batch_idx, (hazy, clean) in enumerate(train_loader):
        iteration += 1
        hazy = hazy.to(device) ## CH change: pin memory to preload directly to GPU
        clean = clean.to(device) ## CH change: pin memory to preload directly to GPU
        output = MyEnsembleNet(hazy)

        torch.cuda.empty_cache() ## CH change: free up memory after each batch

        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        D_loss.backward(retain_graph=True)
        adversarial_loss = torch.mean(1 - fake_out)
        MyEnsembleNet.zero_grad()
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.2 * msssim_loss_
        total_loss.backward()
        D_optim.step()
        G_optimizer.step()
        writer.add_scalars('training', {'training total loss': total_loss.item()}, iteration)
        writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                            'perceptual': perceptual_loss.item(),
                                            'msssim': msssim_loss_.item()}, iteration)
        writer.add_scalars('GAN_training', {'d_loss': D_loss.item(),'d_score': real_out.item(),
            'g_score': fake_out.item()}, iteration)
    if (epoch % 5 == 0):
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            rmse_list = []
            MyEnsembleNet.eval()
            for batch_idx, (hazy_up, hazy_down, hazy, clean) in enumerate(test_loader):
                hazy_up = hazy_up.to(device)
                hazy_down=hazy_down.to(device)
                clean = clean.to(device)
                frame_out_up = MyEnsembleNet(hazy_up)
                frame_out_down = MyEnsembleNet(hazy_down)
                # frame_out=(torch.cat([frame_out_up.permute(0,2,3,1), frame_out_down[:,:,80:640,:].permute(0,2,3,1)], 1)).permute(0,3,1,2)
                frame_out = torch.cat([frame_out_up, frame_out_down[:, :, 80:640, :]], dim=2)  ## CH change: reduce number of permute calls

                if not os.path.exists('output/'):
                    os.makedirs('output/')
                imwrite(frame_out, 'output/' + str(batch_idx) + '.png', range=(0, 1))

                ## CH add ############
                new_psnr, new_rmse = to_psnr(frame_out, clean)
                psnr_list.extend(new_psnr)
                rmse_list.extend(new_rmse)
                ###############

                # psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))
                # rmse_list.extend(to_rmse(frame_out, clean))
            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            avr_rmse = sum(rmse_list) / len(rmse_list)
            writer.add_scalars('testing', {'testing psnr': avr_psnr, 'testing rmse': avr_rmse,
                                           'testing ssim': avr_ssim}, epoch)
            if avr_psnr > best_psnr:
                print(f"New best PSNR achieved: {avr_psnr}. Saving model...")
                torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir, f"{script_start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{save_prefix}_epoch{epoch:05d}.pkl"))
                best_psnr = avr_psnr
            if epoch % 1000 == 0:
                print(f"{epoch} checkpoint. PSNR: {avr_psnr}. Saving model...")
                torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir, f"{script_start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{save_prefix}_epoch{epoch:05d}.pkl"))
end_time = datetime.datetime.now()
print(f"Start time{script_start_time.strftime('%Y-%m-%d_%H-%M-%S')}, End time {end_time.strftime('%Y-%m-%d_%H-%M-%S')}, Total duration: {end_time - script_start_time}")