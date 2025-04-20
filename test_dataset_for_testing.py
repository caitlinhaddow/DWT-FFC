from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test_hazy=[]

        # self.root_hazy = os.path.join(test_dir, 'hazy/')
        self.root_hazy = test_dir  ## CH add

        for i in os.listdir(self.root_hazy):
           self.list_test_hazy.append(i)
        #self.root_hazy = os.path.join(test_dir)


        self.file_len = len(self.list_test_hazy)


    def __getitem__(self, index, is_train=True):
        hazy = Image.open(os.path.join(self.root_hazy, self.list_test_hazy[index])).convert('RGB')
        hazy = self.transform(hazy)
        # print(f"------ shape 1: {hazy.shape[1]}, shape 2: {hazy.shape[2]}")

        # If landscape
        if hazy.shape[1] < hazy.shape[2]:
            # print(f"landscape: {hazy.shape}")

            if hazy.shape[1] == 4000 and hazy.shape[2] == 6000:
                # Original sizings
                hazy_up_left=hazy[:,0:1600, 0:2432]
                hazy_up_middle=hazy[:, 0:1600, 1800:4232]
                hazy_up_right=hazy[:,0:1600, 3568:6000]

                hazy_middle_left=hazy[:,1200:2800, 0:2432]
                hazy_middle_middle=hazy[:, 1200:2800, 1800:4232]
                hazy_middle_right=hazy[:,1200:2800, 3568:6000]

                hazy_down_left=hazy[:,2400:4000, 0:2432]
                hazy_down_middle=hazy[:, 2400:4000, 1800:4232]
                hazy_down_right=hazy[:,2400:4000, 3568:6000]

                name = self.list_test_hazy[index]

            elif hazy.shape[1] == 1200 and hazy.shape[2] == 1600:
                ## ---------- CH code: Do not segment image for 1200x1600 -----------
                hazy_up_left = hazy[:,:,:]
                name = self.list_test_hazy[index]

                return hazy_up_left, name

            else:
                print("Unexpected image size")

        # If portrait
        if hazy.shape[1] > hazy.shape[2]:
            # print("portrait")
            if hazy.shape[1] == 6000 and hazy.shape[2] == 4000:
                hazy_up_left=hazy[:,0:2432, 0:1600]
                hazy_up_middle=hazy[:, 0:2432, 1200:2800]
                hazy_up_right=hazy[:,0:2432, 2400:]

                hazy_middle_left=hazy[:,1800:4232, 0:1600]
                hazy_middle_middle=hazy[:, 1800:4232, 1200:2800]
                hazy_middle_right=hazy[:,1800:4232, 2400:]

                hazy_down_left=hazy[:,3568:6000, 0:1600]
                hazy_down_middle=hazy[:, 3568:6000, 1200:2800]
                hazy_down_right=hazy[:,3568:6000, 2400:]

                name=self.list_test_hazy[index]
            else:
                print("Unexpected image size")
        
        return hazy_up_left, hazy_up_middle, hazy_up_right, hazy_middle_left, hazy_middle_middle, hazy_middle_right, hazy_down_left, hazy_down_middle, hazy_down_right, name
    def __len__(self):
        return self.file_len

