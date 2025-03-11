# Create txt file needed for test_val_dataset.py
# CH Added: temp file, needing for training
import os

def write_file(txt_dir, description):
    data_dir = os.path.join(txt_dir, "hazy/")

    write_file = open(os.path.join(txt_dir, f"{description}.txt"), "w")

    for filename in os.listdir(data_dir):
        _, file_extension = os.path.splitext(filename)
        if file_extension in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
            write_file.write(f"{filename}\n")
    write_file.close()
    print(f"{description} txt file done")


dataset_dir = "/home/caitlinhaddow/Documents/Datasets/_NH2/"  ## SET VARIABLE

# Assumes file structure of dataset_dir > training_data AND test_data > hazy AND clean
train_txt_dir = os.path.join(dataset_dir, "training_data/")
test_txt_dir = os.path.join(dataset_dir, "test_data/")

write_file(train_txt_dir, "train")
write_file(test_txt_dir, "test")



