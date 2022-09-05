from .libs import *
from .utilss import *

class BuildingsDataset(torch.utils.data.Dataset):
    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, path_dataset, mode, class_rgb_values=[[0, 0, 0], [255, 255, 255]], augmentation=None, preprocessing=None):
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode

        self.path_dataset = path_dataset

        self.DATA_PATH = os.path.join(os.getcwd(), self.path_dataset)
        self.train_path, self.val_path, self.test_path = [os.path.join(self.DATA_PATH, x) for x in
                                                          ['train', 'val', 'test']]

        if self.mode == 'train':
            self.data_files = self.get_files(self.train_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        elif self.mode == 'val':
            self.data_files = self.get_files(self.val_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        elif self.mode == 'test':
            self.data_files = self.get_files(self.test_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def get_files(self, data_folder):
        return glob("{}/*.{}".format(os.path.join(data_folder, 'images'), 'jpg'))

    def get_label_file(self, data_path, data_dir, label_dir):
        data_path = data_path.replace(data_dir, label_dir)
        fname, _ = data_path.split('.')
        return "{}.{}".format(fname, 'png')

    def check_size(self, data, label):
        h, w = data.shape[:2]
        if h < 256 or w < 256:
            data = cv2.resize(data, (w*2, h*2))
            label = cv2.resize(label, (w*2, h*2))

        return data, label

    def image_loader(self, data_path, label_path):
        data = cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        data, label = self.check_size(data, label)
        return data, label

    def __getitem__(self, index):
        data_path, label_path = self.data_files[index], self.label_files[index]
        # read images and masks
        image, mask = self.image_loader(data_path, label_path)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        num = [name for name in os.listdir(os.path.join(self.DATA_PATH, self.mode, 'images')) if name.endswith('jpg')]
        return len(num)