from .libs import *

class HairDataset(Dataset):
    color_encoding = [
        ('hair', (255, 255, 255)),
    ]
    def __init__(self, path_dataset, mode='train', num_classes=2, device='cpu', img_size=(480, 360)):
        self.mode = mode
        self.device = device
        self.path_dataset = path_dataset
        self.num_classes = num_classes
        self.img_size = img_size
        # Normailization
        self.normalize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

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

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path)
        label = cv2.imread(label_path)

        return data, label

    def label_for_cross_entropy(self, label):
        semantic_map = np.zeros(label.shape)
        # Fill all value with class 13 - default for all pixels
        semantic_map.fill(self.num_classes - 1)
        # Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = color_info[1]
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
        return semantic_map

    def __getitem__(self, index):
        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)

        # Apply normalization in img
        img = self.normalize(img)
        label = cv2.resize(label, self.img_size[::-1])
        # Convert label for cross entropy
        label = self.label_for_cross_entropy(label)[:,:,0]
        label = torch.from_numpy(label).long()

        return img, label

    def __len__(self):
        num = [name for name in os.listdir(os.path.join(self.DATA_PATH, self.mode, 'images')) if name.endswith('jpg')]
        return len(num)

if __name__ == '__main__':
    # TEST
    from bisenet import BiSeNet
    model = BiSeNet(num_classes=2, training=True)
    dataset_val = FigaroDataset('Figaro_1k/', num_classes=2, mode='val', device='cpu')
    dataloader_val = DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=True
    )
    _, mean_iou = val(model, dataloader_val, 2)