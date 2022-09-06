from networks import *
select_classes = ['background', 'hair']

class DeepLabPredict:
    def __init__(self, pretrained='checkpoints/lastest_model.pth'):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights='imagenet',
                                   classes=2, activation='sigmoid')
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101')
        checkpoint = torch.load(pretrained, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.DEVICE)
        self.model.eval()

    def get_coord_original(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        y1 = min(np.where(threshold == 255)[0])
        y2 = max(np.where(threshold == 255)[0])
        x1 = min(np.where(threshold == 255)[1])
        x2 = max(np.where(threshold == 255)[1])
        return x1, y1, x2, y2

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image path")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def predict(self, image):
        image = self.check_type(image)
        image_original = image.copy()
        h, w = image.shape[:2]

        # Padding original image to 1536*1536 (size input must divisible by 32)
        image = get_validation_augmentation()(image=image)['image']

        x1, y1, x2, y2 = self.get_coord_original(image)

        # Processing input: convert tensor, normalize...
        sample = get_preprocessing(self.preprocessing_fn)(image=image)
        image = sample['image']
        x_tensor = torch.from_numpy(image).to(self.DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = self.model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_building_heatmap = pred_mask[:, :, select_classes.index('hair')]
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), [[0, 0, 0], [255, 255, 255]])
        pred_mask = pred_mask.astype('uint8')
        final = pred_mask[y1:y2, x1:x2]
        final = cv2.resize(final, (w, h))
        return final

if __name__ == '__main__':
    Deeplab = DeepLabPredict()
    mask = Deeplab.predict('7.jpg')
    cv2.imshow('t', mask)
    cv2.waitKey(0)