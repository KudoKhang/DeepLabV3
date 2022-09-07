from networks import *

class DeepLabPredict:
    def __init__(self, pretrained='checkpoints/lastest_model.pth', ENCODER_NAME='resnet50'):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3Plus(encoder_name=ENCODER_NAME, encoder_weights='imagenet',
                                   classes=2, activation='sigmoid')
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_NAME)
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

    def visualize(self, img, label, color = (0, 255, 0)):
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        # TODO: add color only hair via mask
        return cv2.addWeighted(img, 0.6, label, 0.4, 0)

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image path")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def predict(self, image, visualize=True):
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
        if visualize:
            image_visualize = self.visualize(image_original, final.copy())
            # finall = np.hstack([image_original, final, image_visualize])
            # cv2.imshow('DeepLabV3 Predict', finall)
            # cv2.waitKey(0)
            return image_visualize
        return final

#------#------#------#------#------#------#------#------#------#------#------#------
def webcam():
    print("Using webcam, press [q] to exit, press [s] to save")
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    with alive_bar(theme='musical', length=200) as bar:
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            start = time.time()
            frame = Deeplab.predict(frame)
            fps = round(1 / (time.time() - start), 2)
            cv2.putText(frame, "FPS : " + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.imshow('Prediction', frame + 30)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                os.makedirs('results/', exist_ok=True)
                cv2.imwrite('results/' + str(time.time()) + '.jpg', frame)
            if k == ord('q'):
                break
            bar()

if __name__ == '__main__':
    Deeplab = DeepLabPredict()
    # image = Deeplab.predict('dataset/Figaro_1k_png/test/images/805.jpg')
    webcam()