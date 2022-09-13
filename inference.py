from networks import *

class DeepLabPredict:
    def __init__(self, pretrained='checkpoints/lastest_model.pth', ENCODER_NAME='resnet50', is_draw_bbox=False):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3Plus(encoder_name=ENCODER_NAME,
                                       encoder_weights='imagenet',
                                       classes=2, activation='sigmoid')
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_NAME)
        checkpoint = torch.load(pretrained, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.DEVICE)
        self.model.eval()

        # Model detect person
        self.is_draw_bbox = is_draw_bbox
        self.model_detect_person = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/yolov5s.pt')

    def get_coord_original(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        y1 = min(np.where(threshold == 255)[0])
        y2 = max(np.where(threshold == 255)[0])
        x1 = min(np.where(threshold == 255)[1])
        x2 = max(np.where(threshold == 255)[1])
        return (x1, y1, x2, y2)

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

    def check_size(self, img):
        h, w = img.shape[:2]
        ratio = h / w
        if h or w > 1536:
            size = (int(1536 / ratio), 1536) if h > w else (1536, int(w * ratio))
            img = cv2.resize(img, size)
        return img

    def draw_bbox(self, img, t):
        bbox = list(t[:, :5][np.where(t[:, 6] == 'person')][np.where(t[:, 4] > 0.4)])
        for bb in bbox:
            x1, y1, x2, y2 = np.uint32(bb[:4])
            confident = bb[4]
            img = cv2.putText(img, 'Confident: ' + str(round(confident * 100, 2)) + '%', (x1 + 2, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    def detect_person(self, image):
        # TODO: FIX CASE NO PERSON
        results = self.model_detect_person(image)
        t = np.array(results.pandas().xyxy[0])
        # bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')][np.where(t[:,4] > 0.7)])) # Get person have condident score > 0.7
        bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')]))
        if self.is_draw_bbox:
            self.draw_bbox(image, t)
        return bbox

    def post_process_mask(self, pred_mask):
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_building_heatmap = pred_mask[:, :, select_classes.index('hair')]
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), [[0, 0, 0], [255, 255, 255]])
        pred_mask = pred_mask.astype('uint8')
        return pred_mask

    def pre_process_input(self, image):
        image = self.check_size(image)
        # Padding original image to 1536*1536 (size input must divisible by 32)
        image = get_validation_augmentation()(image=image)['image']
        _bbox = self.get_coord_original(image)
        # Processing input: convert tensor, normalize...
        sample = get_preprocessing(self.preprocessing_fn)(image=image)
        image = sample['image']
        image = torch.from_numpy(image).to(self.DEVICE).unsqueeze(0)
        return image, _bbox

    def predict(self, image, visualize=True):
        image = self.check_type(image)
        # image_original = image.copy()
        bbox = self.detect_person(image)
        labels = np.zeros_like(image)
        temp_label = labels.copy()
        for bb in bbox:
            x1, y1, x2, y2 = bb
            person = image[y1: y2, x1: x2]
            h, w = person.shape[:2]
            person, _bbox = self.pre_process_input(person)
            pred_mask = self.model(person)
            pred_mask = self.post_process_mask(pred_mask)
            final = pred_mask[_bbox[1]:_bbox[3], _bbox[0]:_bbox[2]]
            final = cv2.resize(final, (w, h))
            temp_label[y1:y2, x1:x2] = final
            labels += temp_label

        _, labels = cv2.threshold(labels, 20, 255, cv2.THRESH_BINARY)

        if visualize:
            image_visualize = self.visualize(image, labels.copy())
            # finall = np.hstack([image_original, final, image_visualize])
            # cv2.imshow('DeepLabV3 Predict', finall)
            # cv2.waitKey(0)
            return image_visualize
        return labels

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

def time_inference(root):
    path_name = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('jpg')]
    start = time.time()
    for path in tqdm(path_name):
        Deeplab.predict(path)
    end = (time.time() - start) / len(path_name)
    print(f'Avg time inference {len(path_name)} images is:', round(end * 1e3), 'ms')

def image(path='dataset/Figaro_1k/test/images/45.jpg'):
    start = time.time()
    img = Deeplab.predict(path)
    print('Time inference: ', round((time.time() - start) * 1e3, 2), 'ms')
    cv2.imshow('DeepLabv3 Predict', img)
    cv2.waitKey(0)

def video(path_video='tests/hair1.mp4', name='result_'):
    print(f'Processing video ---{os.path.basename(path_video)}--- \nPlease Uong mieng nuoc & an mieng banh de...')
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 30
    os.makedirs('results/', exist_ok=True)
    out = cv2.VideoWriter(f'results/{name}' + path_video.split('/')[-1], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    with alive_bar(theme='musical', length=200) as bar:
        while True:
            _, frame = cap.read()
            try:
                frame = Deeplab.predict(frame)
                out.write(frame)
                bar()
            except:
                out.release()
                break
    out.release()

def process_folder(root):
    list_path = [name for name in os.listdir(root) if name.endswith(('jpg', 'png', 'jpeg'))]
    saved_folder = os.path.join('results', os.path.basename(root))
    os.makedirs(saved_folder, exist_ok=True)
    with alive_bar(total=len(list_path), theme='musical', length=100) as bar:
        for path in list_path:
            try:
                image = Deeplab.predict(os.path.join(root, path))
                cv2.imwrite(os.path.join(saved_folder, path), image)
                bar()
            except:
                print(path)
                continue

if __name__ == '__main__':
    Deeplab = DeepLabPredict(pretrained='checkpoints/lastest_model_r101_figaro.pth', ENCODER_NAME='resnet101', is_draw_bbox=False)
    # time_inference('dataset/Figaro_1k/test/images')
    image('dataset/Figaro_1k/test/images/742.jpg')
    # webcam()
    # video('test/k.mp4')
    # process_folder('dataset/Figaro_1k/test/images/')

