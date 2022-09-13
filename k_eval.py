from inference import DeepLabPredict
from networks import *

class Evaluation:
    def __init__(self, pretrained='checkpoints/lastest_model_timm-mobilenetv3_small_100.pth',ENCODER_NAME='timm-mobilenetv3_small_100', ROOT='dataset/Figaro_1k/', NUM_CLASSES=2):
        self.ROOT = ROOT
        self.NUM_CLASSES = NUM_CLASSES
        self.Deeplab = DeepLabPredict(pretrained=pretrained, ENCODER_NAME=ENCODER_NAME, is_draw_bbox=False)
        self.path_images = os.path.join(ROOT, 'test/images')
        self.path_masks = os.path.join(ROOT, 'test/masks')
        self.lst_images = [name for name in os.listdir(self.path_images) if name.endswith('jpg')]

        self.convert_tensor = transforms.ToTensor()
        self.hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        self.f1_total = []
        self.accuracy_total = []

    def __call__(self):
        with alive_bar(total=len(self.lst_images), theme='musical', length=100) as bar:
            for name in self.lst_images:
                pred = self.Deeplab.predict(os.path.join(self.path_images, name), visualize=False)
                pred = Image.fromarray(pred).convert('L')
                pred = self.convert_tensor(pred).squeeze()

                GT = Image.open(os.path.join(self.path_masks, name[:-3] + 'png')).convert('L')
                GT = self.convert_tensor(GT).squeeze()

                f1 = f1_loss(pred, GT)
                self.f1_total.append(np.array(f1.detach().numpy()))

                accuracy = compute_accuracy(pred, GT)
                self.accuracy_total.append(accuracy)

                pred = np.array(pred.cpu()).astype('int64')
                GT = np.array(GT.cpu()).astype('int64')

                self.hist += fast_hist(pred.flatten(), GT.flatten(), self.NUM_CLASSES)
                bar()

        miou_list = per_class_iu(self.hist)[:-1]
        mean_accuracy, mean_iou, f1_score = np.mean(self.accuracy_total), np.mean(miou_list), np.mean(self.f1_total)
        print(f"Accuracy: {mean_accuracy} --- mIoU: {mean_iou} --- F1_Score: {f1_score}")
        return mean_accuracy, mean_iou, f1

if "__main__" == __name__:
    eval = Evaluation()
    Accucracy, IoU, F1_Score = eval()


