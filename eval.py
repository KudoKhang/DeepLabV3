from networks import *
args = get_args()

# Config hyperparameter
LEARNING_RATE = args.lr
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
ENCODER_NAME = 'resnet50'
print(f"Device using: {DEVICE}")

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER_NAME,
    encoder_weights='imagenet',
    classes=NUM_CLASSES,
    activation='sigmoid',
)

model.to(DEVICE)

# Load pretrained if exist
PRETRAINED = 'checkpoints/lastest_model.pth'
checkpoint = torch.load(PRETRAINED, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch']
iou = checkpoint['iou']
f1score = checkpoint['f1-score']
print('Evaluation by weights ---{}--- have IoU = {}, F1Score = {}, trained with epoch: {} \n'.format(PRETRAINED, iou, f1score, start_epoch))

preprocessing_fn = get_preprocessing_fn(ENCODER_NAME)

val_dataset = BuildingsDataset(path_dataset=ROOT, mode='val',
                                 augmentation=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

valid_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True
    )

# define loss function
loss = smp.utils.losses.DiceLoss()
# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore()
]

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# Loop for training
torch.cuda.empty_cache()
def training():
    valid_logs_list = []
    valid_logs = valid_epoch.run(valid_dataloader)
    valid_logs_list.append(valid_logs)
    print(valid_logs_list)

# use "if __name__ == '__main__' to fix error Parallel"
# https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
if __name__ == '__main__':
    training()
