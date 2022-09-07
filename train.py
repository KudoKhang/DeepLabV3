import segmentation_models_pytorch.utils.metrics

from networks import *
args = get_args()

# Config hyperparameter
EPOCHS = args.epoch
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
start_epoch = 0
max_miou = 0
ENCODER_NAME = 'resnet101'
print(f"Device using: {DEVICE}")

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER_NAME,
    encoder_weights='imagenet',
    classes=NUM_CLASSES,
    activation='sigmoid',
)

model.to(DEVICE)

# Load pretrained if exist
if os.path.exists(os.path.join(args.pretrained, 'lastest_model.pth')):
    checkpoint = torch.load(os.path.join(args.pretrained, 'lastest_model.pth'), map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    miou = checkpoint['iou']
    print('Resume training from ---{}--- have mIoU = {}, start at epoch: {} \n'.format(args.pretrained, miou, start_epoch))

preprocessing_fn = get_preprocessing_fn(ENCODER_NAME)

# Dataloader for train
train_dataset = BuildingsDataset(path_dataset=ROOT, mode='train',
                                 augmentation=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

val_dataset = BuildingsDataset(path_dataset=ROOT, mode='val',
                                 augmentation=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

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
# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

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
    wandb.init(project='Hair_segmentation_DeepLab1', entity='khanghn')
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for epoch in range(0, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if not os.path.exists(args.pretrained):
            os.makedirs(args.pretrained, exist_ok=True)

        # Save lastest weights
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'iou': valid_logs['iou_score']
        }
        torch.save(states, f'{args.pretrained}/lastest_model.pth')

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            print('---Save best model with mIoU = {}--- \n'.format(best_iou_score))
            torch.save(states, f'{args.pretrained}/best_model.pth')
        else:
            print('---Save Failed---')

        wandb.log({"valid_IoU: ": valid_logs['iou_score'],
                    "train_IoU: ": train_logs['iou_score'],
                    "valid_Loss: ": valid_logs['dice_loss'],
                    "train_logs: ": train_logs['dice_loss']})

# use "if __name__ == '__main__' to fix error Parallel"
# https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
if __name__ == '__main__':
    training()
