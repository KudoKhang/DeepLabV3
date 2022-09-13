import segmentation_models_pytorch.utils.metrics
from k_eval import *
from networks import *
args = get_args()

# Config hyperparameter
EPOCHS = args.epoch
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
ENCODER_NAME = args.backbone
VALIDATION_STEP = args.validation_step
print(f"Device using: {DEVICE}")
start_epoch = 0
max_f1score = 0

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
    f1score = checkpoint['f1score']
    print('Resume training from ---{}--- have mIoU = {} F1Score = {}, start at epoch: {} \n'.format(args.pretrained, miou, f1score, start_epoch))

preprocessing_fn = get_preprocessing_fn(ENCODER_NAME)

# Dataloader for train
train_dataset = BuildingsDataset(path_dataset=ROOT, mode='train',
                                 augmentation=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))


train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True,
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

# Loop for training
torch.cuda.empty_cache()
def training():
    wandb.init(project='Hair_segmentation_DeepLab1', entity='khanghn')
    train_logs_list = []

    for epoch in range(start_epoch, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_dataloader)
        train_logs_list.append(train_logs)

        if not os.path.exists(args.pretrained):
            os.makedirs(args.pretrained, exist_ok=True)

        # Save lastest weights
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'iou': train_logs['iou_score'],
            'f1score': train_logs['f1score']
        }
        torch.save(states, f'{args.pretrained}/lastest_model.pth')

        # Validation
        if (epoch + 1) % VALIDATION_STEP == 0:
            eval = Evaluation(pretrained=f'{args.pretrained}/lastest_model.pth', ENCODER_NAME=ENCODER_NAME, ROOT=ROOT)
            Accucracy, IoU, F1_Score = eval()

            # Save model if a better val IoU score is obtained
            if max_f1score < F1_Score:
                max_f1score = F1_Score
                print('---Save best model with F1Score = {}--- \n'.format(max_f1score))
                torch.save(states, f'{args.pretrained}/best_model.pth')
            else:
                print('---Save Failed---')

        wandb.log({"train_IoU": train_logs['iou_score'],
                    "train_FScore": train_logs['fscore'],
                    "train_logs": train_logs['dice_loss'],
                    "Valid_Accuracy": Accucracy,
                    "Valida_IoU": IoU,
                    "Valida_F1Score": F1_Score
                   })

# use "if __name__ == '__main__' to fix error Parallel"
# https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
if __name__ == '__main__':
    training()
