from networks import *
args = get_args()

# Config hyperparameter
select_classes = ['background', 'hair']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
model = smp.DeepLabV3Plus(
    encoder_name='resnet101',
    encoder_weights='imagenet',
    classes=NUM_CLASSES,
    activation='sigmoid',
)
preprocessing_fn = get_preprocessing_fn('resnet101')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(os.path.join('checkpoints', 'lastest_model.pth'), map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch']
miou = checkpoint['iou']

model.to(DEVICE)
model.eval()

test_dataset = BuildingsDataset(path_dataset=ROOT, mode='test',
                                 augmentation=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

test_dataloader = DataLoader(test_dataset)

# test dataset for visualization (without preprocessing transformations)
test_dataset_vis = BuildingsDataset(path_dataset=ROOT, mode='test',
                                 augmentation=get_validation_augmentation())

# id = 1
# image, gt_mask = test_dataset[id]
# image_vis = crop_image(test_dataset_vis[id][0].astype('uint8'))
# x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
# # Predict test image
# pred_mask = model(x_tensor)
# pred_mask = pred_mask.detach().squeeze().cpu().numpy()
# # Convert pred_mask from `CHW` format to `HWC` format
# pred_mask = np.transpose(pred_mask,(1,2,0))
# # Get prediction channel corresponding to building
# pred_building_heatmap = pred_mask[:,:,select_classes.index('hair')]
# pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), [[0,0,0], [255,255,255]])
# pred_mask = crop_image(pred_mask)
# # Convert gt_mask from `CHW` format to `HWC` format
# gt_mask = np.transpose(gt_mask,(1,2,0))
# gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), [[0,0,0], [255,255,255]]))
# cv2.imwrite(os.path.join(f"sample_pred_{id}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])

# Inference with image
image = cv2.imread('dataset/Figaro_1k/test/images/187.jpg')
image = get_validation_augmentation()(image=image)['image']
sample = get_preprocessing(preprocessing_fn)(image=image)
image = sample['image']
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
# Predict test image
pred_mask = model(x_tensor)
pred_mask = pred_mask.detach().squeeze().cpu().numpy()
# Convert pred_mask from `CHW` format to `HWC` format
pred_mask = np.transpose(pred_mask,(1,2,0))
pred_building_heatmap = pred_mask[:,:,select_classes.index('hair')]
pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), [[0,0,0], [255,255,255]])
# pred_mask = crop_image(pred_mask)

print('pause')