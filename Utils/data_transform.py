from torchvision.transforms import transforms
seg_transform = transforms.Compose(
    [
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor()
    ]
)

aug_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 10)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# imageNet的标准化[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
lung_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),
    "original": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.05, 0.05, 0.05], [0.9, 0.9, 0.9])
                               ])}
transform_none = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor()])