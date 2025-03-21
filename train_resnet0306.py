
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as F
from torchvision import transforms
import h5py
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入余弦退火调度器

# 自定义数据集类，patch_size设为224,方便与与训练resnet对接
class SOS(torch.utils.data.Dataset):
    def __init__(self, images_artifact, images_perfect, patch_size=224, transform=None):  
        assert images_artifact.shape[0] == images_perfect.shape[0], \
            "缺陷数据集和完美数据集包含图片数量必须相等"
        self.images_artifact = images_artifact
        self.images_perfect = images_perfect
        self.transform = transform
        self.patch_size = patch_size  # Patch尺寸

    def __len__(self):
        return self.images_artifact.shape[0]

    def __getitem__(self, idx):
        artifact_image = self.images_artifact[idx]
        perfect_image = self.images_perfect[idx]

        # 随机裁剪1个patch
        i, j, h, w = transforms.RandomCrop.get_params(artifact_image, output_size=(self.patch_size, self.patch_size))
        artifact_patch = F.crop(artifact_image, i, j, h, w)
        perfect_patch = F.crop(perfect_image, i, j, h, w)

        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            artifact_patch = F.hflip(artifact_patch)
            perfect_patch = F.hflip(perfect_patch)

        # 随机幅值缩放
        scale_factor = 0.8 + 0.4 * torch.rand(1).item()  # 随机缩放因子[0.8, 1.2]
        artifact_patch = artifact_patch * scale_factor
        # perfect_patch = perfect_patch * scale_factor
        # 将artifact_patch复制为3通道以匹配ResNet输入
        perfect_patch = perfect_patch.repeat(3, 1, 1)  # 单通道 -> 三通道
        artifact_patch = artifact_patch.repeat(3, 1, 1)  # 单通道 -> 三通道
        
        if self.transform:
            artifact_patch = self.transform(artifact_patch)
            perfect_patch = self.transform(perfect_patch)
        
        

        return artifact_patch, perfect_patch

# 计算标准差、平均值
def calculate_mean_std(images):
    # 将图像数据展平，然后计算平均值和标准偏差
    mean = np.mean(images)
    std = np.std(images)

    return mean, std

# 定义损失函数
criterion = nn.MSELoss()

if __name__ == '__main__':
    # 读取数据并做适当格式转换
    file_path = '/home/yinjie/SOS_project/SOS_resnet/img_recon.mat'
    with h5py.File(file_path, 'r') as f:
        img_recon = np.array(f['img_recon'])

    images_artifact = torch.tensor(img_recon).unsqueeze(1)
    print(images_artifact.shape)

    file_path = '/home/yinjie/SOS_project/SOS_resnet/GT_image.mat'
    with h5py.File(file_path, 'r') as f:
        img_GT = np.array(f['GT_image'])

    images_perfect = torch.tensor(img_GT).unsqueeze(1)
    print(images_perfect.shape)

    #---------计算归一化参数-------------------------
    images = np.concatenate((images_artifact, images_perfect), axis=0)
    # mean, std = calculate_mean_std(images)
    # print("平均值: ", mean)
    # print("标准偏差: ", std)

    transform = transforms.Normalize([0.014349332014453183]*3, [0.17099603149748466]*3)  # 该参数需要从TRAIN过程中获得，并手动输入

    SOS_dataset = SOS(images_artifact, images_perfect, transform=transform)
    dataset_size = len(SOS_dataset)
    train_size = int(0.7 * dataset_size)
    valid_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - valid_size

    train_indices = list(range(train_size))
    valid_indices = list(range(train_size, train_size + valid_size))
    test_indices = list(range(train_size + valid_size, dataset_size))

    train_dataset = Subset(SOS_dataset, train_indices)
    valid_dataset = Subset(SOS_dataset, valid_indices)
    test_dataset = Subset(SOS_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=8, pin_memory=False, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, pin_memory=False, shuffle=True, num_workers=4)

    # # 加载预训练的 ResNet-50 模型
    # pretrained_resnet50 = models.resnet50(pretrained=True)


    # # 修改输出层，使其输出单通道的图像
    # num_filters = pretrained_resnet50.fc.in_features  # 获取全连接层输入特征数量
    # pretrained_resnet50.fc = nn.Sequential(
    #     nn.ConvTranspose2d(num_filters, 128, kernel_size=4, stride=2, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # 输出单通道图像
    # )
    pretrained_resnet50 = models.resnet50(pretrained=True)

# 移除原始的全连接层
    features = list(pretrained_resnet50.children())[:-2]  # 保留除最后两层（全局平均池化和全连接层）之外的所有层
    pretrained_resnet50 = nn.Sequential(*features)

    decoder = nn.Sequential(
        # 第一次反卷积，将特征图尺寸放大 2 倍
        nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 第二次反卷积，将特征图尺寸放大 2 倍
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 第三次反卷积，将特征图尺寸放大 2 倍
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 第四次反卷积，将特征图尺寸放大 2 倍
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 第五次反卷积，将特征图尺寸放大 2 倍
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 最后一个卷积层，输出单通道图像
        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    )

    # 组合特征提取部分和解码部分
    model = nn.Sequential(
        pretrained_resnet50,
        decoder
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)  # 将整个模型移动到设备上

    # print(model)

    # inputtest = torch.randn(1,3,224,224)
    # outputtest = model(inputtest)
    # print('输出的形状是',outputtest.shape)

    model_dir = 'SOS_RESNET/models_resnet'
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter("SOS_RESNET/logs_resnet")

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # 创建余弦退火学习率调度器

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=400,  # 周期长度（这里设为总epoch数，使学习率在整个训练中从初始值降到eta_min）
        eta_min=1e-6       # 最小学习率（可调整）
    )
   

    num_epochs = 400
    total_train_step = 0
    loss_total = 0  # 计算每一个epoch对应的loss

    # torch.cuda.memory_allocated()
    # torch.cuda.memory_cached()

    start_time = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"------第{epoch + 1}轮训练开始-------")
        model.train()

        for images_artifact, images_perfect in train_loader:
            images_artifact, images_perfect = images_artifact.float().to(device), images_perfect.float().to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images_artifact)
                # 将目标图像转换为单通道
                if images_perfect.shape[1] != outputs.shape[1]:
                    images_perfect = images_perfect[:, 0:1, :, :]  # 取第一个通道
                loss = criterion(outputs, images_perfect)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_step = total_train_step + 1

           
            if total_train_step % 40 == 0:
                print(f"训练次数为：{total_train_step}, loss:{loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images_artifact, images_perfect in valid_loader:
                images_artifact, images_perfect = images_artifact.float().to(device), images_perfect.float().to(device)
                outputs = model(images_artifact)
                if images_perfect.shape[1] != outputs.shape[1]:
                    images_perfect = images_perfect[:, 0:1, :, :]  # 取第一个通道
                loss = criterion(outputs, images_perfect)
                # print('outputs的形状是',outputs.shape)
                # print('image_perfect的形状是',images_perfect.shape)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            print("验证集上的损失: ", valid_loss)
        scheduler.step()  # 每个epoch结束时调整学习率
          # （可选）记录当前学习率到TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        print('当前学习率是',current_lr)
        writer.add_scalar('learning_rate', current_lr, epoch)

        torch.save(model.state_dict(), os.path.join(model_dir, f"mynet_{epoch}.pth"))
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss.item()
        # }, os.path.join(model_dir, f"mynet_{epoch}.pth"))

        torch.cuda.empty_cache()

    print("执行时间：", time.time() - start_time, "秒")