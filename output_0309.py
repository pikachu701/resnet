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
# from train_resnet0306 import *
import train_resnet0306
import matplotlib.pyplot as plt  # 添加此行

# ---------------------------------------------------模型加载------------------------------------
pretrained_resnet50 = models.resnet50(pretrained=True)

# 移除原始的全连接层
features = list(pretrained_resnet50.children())[:-2]  # 保留除最后两层（全局平均池化和全连接层）之外的所有层
pretrained_resnet50_new = nn.Sequential(*features)

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
model_new = nn.Sequential(
    pretrained_resnet50_new,
    decoder        
)
#---------------------------------------------------------------------------------------------



# 训练好的模型权重文件路径
model_dir = "SOS_RESNET/best_model"
model_file = "best.pth"
model_path = os.path.join(model_dir, model_file)

# 加载模型的状态字典
state_dict = torch.load(model_path)
model_new.load_state_dict(state_dict)

# 创建模型实例
model_output = model_new

# print(model_output)

# 确保模型处于评估模式,并加载模型到GPU
model_output.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_output.to(device)


# 读取数据并做适当格式转换
file_path = '/home/yinjie/SOS_project/SOS_resnet/img_recon.mat'
with h5py.File(file_path, 'r') as f:
    img_recon = np.array(f['img_recon'])

images_artifact = torch.tensor(img_recon).unsqueeze(1).float()   #保持32位精度
# print('完美图像的形状是',images_artifact.shape)
# 对每张完美图像进行中心裁剪
cropped_images_artifact = []
for i in range(images_artifact.shape[0]):
    image = images_artifact[i]
    cropped_image = F.center_crop(image, output_size=(224, 224))
    cropped_images_artifact.append(cropped_image)

cropped_images_artifact = torch.stack(cropped_images_artifact)
print('缺陷图像裁剪后的形状是', cropped_images_artifact.shape)

file_path = '/home/yinjie/SOS_project/SOS_resnet/GT_image.mat'
with h5py.File(file_path, 'r') as f:
    img_GT = np.array(f['GT_image'])

images_perfect = torch.tensor(img_GT).unsqueeze(1).float()

# 对每张真值图像进行中心裁剪
cropped_images_perfect = []
for i in range(images_perfect.shape[0]):
    image = images_perfect[i]
    cropped_image = F.center_crop(image, output_size=(224, 224))
    cropped_images_perfect.append(cropped_image)

cropped_images_perfect = torch.stack(cropped_images_perfect)
print('真值图像裁剪后的形状是', cropped_images_perfect.shape)


# 写入归一化参数，该参数从训练代码中获得
transform = transforms.Normalize([0.014349332014453183]*3, [0.17099603149748466]*3)

# 取出最后400张图像作为测试集
test_images = cropped_images_artifact[-400:]
test_ground_truth = cropped_images_perfect[-400:]

# 对测试集进行归一化处理
normalized_test_images = []
for image in test_images:
    # 由于原图像是单通道，这里复制成三通道以匹配归一化参数
    image = image.repeat(3, 1, 1)
    normalized_image = transform(image)
    normalized_test_images.append(normalized_image)

normalized_test_images = torch.stack(normalized_test_images).to(device)

# 进行推理
with torch.no_grad():
    outputs = model_output(normalized_test_images)

# 将输出转换为numpy数组
outputs_np = outputs.cpu().numpy()
test_images_np = test_images.cpu().numpy()
test_ground_truth_np = test_ground_truth.cpu().numpy()

# 指定保存文件夹
save_folder = 'test_output'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 保存为.mat文件
save_path = os.path.join(save_folder, 'test_output.mat')
with h5py.File(save_path, 'w') as f:
    f.create_dataset('output_images', data=outputs_np)

print(f"测试集输出已保存到 {save_path}")

# 绘制并保存图片
for i in range(400):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制预测后输出图像
    axes[0].imshow(outputs_np[i].squeeze(), cmap='gray')
    axes[0].set_title('Predicted Image')
    axes[0].axis('off')

    # 绘制对应的原始图像
    axes[1].imshow(test_images_np[i].squeeze(), cmap='gray')
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    # 绘制对应的真值图像
    axes[2].imshow(test_ground_truth_np[i].squeeze(), cmap='gray')
    axes[2].set_title('Ground Truth Image')
    axes[2].axis('off')

    # 保存图片
    image_save_path = os.path.join(save_folder, f'comparison_{i}.png')
    plt.savefig(image_save_path)
    plt.close(fig)

print(f"400 张对比图片已保存到 {save_folder} 文件夹中。")  #?is that right?



