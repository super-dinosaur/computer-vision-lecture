import os
import shutil
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

# 该文件用于数据前期处理，使得RGB的GT变成模型训练可接受的图像，basedir改成voc所在地址
# 定义路径
base_dir = "./data/VOCdevkit/VOC2012"
splits_dir = os.path.join(base_dir, "ImageSets/Segmentation")
images_dir = os.path.join(base_dir, "JPEGImages")
annot_dir = os.path.join(base_dir, "Annotations")

# 创建新的目录结构
output_base = "/home/Cv3022244006/code/data"
for split in ['train', 'val']:
    for subdir in ['images', 'masks']:
        os.makedirs(os.path.join(output_base, split, subdir), exist_ok=True)

# VOC数据集的调色板和类别映射
VOC_COLORMAP = [
    [0, 0, 0],        # 背景
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # diningtable
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # pottedplant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128],     # tvmonitor
]

# 创建RGB到类别索引的映射字典
rgb_to_label = {tuple(rgb): i for i, rgb in enumerate(VOC_COLORMAP)}

def read_split_file(split:str):
    """读取分割数据集的文件列表"""
    with open(os.path.join(splits_dir, f"{split}.txt"), 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_annotation(filename):
    """直接处理PNG格式的分割标注图像"""
    # 获取分割标注文件路径
    seg_path = os.path.join(base_dir, "SegmentationClass", f"{filename}.png")
    
    if not os.path.exists(seg_path):
        print(f"Warning: Segmentation mask not found for {seg_path}")
        return None
    
    # 读取分割标注图像，确保以RGB模式读取
    seg_img = Image.open(seg_path).convert('RGB')
    seg_arr = np.array(seg_img)
    
    # 创建输出标签图像
    height, width = seg_arr.shape[:2]
    label_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 将RGB值转换为类别索引
    for y in range(height):
        for x in range(width):
            rgb = tuple(seg_arr[y, x].tolist())  # 将numpy数组转换为tuple
            label_mask[y, x] = rgb_to_label.get(rgb, 0)  # 默认为背景类（0）
    
    return Image.fromarray(label_mask)

def process_split(split:str):
    """处理训练集或验证集"""
    file_list = read_split_file(split)
    
    for filename in file_list:
        # 复制图像
        src_img = os.path.join(images_dir, f"{filename}.jpg")
        dst_img = os.path.join(output_base, split, "images", f"{filename}.jpg")
        shutil.copy2(src_img, dst_img)
        
        # 处理标注
        mask = process_annotation(filename)
        if mask is not None:
            mask.save(os.path.join(output_base, split, "masks", f"{filename}.png"))
        
        print(f"Processed {filename}")

def main():
    # 处理训练集和验证集
    for split in ['train', 'val']:
        print(f"Processing {split} set...")
        process_split(split)
        print(f"Finished processing {split} set")

if __name__ == "__main__":
    main()
