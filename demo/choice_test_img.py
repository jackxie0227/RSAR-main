import os
import random
import shutil

def copy_random_files(src_img_dir, src_ann_dir, dest_img_dir, dest_ann_dir, num_files=100):
    # 确保目标目录存在
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_ann_dir, exist_ok=True)

    # 获取所有图片文件（过滤扩展名）
    valid_ext = {'.bmp', '.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in os.listdir(src_img_dir) 
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    
    # 随机选择文件（不超过实际数量）
    selected_files = random.sample(image_files, min(num_files, len(image_files)))
    print(f"Found {len(image_files)} images, selecting {len(selected_files)}")

    # 复制文件和对应的标注
    copied_count = 0
    for img_file in selected_files:
        # 获取文件名前缀
        base_name = os.path.splitext(img_file)[0]
        
        # 构建标注文件路径
        ann_file = f"{base_name}.txt"
        ann_path = os.path.join(src_ann_dir, ann_file)
        
        # 检查标注文件是否存在
        if not os.path.exists(ann_path):
            print(f"Warning: Annotation {ann_file} not found, skip {img_file}")
            continue
            
        # 构建源和目标路径
        src_img = os.path.join(src_img_dir, img_file)
        dest_img = os.path.join(dest_img_dir, img_file)
        
        src_ann = ann_path
        dest_ann = os.path.join(dest_ann_dir, ann_file)
        
        # 执行复制
        shutil.copy2(src_img, dest_img)
        shutil.copy2(src_ann, dest_ann)
        copied_count += 1

    print(f"Successfully copied {copied_count} pairs of files")

if __name__ == "__main__":
    # 路径配置
    source_img_dir = "RSAR/test/images/"
    source_ann_dir = "RSAR/test/annfiles/"
    
    dest_img_dir = "demo/input/images/"
    dest_ann_dir = "demo/input/annfiles/"

    # 执行复制
    copy_random_files(
        src_img_dir=source_img_dir,
        src_ann_dir=source_ann_dir,
        dest_img_dir=dest_img_dir,
        dest_ann_dir=dest_ann_dir,
        num_files=100
    )