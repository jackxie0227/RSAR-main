import os
import cv2
import numpy as np

METAINFO = {
    'classes': ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
    'palette': [
        (220, 20, 60),   # ship (BGR: 60,20,220)
        (0, 0, 230),     # aircraft
        (106, 0, 228),   # car
        (0, 182, 0),     # tank
        (200, 182, 0),   # bridge
        (0, 182, 200)    # harbor
    ]
}

def parse_annotation(ann_path):
    """
    解析标注文件
    文件格式示例：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    """
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
            
        # 解析坐标和类别
        coords = list(map(float, parts[:8]))
        class_name = parts[8]
        annotations.append({
            'points': np.array(coords, dtype=np.float32).reshape(4, 2),
            'class_name': class_name})
    return annotations

def visualize_rotated_boxes(image, annotations):
    # 创建副本防止修改原图
    vis_image = image.copy()
    
    for ann in annotations:
        points = ann['points']
        class_name = ann['class_name']
        
        # 获取类别颜色（BGR格式）
        class_idx = METAINFO['classes'].index(class_name)
        color = METAINFO['palette'][class_idx][::-1]  # RGB转BGR
        
        # 绘制旋转矩形框
        cv2.polylines(vis_image, [points.astype(np.int32)], isClosed=True, 
                     color=color, thickness=2)
        
        # 添加类别文本
        text_pos = (int(points[0][0]) + 5, int(points[0][1]) + 5)
        cv2.putText(vis_image, class_name, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image

def main():
    # 路径配置
    input_image_dir = 'demo/input/images'
    input_ann_dir = 'demo/input/annfiles'
    output_dir = 'demo/output/gt'
    
    # 支持的图像扩展名
    image_exts = ['.jpg', '.png', '.bmp', '.jpeg']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有图像文件
    for filename in os.listdir(input_image_dir):
        # 检查文件扩展名
        base_name, ext = os.path.splitext(filename)
        if ext.lower() not in image_exts:
            continue
        
        # 构建对应标注文件路径
        ann_path = os.path.join(input_ann_dir, f"{base_name}.txt")
        if not os.path.exists(ann_path):
            continue
        
        # 读取图像
        image_path = os.path.join(input_image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # 解析标注
        annotations = parse_annotation(ann_path)
        
        # 可视化标注
        vis_image = visualize_rotated_boxes(image, annotations)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"vis_{filename}")
        cv2.imwrite(output_path, vis_image)
        print(f"已保存可视化结果：{output_path}")

if __name__ == "__main__":
    main()