import cv2
import numpy as np
RSAR_COLORS = {
    'ship': (60, 20, 220),    # 红色
    'aircraft': (228, 0, 106),  # 蓝色
    'car': (230, 0, 0),     # 紫色
    'tank': (0, 182, 0),      # 绿色
    'bridge': (0, 182, 200),  # 黄色
    'harbor': (200, 182, 0)   # 青色    
}

SIVED_COLORS = {
    'car':(230, 0, 0)
}

def draw_rotated_rectangles(img, meta, pred_instances, threshold=0.3):
    """绘制旋转框并返回过滤后的结果。
    Args:
        img: 原始图像
        pred_instances: 包含预测旋转框、置信度和标签的实例，格式为一个对象，包含以下属性：
            - bboxes: 旋转框的坐标，形状为 (N, 5)，每个框包含 (cx, cy, w, h, angle)
            - scores: 置信度分数，形状为 (N,)
            - labels: 标签，形状为 (N,)
        threshold: 置信度阈值
    Returns:
        tuple: (绘制后的图像, 过滤后的旋转框列表, 过滤后的置信度列表)
    """
    draw_img = img.copy()
    
    try:
        classes = meta.get('classes') 
        COLORS = {}
        for class_name in classes:
            if class_name in RSAR_COLORS:
                COLORS[class_name] = RSAR_COLORS[class_name]
            else:
                raise ValueError("no valid class colour")
    except AttributeError:
        raise ValueError("model.dataset_meta must contain 'classes' key with class names.")
    
    mask = pred_instances.scores >= threshold
    filtered_bboxes = pred_instances.bboxes[mask].cpu().numpy()
    filtered_scores = pred_instances.scores[mask].cpu().numpy()
    filtered_labels = pred_instances.labels[mask].cpu().numpy()

    # 预测框绘制
    for bbox, score, label in zip(filtered_bboxes, filtered_scores, filtered_labels):
        if score < threshold: # 阈值滤除
            continue
        cx, cy, w, h, angle = bbox
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        class_name = classes[label]
        color = COLORS[class_name]
        
        w_2 = w / 2
        h_2 = h / 2
        
        p1 = (int(cx - w_2 * cos_a + h_2 * sin_a), int(cy - w_2 * sin_a - h_2 * cos_a))
        p2 = (int(cx + w_2 * cos_a + h_2 * sin_a), int(cy + w_2 * sin_a - h_2 * cos_a))
        p3 = (int(cx + w_2 * cos_a - h_2 * sin_a), int(cy + w_2 * sin_a + h_2 * cos_a))
        p4 = (int(cx - w_2 * cos_a - h_2 * sin_a), int(cy - w_2 * sin_a + h_2 * cos_a))
        
        pts = np.array([p1, p2, p3, p4], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts], isClosed=True, color=color, thickness=2)
    
    # 预测类别、置信度绘制
    for bbox, score, label in zip(filtered_bboxes, filtered_scores, filtered_labels):
        if score < threshold: # 阈值滤除
            continue
        cx, cy, w, h, angle = bbox
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        class_name = classes[label]
        
        w_2 = w / 2
        h_2 = h / 2
        
        p1 = (int(cx - w_2 * cos_a + h_2 * sin_a), int(cy - w_2 * sin_a - h_2 * cos_a))
        p2 = (int(cx + w_2 * cos_a + h_2 * sin_a), int(cy + w_2 * sin_a - h_2 * cos_a))
        p3 = (int(cx + w_2 * cos_a - h_2 * sin_a), int(cy + w_2 * sin_a + h_2 * cos_a))
        p4 = (int(cx - w_2 * cos_a - h_2 * sin_a), int(cy - w_2 * sin_a + h_2 * cos_a))
        
        label_text = f'{classes[label]} {score:.2f}'
        cv2.putText(draw_img, label_text, (int(cx), int(cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    return draw_img, filtered_bboxes, filtered_scores, filtered_labels