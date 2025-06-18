import os
import glob

def clean_label_files(directory):
    """
    清理标注文件，删除前两行，只保留标注信息
    
    Args:
        directory (str): 包含labels文件夹的目录路径
    """
    labels_path = os.path.join(directory, 'labels')
    if not os.path.exists(labels_path):
        print(f"警告: {labels_path} 不存在")
        return
    
    label_files = glob.glob(os.path.join(labels_path, '*.txt'))
    
    for file_path in label_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) > 2:
            # 只保留第三行及之后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines[2:])
            print(f"已清理文件: {file_path}")
        else:
            print(f"警告: {file_path} 行数不足，跳过处理")

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理训练集
    train_dir = os.path.join(script_dir, 'train')
    print("开始处理训练集...")
    clean_label_files(train_dir)
    
    # 处理测试集
    test_dir = os.path.join(script_dir, 'test')
    print("开始处理测试集...")
    clean_label_files(test_dir)
    
    print("清理完成!")

if __name__ == "__main__":
    main()