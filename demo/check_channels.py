from PIL import Image
import sys

try:
    img_path = 'D:\CodeSoftware\VisualStudioCode\Project\RSAR-main\demo\input\hugeImages\MiniSAR20050519p0001image008.png'
    with Image.open(img_path) as img:
        print(f"通道数: {len(img.getbands())} ({img.mode})")
except IndexError:
    print("使用方法：python check.py <图片路径>")
except FileNotFoundError:
    print(f"文件不存在：{sys.argv[1]}")
except Exception as e:
    print(f"检测失败：{str(e)}")