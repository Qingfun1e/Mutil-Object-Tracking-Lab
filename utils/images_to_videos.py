import cv2
import os
import glob
from pathlib import Path
def images_to_video(image_folder, output_video, fps=30):
    """
    将指定文件夹中的图片序列帧按照名字顺序合成视频
    参数:
    image_folder (str): 包含图片的文件夹路径
    output_video (str): 输出视频的路径和文件名
    fps (int): 视频的帧率，默认为30
    """
    try:
        # 获取文件夹中所有图片的路径
        image_paths = glob.glob(os.path.join(image_folder, '*'))
        # 过滤非图片文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_paths = [path for path in image_paths if Path(path).suffix.lower() in valid_extensions]
        if not image_paths:
            print("错误: 没有找到图片文件")
            return
        # 按照文件名排序
        image_paths.sort()
        # 读取第一张图片获取视频尺寸
        first_image = cv2.imread(image_paths[0])
        if first_image is None:
            print(f"错误: 无法读取图片 {image_paths[0]}")
            return
        height, width, layers = first_image.shape
        # 定义视频编码器和创建VideoWriter对象
        # 使用mp4v编码器，输出mp4格式视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not video.isOpened():
            print("错误: 无法创建视频文件")
            return
        # 遍历所有图片并写入视频
        total_images = len(image_paths)
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图片 {image_path}，已跳过")
                continue
            # 确保图片尺寸与视频尺寸一致
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            video.write(image)
            # 显示进度
            if (i + 1) % 10 == 0 or i + 1 == total_images:
                print(f"已处理 {i + 1}/{total_images} 张图片")
        # 释放资源
        video.release()
        cv2.destroyAllWindows()
        print(f"视频已成功创建: {output_video}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
if __name__ == "__main__":
    # 示例用法
    input_folder = "../MOT17/MOT17/test/MOT17-14-DPM/img1"  # 替换为你的图片文件夹路径
    output_file = "../videos/test_videos2.mp4"  # 输出视频文件名
    frame_rate = 30  # 自定义帧率

    # 调用函数
    images_to_video(input_folder, output_file, frame_rate)
