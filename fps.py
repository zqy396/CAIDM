import cv2
import os

def extract_frames(video_path, out_dir, frame_interval=1):
    """
    从视频中按帧间隔抽帧
    :param video_path: 视频路径
    :param out_dir: 输出目录
    :param frame_interval: 每隔多少帧取一张（1=每帧都取）
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps={fps:.2f}, 总帧数={total}, 时长={total/fps:.2f}s")

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"共保存 {saved} 张图片到 {out_dir}")

if __name__ == "__main__":
    extract_frames("test.mp4", "frames", frame_interval=30)  # 约每 1 秒一帧(按30fps算)