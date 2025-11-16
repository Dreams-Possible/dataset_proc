import cv2
import os

#视频采样并生成帧
def video_to_frames(video_path:str,frames_path:str,sample:int=1):
    #打开视频
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    #创建输出帧目录
    os.makedirs(frames_path, exist_ok=True)
    #总帧数
    sum_frames=0
    #采样帧数
    sample_frames=0
    #采样帧
    while True:
        #读取一帧
        ok,frame=cap.read()
        if not ok:
            break
        #采样
        if sum_frames%sample==0:
            save_path=os.path.join(frames_path,f"{sample_frames}.jpg")
            cv2.imwrite(save_path,frame)
            sample_frames+=1
            print(f"Generated {sample_frames} frames...")
        sum_frames+=1
    #释放视频
    cap.release()
    print(f"Done. Sample {sample_frames} frames into {frames_path}")
