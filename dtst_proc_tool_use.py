from dtst_proc_tool import dpt_aio


# 视频路径
input_video_path="./test.mp4"
# 输出文件夹路径
output_folder_path="./testout"


# 一条龙处理
dpt_aio(input_video_path, output_folder_path, augment=True, split=True, sample=1, blur_fill=True)
