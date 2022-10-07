import cv2
import subprocess as sp

rtsp_url = 'rtsp://172.17.144.1/videoStream'
cap = cv2.VideoCapture('resource/test0.mp4')

# Get video information
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
command = ['ffmpeg',  # linux不用指定
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',  # 像素格式
           '-s', "{}x{}".format(width, height),
           # '-r', str(fps),  # 帧率，一般为29.97
           '-re',  # 控制读取速度：按帧率读取，若不加这一行，则使用最高速传递
           '-i', '-',
           '-c:v', 'libx264',  # 视频编码方式
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'rtsp',  # flv rtsp
           '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
           rtsp_url]  # rtsp rtmp
p = sp.Popen(command, stdin=sp.PIPE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('resource/test0.mp4')
        continue
        # print("Opening camera is failed")
        # break
    # frame = 你的图像处理的函数(frame)
    p.stdin.write(frame.tostring())
