import cv2
import subprocess as sp

rtsp_url = 'rtsp://192.168.88.210/videoStream'
cap = cv2.VideoCapture('resource/test0.mp4')

if cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('result/res.jpg', frame)

# # Get video information
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# command = ['ffmpeg',  # linux不用指定
#            '-y', '-an',
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',  # 像素格式
#            '-s', "{}x{}".format(width, height),
#            '-r', str(fps),  # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
#            '-i', '-',
#            '-c:v', 'libx264',  # 视频编码方式
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'rtsp',  # flv rtsp
#            '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
#            rtsp_url]  # rtsp rtmp
# p = sp.Popen(command, stdin=sp.PIPE)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Opening camera is failed")
#         break
#     # frame = 你的图像处理的函数(frame)
#     p.stdin.write(frame.tostring())
