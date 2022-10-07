from flask import Flask, render_template, Response
import cv2
# import numpy as np
# from PIL import Image
#
# from a_thread_method import RTSCapture
# from inference import Fisheeye_Inference

VideoStreamServer_url = 'rtsp://192.168.88.210/live'


class VideoTest(object):
    def __init__(self):
        # self.model = Fisheeye_Inference()
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(VideoStreamServer_url)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, img = self.video.read()
        print('get')
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 在这里处理视频帧
        # cv2.putText(image, "Hello World", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
        #             2, (46, 204, 113), 3, cv2.LINE_AA)
        # self.f += 1
        # if self.f % 3 == 0:
        #     img = self.model.__call__(img)
        # else:

        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


app = Flask(__name__, static_folder='./static')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    while True:
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoTest()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.88.210', debug=True, port=5000)
