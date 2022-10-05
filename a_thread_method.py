import threading
import cv2


class RTSCapture(cv2.VideoCapture):
    # 使用 RTSCapture.create 方法创建，请不要直接实例化

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]  # 用于识别实时流

    @staticmethod
    def create(url, *schemes):
        """实例化&初始化
        rtscap = RTSCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
        """
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            pass

        return rtscap

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()


if __name__ == '__main__':
    rtscap = RTSCapture.create("rtsp://192.168.88.221/live")
    rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向

    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()  # read_latest_frame() 替代 read()
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if not ok:
            continue

        # 帧处理代码写这里  type(frame) = numpy.ndarray

        cv2.putText(frame, "Hello World", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (46, 204, 113), 3, cv2.LINE_AA)
        cv2.imshow("thread", frame)

    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()