from xml.dom.expatbuilder import FragmentBuilder
from PyQt5 import QtGui
from PyQt5 import QtCore
import cv2
import sys
from PyQt5.QtWidgets import  QFrame, QWidget, QLabel, QApplication, QListWidget, QGridLayout, QBoxLayout, QListWidgetItem
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from socket import *
import json
import queue
import numpy as np

# 좌표 list id = 0
class Item(QWidget):
    def __init__(self, id, x , y, time): # 파라미터는 add_list_item의 custom_wiget에서 가져온다
        QWidget.__init__(self, flags=Qt.Widget)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        pb = QLabel("ID: {}\tX: {}\tY: {}\tTime: {}".format(id, x , y, time))
        layout.addWidget(pb)
        layout.setSizeConstraint(QBoxLayout.SetFixedSize)
        self.setLayout(layout)

# Total list
class Total(QWidget):
    def __init__(self, total): # 파라미터는 add_list_item의 custom_wiget에서 가져온다
        QWidget.__init__(self, flags=Qt.Widget)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        pb = QLabel("{}".format(total)) # {}의 값만 실시간으로 바뀜
        layout.addWidget(pb)
        layout.setSizeConstraint(QBoxLayout.SetFixedSize)
        self.setLayout(layout)        

# 동영상 쓰레드
class Thread(QThread):
    def __init__(self):
        super().__init__()

    changePixmap_0 = pyqtSignal(QImage)
    changePixmap_1 = pyqtSignal(QImage)
    changePixmap_2 = pyqtSignal(QImage)
    changePixmap_3 = pyqtSignal(QImage)

    def preprocessing(self, frame , change_QT):
        rgbImage_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_0, w_0, ch_0 = rgbImage_0.shape
        bytesPerLine_0 = ch_0 * w_0
        convertToQtFormat_0 = QImage(rgbImage_0.data, w_0, h_0, bytesPerLine_0, QImage.Format_RGB888)
        p = convertToQtFormat_0.scaled(360, 240, Qt.KeepAspectRatio) # 영상 가로 x 세로
        change_QT.emit(p)
    
    def run(self):
        # url0 = 'rtsp://192.168.33.19:8554/mystream' 
        vedio = 'sample_vedio.mp4'
        cap_0 = cv2.VideoCapture(vedio)

        while True:

            ret_0, frame = cap_0.read()
            
            # 카메라 4대
            if ret_0:
                frame_0 = frame[0:270,0:380] # 480
                frame_1 = frame[0:270,380:760] # 480:960
                frame_2 = frame[270:540,0:380]
                frame_3 = frame[270:540,380:760]

                self.preprocessing(frame_0, self.changePixmap_0)
                self.preprocessing(frame_1, self.changePixmap_1)
                self.preprocessing(frame_2, self.changePixmap_2)
                self.preprocessing(frame_3, self.changePixmap_3)

            
# UDP 쓰레드
class UDP_Thread(QThread):
    changedata_0 = pyqtSignal(dict)
    changedata_1 = pyqtSignal(dict)
    changedata_2 = pyqtSignal(dict)
    changedata_3 = pyqtSignal(dict)

    # changePixmap
    changePixmap_main_0 = pyqtSignal(QImage)
    changePixmap_main_1 = pyqtSignal(QImage)

    deletedata = pyqtSignal()
    q = queue.Queue
    data_list = []

    def preprocessing(self, posx, posy, time, change_QT):
        data = dict(xpos = posx, ypos = posy, time =time)  # add_list_item에서 custom_wiget 이랑 연결
        change_QT.emit(data)


    # changePixmap
    def main_preprocessing(self, frame , change_QT):
        rgbImage_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # (1000, 900, 3)
        h_0, w_0, ch_0 = rgbImage_0.shape
        bytesPerLine_0 = ch_0 * w_0
        convertToQtFormat_0 = QImage(rgbImage_0.data, w_0, h_0, bytesPerLine_0, QImage.Format_RGB888)
        p = convertToQtFormat_0.scaled(1000, 350) 
        p = p.transformed(QtGui.QTransform())
        change_QT.emit(p)


    def __init__(self):
        super().__init__()
        serverPort = 32002
        self.sock = socket(AF_INET, SOCK_DGRAM)
        self.sock.bind(('192.168.33.6', serverPort))   

    
    def run(self):
        while True:
            message, clientAddress = self.sock.recvfrom(153600)
            message = message.decode()
            message = json.loads(message)
            self.data_list.append(message)
            datas = message['DATA_LIST']

            print('original = ',datas)

            del datas[:-10] # 데이터 수 삭제
            data_list = list({v['WORKER']:v for v in datas}.values()) # WORKER 값 중복제거
            print('data_list = ',data_list)


            data_0 = {}
            data_1 = {}
            data_2 = {}
            data_3 = {}
            
            img = np.zeros((800,1200,3), np.uint8) # 좌표 평면도 크기

            # cv2.line(파일, 시작, 종료, 색상, 두께)
            cv2.line(img, (1200, 800), (0, 800), (255, 255, 255), 6) # 맨 아래 선
            cv2.line(img, (1200, 800), (1200, 0), (255, 255, 255), 6) # 오른쪽 끝 선

            img_h = int(img.shape[0])
            img_w = int(img.shape[1])

            num_vertical_slice = range(0, int(img_h / 50))
            num_horizontal_slice = range(0,int(img_w / 50))

            for n in num_horizontal_slice:
                cv2.line(img,(n * 50, 0),(n * 50, img_w),(255,255,255),2)

            for n in num_vertical_slice:
                cv2.line(img,(0,n * 50),(img_w, n * 50),(255,255,255),2)


         

            for data in data_list:

                if data['WORKER']  == '0':
                    
                    data['WORKER'] = {'IMG_X': data['IMG_X'], 'IMG_Y': data['IMG_Y'], 'DATE':data['DATE'] }
                    data_0 = data['WORKER']
                    pyqt_list_0 = list(data_0.values())
                    # print(pyqt_list_0)
                    # print(len(pyqt_list_0))

                    try:
                        # self.count = pyqt_list_0[3] # 총 인원
                        self.posx = pyqt_list_0[0]
                        self.posy = pyqt_list_0[1]
                        self.time = pyqt_list_0[2][8:10] + ':' + pyqt_list_0[2][10:12]
                        self.preprocessing(self.posx, self.posy, self.time , self.changedata_0)

                        pos_x = int(self.posx)
                        pos_y = int(self.posy)
                        
                        cv2.circle(img, (pos_x , pos_y), 10, (0,0,255), -1) 
                        cv2.putText(img, "Cam0",  (pos_x-40 ,pos_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        self.main_preprocessing(img, self.changePixmap_main_0)
                       
                    except:
                        pass

                elif data['WORKER'] == '1':
                    
                    data['WORKER'] = {'IMG_X': data['IMG_X'], 'IMG_Y': data['IMG_Y'], 'DATE':data['DATE'] }
                    data_1 = data['WORKER']
                    pyqt_list_1 = list(data_1.values())

                    try:
                        # self.count = pyqt_list_1[3] # 총 인원
                        self.posx = pyqt_list_1[0]
                        self.posy = pyqt_list_1[1]
                        self.time = pyqt_list_1[2][8:10] + ':' + pyqt_list_1[2][10:12]
                        self.preprocessing(self.posx, self.posy, self.time, self.changedata_1)

                        pos_x = int(self.posx)
                        pos_y = int(self.posy)
                        
                        cv2.circle(img, (pos_x , pos_y), 10, (0,0,255), -1) 
                        cv2.putText(img, "Cam1",  (pos_x-40 ,pos_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        self.main_preprocessing(img, self.changePixmap_main_1)

                    except:
                        pass

                elif data['WORKER'] == '2':
                    data['WORKER'] = {'IMG_X': data['IMG_X'], 'IMG_Y': data['IMG_Y'], 'DATE':data['DATE'] }
                    data_2 = data['WORKER']
                    pyqt_list_2 = list(data_2.values())

                    try:
                        # self.count = pyqt_list_2[3] # 총 인원
                        self.posx = pyqt_list_2[0]
                        self.posy = pyqt_list_2[1]
                        self.time = pyqt_list_2[2][8:10] + ':' + pyqt_list_2[2][10:12]
                        self.preprocessing(self.posx, self.posy, self.time, self.changedata_2)
                    except:
                        pass

                elif data['WORKER'] == '3':
                    data['WORKER'] = {'IMG_X': data['IMG_X'], 'IMG_Y': data['IMG_Y'], 'DATE':data['DATE'] }
                    data_3 = data['WORKER']
                    pyqt_list_3 = list(data_3.values())

                    try:
                        # self.count = pyqt_list_3[3] # 총 인원
                        self.posx = pyqt_list_3[0]
                        self.posy = pyqt_list_3[1]
                        self.time = pyqt_list_3[2][8:10] + ':' + pyqt_list_3[2][10:12]
                        self.preprocessing(self.posx, self.posy, self.time, self.changedata_3)
                    except:
                        pass

    def __del__(self):
        self.sock.close()


# 메인 APP 실행 
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 50
        self.top = 20
        self.width = 1600
        self.height = 920
        self.initUI()

        #grid_image = np.zeros... 

    def setImage_0(self, image_0):
        self.label_0.setPixmap(QPixmap.fromImage(image_0))

    def setImage_1(self, image_1):
        self.label_1.setPixmap(QPixmap.fromImage(image_1))

    def setImage_2(self, image_2):
        self.label_2.setPixmap(QPixmap.fromImage(image_2))   

    def setImage_3(self, image_3):
        self.label_3.setPixmap(QPixmap.fromImage(image_3))

    def setImage_main(self, image_main):
        self.label_main.setPixmap(QPixmap.fromImage(image_main))


    # 좌표 값
    def add_list_item_0(self, qitem):
        item = QListWidgetItem(self.listwidget_0) # 아래에서 listwidget 선언한거 변수에 할당
        custom_widget = Item(id='0', x = qitem['xpos'], y = qitem['ypos'], time = qitem['time'])
        item.setSizeHint(custom_widget.sizeHint())
        self.listwidget_0.setItemWidget(item, custom_widget)
        self.listwidget_0.takeItem(0) # row 번째까지만
        self.listwidget_0.addItem(item) # 바꿔치기
        self.listwidget_0.setStyleSheet("background-color: black; font-size:18pt; color: #ffffff;")

    def add_list_item_1(self, qitem):
        item = QListWidgetItem(self.listwidget_1) # 아래에서 listwidget 선언한거 변수에 할당
        custom_widget = Item(id='1', x = qitem['xpos'], y = qitem['ypos'], time = qitem['time'])
        item.setSizeHint(custom_widget.sizeHint())
        self.listwidget_1.setItemWidget(item, custom_widget)
        self.listwidget_1.takeItem(0) # row 번째까지만
        self.listwidget_1.addItem(item) # 바꿔치기
        self.listwidget_1.setStyleSheet("background-color: black; font-size:18pt; color: #ffffff;")

    def add_list_item_2(self, qitem):
        item = QListWidgetItem(self.listwidget_2) # 아래에서 listwidget 선언한거 변수에 할당
        custom_widget = Item(id='2', x = qitem['xpos'], y = qitem['ypos'], time = qitem['time'])
        item.setSizeHint(custom_widget.sizeHint())
        self.listwidget_2.setItemWidget(item, custom_widget)
        self.listwidget_2.takeItem(0) # row 번째까지만
        self.listwidget_2.addItem(item) # 바꿔치기
        self.listwidget_2.setStyleSheet("background-color: black; font-size:18pt; color: #ffffff;")

    def add_list_item_3(self, qitem):
        item = QListWidgetItem(self.listwidget_3) # 아래에서 listwidget 선언한거 변수에 할당
        custom_widget = Item(id='3', x = qitem['xpos'], y = qitem['ypos'], time = qitem['time'])
        item.setSizeHint(custom_widget.sizeHint())
        self.listwidget_3.setItemWidget(item, custom_widget)
        self.listwidget_3.takeItem(0) # row 번째까지만
        self.listwidget_3.addItem(item) # 바꿔치기
        self.listwidget_3.setStyleSheet("background-color: black; font-size:18pt; color: #ffffff;")
        
    
    # total 값
    def add_total(self, qitem):
        item = QListWidgetItem(self.total_widget) # 아래에서 listwidget 선언한거 변수에 할당
        custom_widget = Total(total = qitem['total'])
        # item.setSizeHint(custom_widget.sizeHint())
        self.total_widget.setItemWidget(item, custom_widget)
        self.total_widget.takeItem(0)
        self.total_widget.addItem(item)
        self.total_widget.setStyleSheet("background-color: black; font-size:32pt; color: #ffffff; font-weight:bold;")


    # 스타일링
    def initUI(self): 

        # 전체 창 크기 설정
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 1. Main 상단 프레임 logo 부분
        # 프레임
        self.frame_logo = QFrame()
        self.frame_logo.setStyleSheet("background-color: #242730;")
        self.frame_logo.setFixedWidth(1605)
        self.frame_logo.setFixedHeight(100)

        # Main logo 이미지
        self.label_logo = QLabel(self.frame_logo)
        self.label_logo.setGeometry(QtCore.QRect(5, 10, 400, 80))
        # self.label_logo.setText(("<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff; background-color: red;\">test</span></p></body></html>"))
        self.label_logo.setPixmap(QtGui.QPixmap("logo_image/PAI_logo.png"))
        

        # 2. Tracking log 부분
        # 프레임
        self.frame_Tracking = QFrame()
        self.frame_Tracking.setStyleSheet("background-color: black")
        self.frame_Tracking.setFixedWidth(1100)
        self.frame_Tracking.setFixedHeight(475)

        # Tracking log 이미지
        self.label_Tracking = QLabel(self.frame_Tracking)
        self.label_Tracking.setGeometry(QtCore.QRect(30, 30, 300, 40))
        # self.label_Tracking.setText(("<html><head/><body><p><span style=\" font-size:16pt; font-weight:900; color:#ffffff;\">TRACKING LOG</span></p></body></html>"))
        self.label_Tracking.setPixmap(QtGui.QPixmap("logo_image/Tracking_log.png"))
        
        # main 동영상 라벨
        self.label_main = QLabel(self.frame_Tracking)
        self.label_main.setFixedWidth(1000)
        self.label_main.setFixedHeight(350)
        self.label_main.move(60,95)
        self.label_main.setStyleSheet("background-color: black")
        

        # 3. 카메라 부분
        self.frame_camera = QFrame()
        self.frame_camera.setStyleSheet("background-color: black")
        self.frame_camera.setFixedWidth(1605)
        self.frame_camera.setFixedHeight(312)
        # self.frame_camera.move(10, 10)
        # self.frame_camera.resize(self.size().width(), self.size().height())

        # Camera log 이미지
        self.label_logo = QLabel(self.frame_camera)
        self.label_logo.setGeometry(QtCore.QRect(20, 15, 200, 40))
        # self.label_logo.setText(("<html><head/><body><p><span style=\" font-size:16pt; font-weight:900; color:#ffffff;\">Camera</span></p></body></html>"))
        self.label_logo.setPixmap(QtGui.QPixmap("logo_image/camera.png"))

        self.label_0 = QLabel(self.frame_camera)
        self.label_0.setGeometry(QtCore.QRect(30, 70, 360, 220)) # right top
        self.label_0.setStyleSheet("background-color: black")
        self.label_0.setAlignment(Qt.AlignCenter)

        self.label_1 = QLabel(self.frame_camera)
        self.label_1.setGeometry(QtCore.QRect(425, 70, 360, 220)) # right top
        self.label_1.setStyleSheet("background-color: black")
        self.label_1.setAlignment(Qt.AlignCenter)

        self.label_2 = QLabel(self.frame_camera)
        self.label_2.setGeometry(QtCore.QRect(820, 70, 360, 220)) # right top
        self.label_2.setStyleSheet("background-color: black")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.label_3 = QLabel(self.frame_camera)
        self.label_3.setGeometry(QtCore.QRect(1215, 70, 360, 220)) # right top
        self.label_3.setStyleSheet("background-color: black")
        self.label_3.setAlignment(Qt.AlignCenter)


        # 4. x,y 좌표 리스트
        # 리스트 프레임
        self.frame_list = QFrame()
        # self.frame_list.setFrameShape(QFrame.Box)
        self.frame_list.setStyleSheet("background-color: black")
        self.frame_list.setFixedWidth(500)
        self.frame_list.setFixedHeight(475)


        # Total 이미지 
        self.label_logo = QLabel(self.frame_list)
        self.label_logo.setGeometry(QtCore.QRect(30, 20, 200, 40))
        # self.label_logo.setText(("<html><head/><body><p><span style=\"background-color: blue; font-size:16pt; font-weight:900; color:#ffffff;\"></span></p></body></html>"))
        self.label_logo.setPixmap(QtGui.QPixmap("logo_image/Total.png"))

        # 명칭 Text
        self.label_logo = QLabel(self.frame_list)
        self.label_logo.setGeometry(QtCore.QRect(42, 95, 410, 60))
        self.label_logo.setText(("<html><head/><body><p><span style=\"background-color: black; font-size:16pt; font-weight:600; color:#ffffff;\">&nbsp; ID &nbsp; &nbsp; &nbsp; Location &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Time &nbsp; &nbsp; &nbsp; &nbsp;</span></p></body></html>"))
        # self.label_logo.setText("test code for count people")

        # 좌표 list 프론트 부분
        # listwidget_0
        self.listwidget_0 = QListWidget(self.frame_list)
        self.listwidget_0.setGeometry(QtCore.QRect(30, 160, 440, 50))
        self.listwidget_0.setFrameShape(QFrame.Panel | QFrame.Plain)
        # listwidget_1
        self.listwidget_1 = QListWidget(self.frame_list)
        self.listwidget_1.setGeometry(QtCore.QRect(30, 220, 440, 50))
        self.listwidget_1.setFrameShape(QFrame.Panel | QFrame.Plain)
        # listwidget_2
        self.listwidget_2 = QListWidget(self.frame_list)
        self.listwidget_2.setGeometry(QtCore.QRect(30, 280, 440, 50))
        self.listwidget_2.setFrameShape(QFrame.Panel | QFrame.Plain)
        # listwidget_3
        self.listwidget_3 = QListWidget(self.frame_list)
        self.listwidget_3.setGeometry(QtCore.QRect(30, 340, 440, 50))
        self.listwidget_3.setFrameShape(QFrame.Panel | QFrame.Plain)

        # total list 프론트 부분
        self.total_widget = QListWidget(self.frame_list)
        self.total_widget.setGeometry(QtCore.QRect(160, 10, 45, 60))
        self.total_widget.setFrameShape(QFrame.Panel | QFrame.Plain)
        
        # 동영상 쓰레드 부분
        th_0 = Thread()
        th_0.changePixmap_0.connect(self.setImage_0)
        th_0.changePixmap_1.connect(self.setImage_1)
        th_0.changePixmap_2.connect(self.setImage_2)
        th_0.changePixmap_3.connect(self.setImage_3)
        # th_0.changePixmap_main.connect(self.setImage_main)
        th_0.start()


        # 좌표 List 쓰레드 부분
        th_1 = UDP_Thread()

        th_1.changedata_0.connect(self.add_list_item_0)
        th_1.changedata_1.connect(self.add_list_item_1)
        th_1.changedata_2.connect(self.add_list_item_2)
        th_1.changedata_3.connect(self.add_list_item_3)

        # total list 는 모든 id 값 적용
        # th_1.changedata_0.connect(self.add_total) 
        # th_1.changedata_1.connect(self.add_total)
        # th_1.changedata_2.connect(self.add_total)
        # th_1.changedata_3.connect(self.add_total)

        th_1.changePixmap_main_0.connect(self.setImage_main)
        th_1.changePixmap_main_1.connect(self.setImage_main)
        th_1.start()

        

        # 6. 그리드 정렬
        # row 행
        # colum 열
        # 행, 열
        
        grid = QGridLayout()
        grid.horizontalSpacing()
        self.setLayout(grid)

        # 기존
        # grid.addWidget(self.frame_logo, 0,0,1,5) # 시작 col, 시작 row, 합쳐질 row 수, 합쳐질 colum 수
        # grid.addWidget(self.frame_Tracking, 1,0,2,4, alignment=Qt.AlignTop)
        # grid.addWidget(self.frame_list, 1,1,2,4, alignment=Qt.AlignRight|Qt.AlignTop)
        # grid.addWidget(self.frame_camera, 2,0)

        grid.addWidget(self.frame_logo, 0,0,1,5) # 시작 col, 시작 row, 합쳐질 row 수, 합쳐질 colum 수
        grid.addWidget(self.frame_Tracking, 1,0,2,4, alignment=Qt.AlignTop)
        grid.addWidget(self.frame_list, 1,1,2,4, alignment=Qt.AlignRight|Qt.AlignTop)
        grid.addWidget(self.frame_camera, 2,0,1,0)

        self.show()

# 메인 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())