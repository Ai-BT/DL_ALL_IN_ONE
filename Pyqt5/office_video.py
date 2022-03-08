
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


class Item(QWidget):
    def __init__(self, id, x , y, time):
        QWidget.__init__(self, flags=Qt.Widget)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        pb = QLabel("ID: {}\tX: {}\tY: {}\tTime: {}".format(id, x , y, time))
        layout.addWidget(pb)
        layout.setSizeConstraint(QBoxLayout.SetFixedSize)
        self.setLayout(layout)

class Thread(QThread):
    def __init__(self):
        super().__init__()

    changePixmap_0 = pyqtSignal(QImage)
    changePixmap_1 = pyqtSignal(QImage)
    changePixmap_2 = pyqtSignal(QImage)
    changePixmap_3 = pyqtSignal(QImage)
    changePixmap_main = pyqtSignal(QImage)

    def preprocessing(self, frame , change_QT):
        rgbImage_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_0, w_0, ch_0 = rgbImage_0.shape
        bytesPerLine_0 = ch_0 * w_0
        convertToQtFormat_0 = QImage(rgbImage_0.data, w_0, h_0, bytesPerLine_0, QImage.Format_RGB888)
        p = convertToQtFormat_0.scaled(360, 240, Qt.KeepAspectRatio) # 영상 가로 x 세로
        change_QT.emit(p)

    def main_preprocessing(self, frame , change_QT):
        rgbImage_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_0, w_0, ch_0 = rgbImage_0.shape
        bytesPerLine_0 = ch_0 * w_0
        convertToQtFormat_0 = QImage(rgbImage_0.data, w_0, h_0, bytesPerLine_0, QImage.Format_RGB888)
        p = convertToQtFormat_0.scaled(900, 1000, Qt.KeepAspectRatio) # 각도 회전으로 인한 변경 (hight x width)
        p = p.transformed(QtGui.QTransform().rotate(90))
        
        change_QT.emit(p)
    
    def run(self):
        url0 = 'rtsp://192.168.33.19:8554/mystream'
        #url0 = 'rtsp://192.168.33.146:8554/mystream'
        cap_0 = cv2.VideoCapture(url0)

        while True:
            ret_0, frame = cap_0.read()
            #print(frame.shape)

            if ret_0:
                # frame_0 = frame[0:540,0:960]
                # frame_1 = frame[0:540,960:1920]
                # frame_2 = frame[540:1080,0:960]
                # frame_3 = frame[540:1080,960:1920]
                # frame_main = frame[0:1080,1920:1920 + 720]
                frame_0 = frame[0:270,0:380] # 480
                frame_1 = frame[0:270,380:760] # 480:960
                frame_2 = frame[270:540,0:380]
                frame_3 = frame[270:540,380:760]
                frame_main = frame[0:540,760:960]

                self.preprocessing(frame_0, self.changePixmap_0)
                self.preprocessing(frame_1, self.changePixmap_1)
                self.preprocessing(frame_2, self.changePixmap_2)
                self.preprocessing(frame_3, self.changePixmap_3)
                self.main_preprocessing(frame_main, self.changePixmap_main)

# UDP
class UDP_Thread(QThread):
    changedata = pyqtSignal(dict)
    deletedata = pyqtSignal()
    q = queue.Queue
    data_list = []

    def preprocessing(self, posx, posy, time, change_QT):
        id = 0
        data = dict(id = id, xpos = posx, ypos = posy, time =time )
        change_QT.emit(data)

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

            pyqt_list = []
            for data in datas:
                for key, value in data.items():
                    img_x = list(data.keys())[4], list(data.values())[4]
                    img_y = list(data.keys())[5], list(data.values())[5]
                    date_time = list(data.keys())[10], list(data.values())[10]
                    pyqt_list.append(img_x)
                    pyqt_list.append(img_y)
                    pyqt_list.append(date_time)
            

            self.posx = pyqt_list[0][1]
            self.posy = pyqt_list[1][1]
            self.time = pyqt_list[2][1][8:10] + ':' + pyqt_list[2][1][10:12]
            self.preprocessing(self.posx, self.posy, self.time, self.changedata)


    def __del__(self):
        self.sock.close()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 50
        self.top = 20
        self.width = 1600
        self.height = 920
        self.initUI()

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


    def add_list_item(self, qitem):
        item = QListWidgetItem(self.listwidget)
        custom_widget = Item(id=qitem['id'], x = qitem['xpos'], y = qitem['ypos'], time = qitem['time'])
        item.setSizeHint(custom_widget.sizeHint())
        self.listwidget.setItemWidget(item, custom_widget)
        self.listwidget.addItem(item)

        self.listwidget.sortItems(Qt.DescendingOrder)
        
        self.listwidget.setStyleSheet("background-color: black; font-size:16pt; color: #ffffff;")


    def delete_list_item(self):
        self.listwidget.clear()


    # sort 부분
    def sortItemsDescending(self):
        self.listwidget.sortItems(Qt.DescendingOrder)




    def onSorted(self):
        if self.sortOrder.isChecked():
            order = Qt.AscendingOrder
        else:
            order = Qt.DescendingOrder

        self.listwidget.sortItems(order)



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
        self.label_main.setFixedHeight(380)
        self.label_main.move(60,80)
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
        self.label_logo.setGeometry(QtCore.QRect(42, 55, 410, 60))
        self.label_logo.setText(("<html><head/><body><p><span style=\"background-color: black; font-size:16pt; font-weight:600; color:#ffffff;\">&nbsp; ID &nbsp; &nbsp; &nbsp; Location &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Time &nbsp; &nbsp; &nbsp; &nbsp;</span></p></body></html>"))

        # 좌표 list 출력
        # currentRowChanged
        self.listwidget = QListWidget(self.frame_list)
        # self.listwidget.setStyleSheet("background-color: black;")
        # self.frame_list.setFrameShape(QFrame.Panel | QFrame.Sunken)
        self.listwidget.setGeometry(QtCore.QRect(30, 100, 440, 370))
        

        # 5. 쓰레드 부분
        th_0 = Thread()
        th_0.changePixmap_0.connect(self.setImage_0)
        th_0.changePixmap_1.connect(self.setImage_1)
        th_0.changePixmap_2.connect(self.setImage_2)
        th_0.changePixmap_3.connect(self.setImage_3)
        th_0.changePixmap_main.connect(self.setImage_main)
        th_0.start()

        th_1 = UDP_Thread()
        th_1.changedata.connect(self.add_list_item)
        # th_1.changedata.connect(self.sortItemsDescending)
        th_1.deletedata.connect(self.delete_list_item)
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


        # grid.addWidget(self.label_0, 2,0)
        # grid.addWidget(self.label_1, 2,1)
        # grid.addWidget(self.label_2, 2,2)
        # grid.addWidget(self.label_3, 2,3)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())