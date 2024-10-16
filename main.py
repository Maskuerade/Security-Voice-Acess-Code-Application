from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QVBoxLayout, QWidget, QSlider, QComboBox, QGraphicsRectItem,QGraphicsView,QGraphicsScene,QAbstractItemView,QTableWidgetItem,QCheckBox,QHBoxLayout,QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor,QMouseEvent
from PyQt5.QtCore import Qt, QRectF,pyqtSignal,QFile,QTextStream
from PyQt5.QtCore import Qt, QRectF, QObject, pyqtSignal
import sys
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from modules import accessGate, recongize ,plotSpectro



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.setWindowTitle("Voice Control Program")
        self.apply_stylesheet("ManjaroMix.qss")
        self.button=self.recordBtn
        self.users= [
            {"name":"ashf","access":0},
            {"name":"mask","access":0},
            {"name":"morad","access":0},
            {"name":"ziad","access":0},
            {"name":"emad","access":0},
            {"name":"Lama Zakria","access":0},
            {"name":"Lamees Mohee","access":0},
            {"name":"Amira Ashraf","access":0},
        ]
        init_connectors(self)
        self.outputLabel.setText("Press the record button to start recording")
        self.words = [
            "Open middle door","Unlock the gate","Grant me access",
        ]
       # modules.generateAudiosSpectrograms()

    def record(self):
        self.outputLabel.setText("")
        accessUsers = []
        for i, user in enumerate(self.users):
            if user["access"]:
                accessUsers.append(user["name"])
        sentence, sentencesProbabilities = accessGate(self.recordBtn)
        plotSpectro(self)
        self.outputLabel.setText(f"Sentence is :{sentence}")
        personName,class_probabilities = recongize()
        isOther = max(class_probabilities)>0.6
        self.recordBtn.setText("Start Recording")
        for user in self.users:
            if user["name"] == personName:
                
                if(isOther): 
                    personName="Other"
                    self.outputLabel.setText(f" \nHello Mr, {personName}\nAccess: Denied\nPress the record button to record again")
                    break
                if (user["access"] == 0 ):
                    self.outputLabel.setText(f"Sentence is :{sentence} \nHello Mr, {personName}\nAccess: Denied\nPress the record button to record again")
                else:
                    self.outputLabel.setText(f"Sentence is :{sentence} \nHello Mr, {personName}\nAccess: Granted\nPress the record button to record again")
        # for i, probability in enumerate(class_probabilities[0]):
        #     self.voiceControlTable.setItem(i,2,QTableWidgetItem(f"  {probability}%")) 
        for i in range(len(self.users)):
            self.voiceControlTable.setItem(i, 2, QTableWidgetItem(f"  {class_probabilities[i]*100 }%"))            

    def setUserAccess(self,index,state):
        self.users[index]["access"] = bool(state)  # check box

    def apply_stylesheet(self, stylesheet_path):
        stylesheet = QFile(stylesheet_path)
        if stylesheet.open(QFile.ReadOnly | QFile.Text):
            stream = QTextStream(stylesheet)
            qss = stream.readAll()
            self.setStyleSheet(qss)
        else:
            print(f"Failed to open stylesheet file: {stylesheet_path}")    
            
            
def init_connectors(self):
     self.recordBtn.clicked.connect(lambda:self.record())
     self.voiceControlTable.setColumnCount(3)
     self.voiceControlTable.setHorizontalHeaderLabels(["Name", "Access","Ratio"])
     self.voiceControlTable.setRowCount(len(self.users))
     self.voiceControlTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
     self.voiceControlTable.setColumnWidth(0, 200)
     self.voiceControlTable.setColumnWidth(1, 60)
     self.voiceControlTable.setColumnWidth(2,100)
   #  self.voiceControlTable.setColumnWidth(2, 50)
     self.voiceControlTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
     self.voiceControlTable.setAlternatingRowColors(True)
     self.voiceControlTable.setShowGrid(False)
     for i, user in enumerate(self.users):
            self.voiceControlTable.setItem(i, 0, QTableWidgetItem(user["name"]))
            widget = QWidget()
            checkbox = QCheckBox()
            checkbox.setChecked(user["access"])
            checkbox.stateChanged.connect(lambda state, i=i: self.setUserAccess(i, state))
            layout = QHBoxLayout()
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)
            self.voiceControlTable.setCellWidget(i, 1, widget)
            self.voiceControlTable.setItem(i, 2, QTableWidgetItem("  0%"))




def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
