import os
import subprocess
import PyQt5.QtWidgets as QtWidgets 
from qt_compiled_gui import Ui_Ripeness

class Gui_Window():
    def __init__(self):
        self.string = "python segment/predict.py --weights yolov7-seg.pt --source "# \"E:\yolo7\Final_Project\orange.jpg\" --class 49 47 --save-seg --img 1280"
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_Ripeness()
        self.ui.setupUi(self.MainWindow)
        self.current_dir = os.getcwd()
        self.ui.lineEdit.setText(self.current_dir)
        self.ui.pushButton.clicked.connect(self.get_file)
        self.ui.pushButton_2.clicked.connect(self.run_script)
        self.ui.radioButton_2.clicked.connect(self.disable_count)
        self.ui.radioButton.clicked.connect(self.enable_count)
        self.ui.pushButton_3.clicked.connect(self.open_explorer)
        os.chdir("../")
        self.MainWindow.show()
        self.app.exec()
        

    def get_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Single File', self.current_dir, '*.*')
        if fileName != "":
            self.current_dir = fileName
            self.ui.lineEdit.setText(self.current_dir)
    
    def run_script(self):
        self.string += "\"" + self.ui.lineEdit.text() + "\""
        if self.ui.radioButton.isChecked(): #run ripeness
            if self.ui.checkBox_2.isChecked(): #count detections
                self.string += " --sort-seg --classes 49 47 --trk "
            if self.ui.checkBox.isChecked(): #save segments
                self.string += " --save-seg "
            if not self.ui.checkBox_3.isChecked(): #show tracks not selected
                self.string += " --trk-clr "
            if self.ui.lineEdit_2.text() != "":
                self.string += " --img "+ self.ui.lineEdit_2.text()
        subprocess.call(self.string,shell=True)
        self.string = "python segment/predict.py --weights yolov7-seg.pt --source "

    
    def disable_count(self):
        self.ui.checkBox_2.setChecked(False)
        self.ui.checkBox_2.setCheckable(False)
        self.ui.checkBox_2.setEnabled(False)

    def enable_count(self):
        self.ui.checkBox_2.setEnabled(True)
        self.ui.checkBox_2.setCheckable(True)
        self.ui.checkBox_2.setChecked(True)
    
    def open_explorer(self):
        workdir = os.getcwd()
        workdir += "\\runs\predict-seg"
        os.startfile(workdir)


if __name__ == '__main__':
   Gui_Window()
