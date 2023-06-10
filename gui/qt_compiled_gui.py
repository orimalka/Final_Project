# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt_designer_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Ripeness(object):
    def setupUi(self, Ripeness):
        Ripeness.setObjectName("Ripeness")
        Ripeness.resize(550, 261)
        self.label = QtWidgets.QLabel(Ripeness)
        self.label.setGeometry(QtCore.QRect(20, 20, 47, 13))
        self.label.setObjectName("label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(Ripeness)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 40, 521, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayoutWidget = QtWidgets.QWidget(Ripeness)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(380, 80, 144, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout.addWidget(self.lineEdit_2)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(Ripeness)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(20, 80, 331, 101))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout_2.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_2.addWidget(self.checkBox)
        self.checkBox_3 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout_2.addWidget(self.checkBox_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.groupBox = QtWidgets.QGroupBox(self.horizontalLayoutWidget_2)
        self.groupBox.setObjectName("groupBox")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 50, 158, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 20, 158, 17))
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(Ripeness)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 190, 521, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)

        self.retranslateUi(Ripeness)
        QtCore.QMetaObject.connectSlotsByName(Ripeness)

    def retranslateUi(self, Ripeness):
        _translate = QtCore.QCoreApplication.translate
        Ripeness.setWindowTitle(_translate("Ripeness", "Ripeness Detection"))
        self.label.setText(_translate("Ripeness", "File Select: "))
        self.pushButton.setText(_translate("Ripeness", "Browse"))
        self.label_2.setText(_translate("Ripeness", "Yolo Segmentation Resolution"))
        self.lineEdit_2.setPlaceholderText("Default: 640")
        self.checkBox_2.setText(_translate("Ripeness", "Count detections"))
        self.checkBox_2.setChecked(True)
        self.checkBox.setText(_translate("Ripeness", "Save segmented Images"))
        self.checkBox_3.setText(_translate("Ripeness", "Show SORT tracks"))
        self.groupBox.setTitle(_translate("Ripeness", "Model Mode"))
        self.radioButton_2.setText(_translate("Ripeness", "Run Normal Yolo7"))
        self.radioButton.setText(_translate("Ripeness", "Run Ripeness model"))
        self.radioButton.setChecked(True)
        self.pushButton_3.setText(_translate("Ripeness", "Open Export Folder"))
        self.pushButton_2.setText(_translate("Ripeness", "Run Prediction"))
