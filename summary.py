# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\Tugas sek\Kuliah\Materi\Semester 6\Teknik Pengolahan Citra\Project\Project GUI\TPC_DL\summary.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 111, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 440, 111, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(410, 440, 111, 16))
        self.label_3.setObjectName("label_3")
        self.tableWidget_sum = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget_sum.setGeometry(QtCore.QRect(20, 40, 741, 391))
        self.tableWidget_sum.setObjectName("tableWidget_sum")
        self.tableWidget_sum.setColumnCount(0)
        self.tableWidget_sum.setRowCount(0)
        self.tableWidget_confMat = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget_confMat.setGeometry(QtCore.QRect(20, 460, 371, 121))
        self.tableWidget_confMat.setObjectName("tableWidget_confMat")
        self.tableWidget_confMat.setColumnCount(0)
        self.tableWidget_confMat.setRowCount(0)
        self.tableWidget_acc = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget_acc.setGeometry(QtCore.QRect(400, 460, 361, 121))
        self.tableWidget_acc.setObjectName("tableWidget_acc")
        self.tableWidget_acc.setColumnCount(0)
        self.tableWidget_acc.setRowCount(0)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Summary:"))
        self.label_2.setText(_translate("MainWindow", "Confussion matrix:"))
        self.label_3.setText(_translate("MainWindow", "Accuracy :"))
