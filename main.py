# This Python file uses the following encoding: utf-8
import sys
import os
import shutil
from datetime import datetime
from PIL import Image

import numpy as np
import pandas as pd
import PyQt5
import tensorflow as tf
import cv2

import gui
import summary
import information
import process_thread

app = PyQt5.QtWidgets.QApplication(sys.argv)
window1 = PyQt5.QtWidgets.QMainWindow()
ui = gui.Ui_MainWindow()
ui.setupUi(window1)
window2 = PyQt5.QtWidgets.QMainWindow()
ui_summary = summary.Ui_MainWindow()
ui_summary.setupUi(window2)
window3 = PyQt5.QtWidgets.QMainWindow()
ui_information = information.Ui_MainWindow()
ui_information.setupUi(window3)
threadpool = PyQt5.QtCore.QThreadPool()
class_label = ['glioma', 'meningioma', 'no_tumor', 'pituitary', 'No Data']

current_dir = os.getcwd()
is_predicted = False
current_modelSelect=-1
img_num = 0
is_valid_clicked = False

# Slot
def get_fileDir():
    ui.label_res.setText("...")
    ui.label_res_2.setText("...")
    browse_dir()
    index_changed()

def get_fileClick():
    ui.label_res.setText("...")
    ui.label_res_2.setText("...")
    browse_img()
    index_changed()
    
def index_changed():
    global img_index
    img_index = ui.spinBox_imgIndex.value()
    set_img()
    img_pathLabel()
    set_indexButton()
    img_validRes()
    img_predictRes()

def index_incr():
    ui.spinBox_imgIndex.setValue(max(1, min(img_index + 1, img_num)))

def index_decr():
    ui.spinBox_imgIndex.setValue(max(1, min(img_index - 1, img_num)))

def model_changed():
    global model_path
    if ui.comboBox_modelSelect.currentIndex()==0:
        model_path = current_dir+'/model/model_Improved_Resnet50'
        if current_modelSelect == 0:
            ui.pushButton_modelSet.setDisabled(True)
        else :
            ui.pushButton_modelSet.setEnabled(True)
            
    if ui.comboBox_modelSelect.currentIndex()==1:
        model_path = current_dir+'/model/model_LU_Net'
        if current_modelSelect == 1:
            ui.pushButton_modelSet.setDisabled(True)
        else :
            ui.pushButton_modelSet.setEnabled(True)

def set_model():
    worker = process_thread.Worker(load_dlModel)
    worker.signal.started.connect(progressBar_loadModel)
    worker.signal.finished.connect(progressBar_stop)
    worker.signal.error.connect(error_handle)
    ui.pushButton_cancelProcess.clicked.connect(worker.stop)
    threadpool.start(worker)

def process_call():
    worker = process_thread.Worker(process)
    worker.signal.started.connect(progressBar_process)
    worker.signal.finished.connect(progressBar_stop)
    worker.signal.error.connect(error_handleProcess)
    ui.pushButton_cancelProcess.clicked.connect(worker.stop)
    threadpool.start(worker)

def validation_clicked():
    global is_valid_clicked
    if not is_valid_clicked:
        if any(all_dataframe['validation_res'].str.contains('No Data')) == True:
            box = PyQt5.QtWidgets.QMessageBox()
            box.setText("Warning")
            box.setInformativeText("Some data didn't have valid classification. For more info, you can go to information page")
            box.setWindowTitle("Warning")
            box.exec_()
        is_valid_clicked = True
    else :
        is_valid_clicked = False
    validation_state()

def save_call():
    global folder_path
    worker = process_thread.Worker(save)
    worker.signal.started.connect(progressBar_save)
    worker.signal.finished.connect(progressBar_stop)
    worker.signal.error.connect(error_handleProcess)
    ui.pushButton_cancelProcess.clicked.connect(worker.stop)

    folder_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(window1, 'Save Folder', current_dir+"/Result")
    threadpool.start(worker)

def summary_call():
    window2.show()
    fill_summary()
    fill_confussion_matrix()
    fill_performance()

def information_call():
    window3.show()

# Signal
ui.pushButton_getImg.clicked.connect(get_fileClick)
ui.pushButton_getDir.clicked.connect(get_fileDir)
ui.pushButton_Right.clicked.connect(index_incr)
ui.pushButton_Left.clicked.connect(index_decr)
ui.spinBox_imgIndex.valueChanged.connect(index_changed)
ui.comboBox_modelSelect.currentIndexChanged.connect(model_changed)
ui.pushButton_predict.clicked.connect(process_call)
ui.pushButton_modelSet.clicked.connect(set_model)
ui.pushButton_save.clicked.connect(save_call)
ui.pushButton_valid.clicked.connect(validation_clicked)
ui.pushButton_summary.clicked.connect(summary_call)
ui.pushButton_help.clicked.connect(information_call)

#Thread slot

def progressBar_stop():
    ui.label_processName.setText("State : Idle")
    ui.progressBar.setMaximum(1)
    button_idleState()
    button_state()
    index_changed()

def progressBar_loadModel():
    ui.label_processName.setText("Loading model")
    ui.progressBar.setMaximum(0)
    button_busyState()

def load_dlModel():
    global dl_model, model1, model2
    global current_modelSelect
    if ui.comboBox_modelSelect.currentIndex()==0:
        try :
            type(model1)
        except :
            model1 = tf.keras.models.load_model(model_path)                
        dl_model = model1
        current_modelSelect = 0
        ui.label_selectedModel.setText("Selected Model : Improved Resnet50")
        

    if ui.comboBox_modelSelect.currentIndex()==1:
        try :
            type(model2)
        except :
            model2 = tf.keras.models.load_model(model_path)
        dl_model = model2
        current_modelSelect = 1
        ui.label_selectedModel.setText("Selected Model : Lu-Net")

def progressBar_process():
    ui.label_processName.setText("Classifying image")
    ui.progressBar.setMaximum(0)
    button_busyState()

def process():
    global all_dataframe
    global is_predicted
    predict_label = []
    predict_label_belief = []
    for dir in all_dataframe['file_path'] :
        img = Image.open(dir).convert('L').resize((224, 224), resample=0)
        img_arr = (np.array(img))/255.0
        img_arr = np.stack([img_arr], axis=0)
        res_array = dl_model.predict(img_arr)
        predict_label.append(class_label[np.argmax(res_array)])
        predict_label_belief.append(round(np.max(res_array)*1000)/1000.0)
    all_dataframe.prediction_res= predict_label
    all_dataframe.prediction_belief = predict_label_belief

    all_dataframe.isTrue = np.where(all_dataframe['prediction_res']==all_dataframe['validation_res'], True, False)
    is_predicted = True

def progressBar_save():
    ui.label_processName.setText("Saving")
    ui.progressBar.setMaximum(0)
    button_busyState()

def save():
    global folder_path
    now = datetime.now()
    formatted_path = folder_path + "/Classification Result at " + str(now.strftime("%B %d, %Y %H-%M-%S"))
    os.mkdir(formatted_path)
    for i in range (len(all_dataframe['file_path'])):
        file_dest = os.path.join(formatted_path, all_dataframe['prediction_res'].values[i])
        if not os.path.exists(file_dest):
            os.makedirs(file_dest)
        shutil.copy2(all_dataframe['file_path'].values[i], file_dest)
        
    fill_confussion_matrix()
    fill_performance()
    all_dataframe.to_csv(formatted_path+'/result.csv', sep=';')
    confussion_dataframe.to_csv(formatted_path+'/confusion_matrix.csv', sep=';')
    perf_dataframe.to_csv(formatted_path+'/performance.csv', sep=';')

def error_boxShow():
    box = PyQt5.QtWidgets.QMessageBox()
    box.setText("Error")
    box.setInformativeText(error_msg)
    box.setWindowTitle("Error")
    box.exec_()

#GUI function
def set_img():
    if img_num>=1:
        img = PyQt5.QtGui.QPixmap(all_dataframe['file_path'].values[img_index-1])
        scene = PyQt5.QtWidgets.QGraphicsScene()
        scene.clear()
        scene.addItem(PyQt5.QtWidgets.QGraphicsPixmapItem(img))
        ui.graphicsView_mainImg.setScene(scene)

def img_pathLabel():
    if img_num>=1:
        ui.lineEdit_filePath.setText(all_dataframe['file_path'].values[img_index-1])

def img_predictRes():
    if img_num>=1:
        ui.label_res.setText(all_dataframe['prediction_res'].values[img_index-1])
        
def img_validRes():
    if img_num>=1:
        ui.label_res_2.setText(all_dataframe['validation_res'].values[img_index-1])

def set_indexButton():
    if img_index <= 1:
        ui.pushButton_Left.setDisabled(True)
    else :
        ui.pushButton_Left.setEnabled(True)

    if img_index >= img_num:
        ui.pushButton_Right.setDisabled(True)
    else :
        ui.pushButton_Right.setEnabled(True)

def button_state():
    if img_num == 0 or current_modelSelect == -1:
        ui.pushButton_predict.setDisabled(True)
    else :
        ui.pushButton_predict.setDisabled(False)

    if img_num == 0:
        ui.pushButton_summary.setDisabled(True)
        ui.pushButton_valid.setDisabled(True)
    else :
        ui.pushButton_summary.setDisabled(False)
        ui.pushButton_valid.setDisabled(False)

    if not is_predicted:
        ui.pushButton_save.setDisabled(True)
    else :
        ui.pushButton_save.setDisabled(False)

def button_busyState():
    ui.pushButton_getDir.setDisabled(True)
    ui.pushButton_getImg.setDisabled(True)    
    ui.pushButton_help.setDisabled(True)
    ui.pushButton_Left.setDisabled(True)
    ui.pushButton_modelSet.setDisabled(True)
    ui.pushButton_predict.setDisabled(True)
    ui.pushButton_Right.setDisabled(True)
    ui.pushButton_save.setDisabled(True)
    ui.pushButton_summary.setDisabled(True)
    ui.comboBox_modelSelect.setDisabled(True)
    ui.pushButton_valid.setDisabled(True)

    ui.pushButton_cancelProcess.setEnabled(True)

def button_idleState():
    ui.pushButton_getDir.setEnabled(True)
    ui.pushButton_getImg.setEnabled(True)    
    ui.pushButton_help.setEnabled(True)
    ui.pushButton_Left.setEnabled(True)
    ui.pushButton_predict.setEnabled(True)
    ui.pushButton_Right.setEnabled(True)
    ui.pushButton_save.setEnabled(True)
    ui.pushButton_summary.setEnabled(True)
    ui.comboBox_modelSelect.setEnabled(True)
    ui.pushButton_valid.setEnabled(True)

    ui.pushButton_cancelProcess.setDisabled(True)

def validation_state():
    if is_valid_clicked:
        ui.pushButton_valid.setGeometry(700, 380, 91, 31)
        ui.pushButton_valid.setText("Hide Validation")
        ui.label_5.show()
        ui.label_res_2.show()
    else :
        ui.pushButton_valid.setGeometry(560, 380, 91, 31)
        ui.pushButton_valid.setText("Validation")
        ui.label_5.hide()
        ui.label_res_2.hide()

def fill_summary():
    headers = list(all_dataframe)
    ui_summary.tableWidget_sum.setRowCount(all_dataframe.shape[0])
    ui_summary.tableWidget_sum.setColumnCount(all_dataframe.shape[1])
    ui_summary.tableWidget_sum.setHorizontalHeaderLabels(headers)

    content_array = all_dataframe.values
    for row in range(all_dataframe.shape[0]):
        for col in range(all_dataframe.shape[1]):
            ui_summary.tableWidget_sum.setItem(row, col, PyQt5.QtWidgets.QTableWidgetItem(str(content_array[row, col])))

def fill_confussion_matrix():
    global confussion_dataframe
    zero_list = [0, 0, 0, 0, 0]
    idx_label = ['glioma_valid', 'meningioma_valid', 'no tumor_valid', 'pituitary_valid', 'No Data_valid']
    data = {'glioma_pred':zero_list, 'meningioma_pred':zero_list, 'no tumor_pred':zero_list, 'pituitary_pred':zero_list, 'No Data_pred':zero_list}
    confussion_dataframe = pd.DataFrame(data, index=idx_label)

    for i in range(len(all_dataframe['file_path'])):
        idx_pred = class_label.index(all_dataframe['prediction_res'].values[i])
        idx_val = class_label.index(all_dataframe['validation_res'].values[i])

        confussion_dataframe.iat[idx_val, idx_pred] += 1 
    

    headers = list(confussion_dataframe)

    ui_summary.tableWidget_confMat.setRowCount(confussion_dataframe.shape[0])
    ui_summary.tableWidget_confMat.setColumnCount(confussion_dataframe.shape[1])
    ui_summary.tableWidget_confMat.setHorizontalHeaderLabels(headers)
    ui_summary.tableWidget_confMat.setVerticalHeaderLabels(idx_label)

    content_array = confussion_dataframe.values
    for row in range(confussion_dataframe.shape[0]):
        for col in range(confussion_dataframe.shape[1]):
            ui_summary.tableWidget_confMat.setItem(row, col, PyQt5.QtWidgets.QTableWidgetItem(str(content_array[row, col])))

def fill_performance():
    global perf_dataframe
    zero_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    index_perf = ['Accuracy', 'Misclassification', 'Precision', 'Recall', 'Specificity', 'F1-score']
    data = {'glioma':zero_list, 'meningioma':zero_list, 'no tumor':zero_list, 'pituitary':zero_list}
    perf_dataframe = pd.DataFrame(data, index=index_perf)
    for i in range(len(perf_dataframe.columns)):
        TP = float(confussion_dataframe.iat[i, i])
        FP = float(confussion_dataframe.iloc[:,i].sum()-TP)
        FN = float(confussion_dataframe.iloc[i].sum()-TP)
        TN = float(confussion_dataframe.values.sum()-TP-FP-FN)        
        
        perf_dataframe.iat[0, i] = round((TP+TN)/max((TP+FP+TN+FN),1.0)*1000)/1000.0
        perf_dataframe.iat[1, i] = round((FP+FN)/max((TP+FP+TN+FN),1.0)*1000)/1000.0
        perf_dataframe.iat[2, i] = round(TP/max((TP+FP),1.0)*1000)/1000.0
        perf_dataframe.iat[3, i] = round(TP/max((TP+FN),1.0)*1000)/1000.0
        perf_dataframe.iat[4, i] = round(TN/max((FP+TN),1.0)*1000)/1000.0
        perf_dataframe.iat[5, i] = round((TP+TP)/max((TP+TP+FP+FN),1.0)*1000)/1000.0
        

    headers = list(perf_dataframe)

    ui_summary.tableWidget_acc.setRowCount(perf_dataframe.shape[0])
    ui_summary.tableWidget_acc.setColumnCount(perf_dataframe.shape[1])
    ui_summary.tableWidget_acc.setHorizontalHeaderLabels(headers)
    ui_summary.tableWidget_acc.setVerticalHeaderLabels(index_perf)

    content_array = perf_dataframe.values
    for row in range(perf_dataframe.shape[0]):
        for col in range(perf_dataframe.shape[1]):
            ui_summary.tableWidget_acc.setItem(row, col, PyQt5.QtWidgets.QTableWidgetItem(str(content_array[row, col])))

def error_handleProcess(errorMessage):
    global current_modelSelect
    global error_msg
    error_msg = errorMessage
    error_boxShow()
    button_state()

def error_handle(errorMessage):
    global current_modelSelect
    global error_msg
    error_msg = errorMessage
    error_boxShow()
    current_modelSelect = -1  
    ui.label_selectedModel.setText("Selected Model : None")
    button_state()

#Program function
def browse_dir():
    global img_num
    global file_temp
    global filename
    global all_dataframe
    global is_predicted
    is_predicted = False
    
    full_path = []
    file_temp = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(window1, 'Open img', current_dir)
    for dirpath, _, filename in os.walk(file_temp):
        for name in filename:
            if '.jpg' or '.png' or '.jpeg' in name:
                full_path.append(os.path.join(dirpath, name))
    
    all_dataframe = pd.DataFrame()
    all_dataframe['file_path'] = full_path
    all_dataframe['validation_res'] = "No Data"
    all_dataframe['prediction_res'] = "No Data"
    all_dataframe['prediction_belief'] = np.NaN
    all_dataframe['isTrue'] = False
    img_num = all_dataframe['file_path'].count()
    ui.spinBox_imgIndex.setMaximum(max(img_num, 1))
    ui.label_imgImport.setText("{} image imported".format(img_num))
    get_valid_val()
    button_state()

def browse_img():
    global img_num
    global file_temp
    global filename
    global all_dataframe
    global is_predicted
    is_predicted = False
    file_temp = PyQt5.QtWidgets.QFileDialog.getOpenFileNames(window1, 'Open img', current_dir, "Image files (*.jpg *.gif *.png)")
    filename = file_temp[0]
    all_dataframe = pd.DataFrame()
    all_dataframe['file_path'] = filename
    all_dataframe['validation_res'] = "No Data"
    all_dataframe['prediction_res'] = "No Data"
    all_dataframe['prediction_belief'] = np.NaN
    all_dataframe['isTrue'] = False
    img_num = all_dataframe['file_path'].count()
    ui.spinBox_imgIndex.setMaximum(max(img_num, 1))
    ui.label_imgImport.setText("{} image imported".format(img_num))
    get_valid_val()
    button_state()
    
def get_valid_val():
    if img_num>=1:   
        last2path = all_dataframe['file_path'].str.split().str[-2:]
        valid_label = []
        for file_dir in last2path:
            file_dir = "".join(file_dir)
            if "glioma" in file_dir:
                valid_label.append("glioma")
            elif "meningioma" in file_dir:
                valid_label.append("meningioma")
            elif "no_tumor" in file_dir:
                valid_label.append("no_tumor")
            elif "pituitary" in file_dir:
                valid_label.append("pituitary")
            else :
                valid_label.append("No Data")
        all_dataframe.validation_res = valid_label

def init():
    ui.pushButton_cancelProcess.setDisabled(True)
    model_changed()
    validation_state()
    button_state()

if __name__ == "__main__":
    window1.show()
    init()
    sys.exit(app.exec_())

