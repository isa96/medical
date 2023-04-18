import PyQt5
import tensorflow as tf
import traceback
import sys

class WorkerSignals(PyQt5.QtCore.QObject):
    finished = PyQt5.QtCore.pyqtSignal()
    started = PyQt5.QtCore.pyqtSignal()
    error = PyQt5.QtCore.pyqtSignal(str)
    processing = PyQt5.QtCore.pyqtSignal()
    
class Worker(PyQt5.QtCore.QRunnable):

    def __init__(self, func):
        super(Worker, self).__init__()
        self.func = func
        self.signal = WorkerSignals()

    def run(self):
        self.signal.started.emit()
        try :
            self.func()
        except :
            traceback.print_exc()
            self.signal.error.emit(traceback.format_exc())
        finally :
            self.signal.finished.emit()

    def stop(self):
        self.signal.finished.emit()

