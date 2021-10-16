import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import *
import torch
from torchvision import datasets, transforms
from PIL import Image
import PIL.ImageOps 
from PIL.ImageQt import ImageQt
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

# this is a custom class which handles the drawable canvas. code is largely sourced from https://www.mfitzp.com/tutorials/bitmap-graphics/?fbclid=IwAR0YMH_wldpyl8NFJJ7pkKXuEpEHhkeypR6ileJa22pyE0DuIhHyOCopCDI with a few minor tweaks.
class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        # setup the canvas pixmap
        pixmap = QtGui.QPixmap(560, 560)
        # fill the canvas with white colour
        pixmap.fill(QtGui.QColor("white"))
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(50)
        p.setColor(self.pen_color)
        p.setCapStyle(QtCore.Qt.RoundCap)
        p.setJoinStyle(Qt.RoundJoin)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # This initializes the model being used. If non existent, throws an error message.
        try:
            self.model = torch.load('my_mnist_model.pt')
        except:
            msg = QMessageBox()
            msg.setWindowTitle("ERROR")
            msg.setText("Please train a model before trying to recognize digits")
            x = msg.exec_()
            
        # Code for graph Below:
        # a figure instance to plot on
        
        self.figure = plt.figure(figsize=(2,5))
        self.prediction = " "
        
        #this is the Canvas Widget that displays the 'figure' different from the drawable canvas
        # it takes the 'figure' instance as a parameter to __init__
        self.canvasPlot = FigureCanvas(self.figure)
        
        # creates the figure plot/axis
        ax = self.figure.add_subplot(111)
        ax.barh((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),(0,0,0,0,0,0,0,0,0,0))
        ax.set_aspect(0.3)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(np.arange(10))
        ax.set_title('Class Probability')
        ax.set_xlim(0, 1.1)
        plt.tight_layout()

        self.figure.add_subplot(ax)
        
        # refresh canvas
        self.canvasPlot.draw()

        # Code for other features below: 
        # Setup the DRAWING Canvas
        self.canvas = Canvas()
        
        # Setup the layout of the application below
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        hor2 = QtWidgets.QHBoxLayout()
        l.addLayout(hor2)
        hor2.addWidget(self.canvas)
        
        vert = QtWidgets.QVBoxLayout()
        hor2.addLayout(vert)
        label = "Prediction: "
        
        # creates the prediction label in the top right of gui for displaying the predicted result
        self.classProbabilityLabel = QLabel(label)
        self.classProbabilityLabel.setStyleSheet("background-color: lightgreen; border: 2px solid black;")
        self.classProbabilityLabel.setFont(QFont('Arial',24))
        
        vert.addWidget(self.classProbabilityLabel)
        vert.addWidget(self.canvasPlot)

        buttonsLayout = QtWidgets.QHBoxLayout()
        l.addLayout(buttonsLayout)
        
        # creates the 3 buttons and adds them to the bottom of the gui
        clearBut = QPushButton("Clear")
        self.modelBut = StringBox()
        recognizeBut = QPushButton("Recognize")
        buttonsLayout.addWidget(recognizeBut)
        buttonsLayout.addWidget(clearBut)
        buttonsLayout.addWidget(self.modelBut)

        self.setCentralWidget(w)

        # creating menu bar
        mainMenu = self.menuBar()
  
        # creating "file" and "view" menus
        fileMenu = mainMenu.addMenu("File")
        viewMenu = mainMenu.addMenu("View")
    
        # creating clear action
        clearAction = QAction("Clear", self)
        # adding short cut to the clear action
        clearAction.setShortcut("Ctrl+C")
        # adding clear to the file menu
        fileMenu.addAction(clearAction)
        # connecting clear to file->clear and keyboard shortcut
        clearAction.triggered.connect(self.clear)
        # connecting clear to clear button
        clearBut.clicked.connect(self.clear)

        # creating recognize action
        recognizeAction = QAction("Recognize", self)
        # adding shortcut to the recognize action
        recognizeAction.setShortcut("Ctrl+R")
        # adding recognize to the file menu
        fileMenu.addAction(recognizeAction)
        # connecting regognzie to file->recognize and keyboard shortcut
        recognizeAction.triggered.connect(self.recognize)
        # connecting recognize to recognize button
        recognizeBut.clicked.connect(self.recognize)

        # creating training action and adding the option to the menus
        trainModelAction = QAction("Train Model", self)
        fileMenu.addAction(trainModelAction)
        trainModelAction.triggered.connect(self.trainTheModel)

        # Creating the action for viewing training and testing data sets
        viewTrainingImages = QAction("View training images", self)
        viewMenu.addAction(viewTrainingImages)
        viewTrainingImages.triggered.connect(self.trainingPopout)
        viewTestingImages = QAction("View testing images", self)
        viewMenu.addAction(viewTestingImages)
        viewTestingImages.triggered.connect(self.testPopout)

    def trainTheModel(self):

        # This trains the model by importing the script ai_with_class and running it. 
        msg = QMessageBox()
        msg.setStyleSheet("QLabel{min-width: 500px; min-height:100px}")
        msg.setText("Its training just hold on")
        x = msg.exec_()
        import ai_with_class
        self.model = torch.load('my_mnist_model.pt')

    def view_classify(self, ps):
        # sourced from https://github.com/amitrajitbose/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb
        # clearing old figure
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)
        ps = ps.data.numpy().squeeze()
        ax.barh((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), ps)
        ax.set_aspect(0.3)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(np.arange(10))
        ax.set_title('Class Probability')
        ax.set_xlim(0, 1.1)
        plt.tight_layout()
        self.figure.add_subplot(ax)
        
        # refresh canvas
        self.canvasPlot.draw()
        
    def recognize(self):
        # The try except statement below checks if there are any errors with recognizing the digit and if there is a model currently assigned.
        # If no model is assigned, or there is an error, an error message box will appear. 
        try: 
            # creates a buffer to save the canvas drawn image to
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)

            # saves the canvas drawn image to the buffer
            self.canvas.pixmap().save(buffer, "PNG")

            # opens the image saved in the buffer and converts it to grayscale
            pil_im = Image.open(io.BytesIO(buffer.data())).convert('L')

            # resizes the image to 20x20 pixels
            pil_im = pil_im.resize((20, 20))

            # creates a new white background thats 28x28 pixels grayscale and pastes the 20x20 canvas drawn image on top
            backgroundIm = Image.new('L', (28, 28), color= 'white')
            backgroundIm.paste(pil_im, (4, 4))

            # inverts the colours of the 28x28 image to match the MNIST dataset
            backgroundIm = PIL.ImageOps.invert(backgroundIm)

            # converts the image to a tensor and normalizes it
            img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])(backgroundIm)

            # this flattens the image to be ready for model prediction
            img = img.view(1, 784)
            
            #predicting the class
            if self.modelBut.value() == 0:
                with torch.no_grad():
                    logps = self.model(img)
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
            
            # displays the predicted probabilities on a horizontal bar graph
            self.view_classify(ps)

            # updates teh prediction label for the digits
            self.prediction = probab.index(max(probab))
            label = "Prediction: " + str(self.prediction) 
            self.classProbabilityLabel.setText(label)
            self.update()
        except:
            message = QMessageBox()
            message.setWindowTitle("ERROR")
            message.setText("No model trained, please train a model before trying to recognize digits")
            message.exec_() 
            
    def clear(self):
        # fills the drawing canvas with white to clear it
        self.canvas.pixmap().fill(QtGui.QColor("white"))
        self.canvas.update()
        
        # clearing old figure
        self.figure.clear()
         # create an axis
        ax = self.figure.add_subplot(111)
        ax.barh((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),(0,0,0,0,0,0,0,0,0,0))
        ax.set_aspect(0.3)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(np.arange(10))
        ax.set_title('Class Probability')
        ax.set_xlim(0, 1.1)
        plt.tight_layout()
        self.figure.add_subplot(ax)
        
        # refresh canvas
        self.canvasPlot.draw()
        self.classProbabilityLabel.setText("Prediction:  ")

        # this downloads and shows the 60000 MNIST data training images
    def trainingPopout(self):
        trainset = datasets.MNIST(root='../mnistdata/', download=True, train=True)
        result = ScrollMessageBox(trainset, "Training MNIST Dataset" , None)
        result.exec_()

        # this downloads and shows the 10000 MNIST data testing images
    def testPopout(self):
        valset = datasets.MNIST(root='../mnistdata/', download=True, train=False) 
        result = ScrollMessageBox(valset, "Testing MNIST Dataset" , None)
        result.exec_()

    # This is for future functionality that we didnt get around to doing. It is currently unused. 
    def downloadData(self):
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
        trainset = datasets.MNIST(root='../mnistdata/', download=True, train=True, transform=transform)
        valset = datasets.MNIST(root='../mnistdata/', download=True, train=False, transform=transform)

# This class is for the pop up window with a scroll bar used to display the MNIST data images
# sourced from https://stackoverflow.com/questions/47345776/pyqt5-how-to-add-a-scrollbar-to-a-qmessagebox
class ScrollMessageBox(QMessageBox):
   def __init__(self, l,title, *args, **kwargs):
      QMessageBox.__init__(self, *args, **kwargs)
      scroll = QScrollArea(self)
      scroll.setWidgetResizable(True)
      QMessageBox.setWindowTitle(self, title)
      self.content = QWidget()
      scroll.setWidget(self.content)
      lay = QVBoxLayout(self.content)
      for item in l:
        img = QLabel()
        qimage = ImageQt(item[0])
        img.setPixmap(QPixmap.fromImage(qimage))
        lay.addWidget(img)
      self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
      self.setStyleSheet("QScrollArea{min-width:300 px; min-height: 400px}")

# this class is used for the functionality to switch between models
class StringBox(QSpinBox):
    def __init__(self):
        super().__init__()

        strings = ["Model 1", "Model 2"]
        self.setStrings(strings)

    def setStrings(self, strings):
        strings = list(strings)
        # making tuple from the string list
        self._strings = tuple(strings)
        # creating a dictionary
        self._values = dict(zip(strings, range(len(strings))))
        # setting range to it the spin box
        self.setRange(0, len(strings)-1)
  
    # overwriting the textFromValue method
    def textFromValue(self, value):
        # returning string from index
        # _string = tuple
        return self._strings[value]

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()