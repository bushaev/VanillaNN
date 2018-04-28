from tkinter import *
from tkinter.colorchooser import askcolor
from activations import Relu, Softmax
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
from network import ClassificationNetwork
from fully_connected import Dense
from utils import data, Scaler
import numpy as np
from cost_functions import NLL


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='Predict', command=self.predict)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='reset', command=self.clear)
        self.brush_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=50, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=2)

        self.c = Canvas(self.root, bg='white', width=200, height=200)
        self.c.grid(row=1, columnspan=3)

        self.setup()
        self.image = Image.new("RGB", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)

        # self.model = ClassificationNetwork([784, 100, 10], activation=[Relu, Softmax])
        self.model = ClassificationNetwork([
            Dense(784, 100, activation=Relu),
            Dense(100, 10, activation=Softmax),
        ], cost=NLL())
        X_tr, y_tr, _, _ = data('mnist')
        self.sc = Scaler(X_tr)
        X_tr = self.sc.transform(X_tr)
        self.model.optimize(X_tr, y_tr, lr=0.5, batch_size=64, nb_epoch=5)
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def predict(self):
        im = self.image.resize((28, 28)).convert('L')
        plt.imshow(im)
        img = np.asarray(im).reshape((1, -1))
        img = self.sc.transform(img)
        print ("You've drawn a ", self.model.predict(img))
        plt.show()

    def clear(self):
        self.c.delete('all')
        self.image = Image.new("RGB", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)


    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=(255, 255, 255), width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()