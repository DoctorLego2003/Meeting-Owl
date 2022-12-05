import cv2
import numpy as np

def organise(show):
    if len(show) == 0:
        pass
    elif len(show) ==1:
        cv2.imshow('Zoomed', show[0])
    elif len(show) == 2:
        horizontal_stack = np.hstack((show[0], show[1]))
        cv2.imshow('Zoomed', horizontal_stack)
    elif len(show) == 3:
        horizontal_stack1 = np.hstack((show[0], show[1]))
        full_stack = np.hstack((horizontal_stack1, show[2]))
        cv2.imshow('Zoomed', full_stack)
    elif len(show) == 4:
        horizontal_stack1 = np.hstack((show[0], show[1]))
        horizontal_stack2 = np.hstack((show[2], show[3]))
        numpy_vertical = np.vstack((horizontal_stack1, horizontal_stack2))
        cv2.imshow('Zoomed',numpy_vertical)