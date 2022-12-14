import cv2
import numpy as np

#def organise(show):
#    if len(show) == 0:
#        pass
#    elif len(show) == 1:
#        cv2.imshow('Zoomed', show[0])
#    elif len(show) == 2:
#        horizontal_stack = np.hstack((show[0], show[1]))
#        cv2.imshow('Zoomed', horizontal_stack)
#    elif len(show) == 3:
#        horizontal_stack1 = np.hstack((show[0], show[1]))
#        full_stack = np.hstack((horizontal_stack1, show[2]))
#        cv2.imshow('Zoomed', full_stack)
#    elif len(show) == 4:
#        horizontal_stack1 = np.hstack((show[0], show[1]))
#        horizontal_stack2 = np.hstack((show[2], show[3]))
#        numpy_vertical = np.vstack((horizontal_stack1, horizontal_stack2))
#        cv2.imshow('Zoomed',numpy_vertical)

def organise(show):
    amount_tabs = len(show)
    vorige = []
    if amount_tabs == 0:
        if cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow('Main')
        return None
    get = roostergetal(show)
    get2 = amount_tabs//get
    if amount_tabs % get != 0:
        get2 += 1
    for j in range(get2):
        new = show[get * j]
        for i in range(1, get):
            if i + get*j >= amount_tabs:
                black = np.zeros((400, 400, 3), np.uint8)
                black[:,:] = (0,0,0)
                new = np.hstack((new, black))
            else:
                new = np.hstack((new, show[i + get * j]))
        if len(vorige) == 0:
            vorige = new
        else:
            vorige = np.vstack((vorige, new))
    cv2.imshow('Main', vorige)


def roostergetal(show):
    getal = len(show)
    i = 0
    while i**2 < getal:
        i += 1
    return i
