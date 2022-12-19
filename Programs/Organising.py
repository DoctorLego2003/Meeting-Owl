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

def organise(bigshow, show):
    black = np.zeros((400, 400, 3), np.uint8)
    black[:, :] = (0, 0, 0)

    black2 = np.zeros((200, 200, 3), np.uint8)
    black2[:, :] = (0, 0, 0)

    amount_big_tabs = len(bigshow)
    amount_small_tabs = len(show)
    vorige = []

    if amount_big_tabs == 0 and amount_small_tabs == 0:
        if cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow('Main')
        return None
    get = roostergetal(bigshow)
    if get != 0:
        get2 = amount_big_tabs//get
        if amount_big_tabs % get != 0:
            get2 += 1
        for j in range(get2):
            new = bigshow[get * j]
            for i in range(1, get):
                if i + get*j >= amount_big_tabs:
                    new = np.hstack((new, black))
                else:
                    new = np.hstack((new, bigshow[i + get * j]))
            if len(vorige) == 0:
                vorige = new
            else:
                vorige = np.vstack((vorige, new))

        xdim, ydim = klein_roostergetal(show, get, get2)
        vorige2 = []
        for j in range(xdim):
            new2 = show[xdim * j]
            for i in range(1, 2 * get2):
                if i + xdim * j >= amount_small_tabs:
                    new2 = np.vstack((new2, black2))
                else:
                    new2 = np.vstack((new2, show[i + 2 * get * j]))
            if len(vorige2) == 0:
                vorige2 = new2
            else:
                vorige2 = np.hstack((vorige2, new2))
        if len(vorige2) != 0:
            vorige = np.hstack((vorige, vorige2))

        vorige2 = []
        for j in range(ydim):
            new2 = show[2 * xdim * get2 + (2 * get + xdim) * j]
            for i in range(1, 2 * get + xdim):
                if i + 2 * xdim * get2 + (2 * get + xdim) * j >= amount_small_tabs:
                    new2 = np.hstack((new2, black2))
                else:
                    new2 = np.hstack((new2, show[2 * xdim * get2 + i + (2 * get + xdim) * j]))
            if len(vorige2) == 0:
                vorige2 = new2
            else:
                vorige2 = np.vstack((vorige2, new2))

        if len(vorige2) != 0:
            vorige = np.vstack((vorige, vorige2))
    else:
        get = roostergetal(show)
        get2 = amount_small_tabs // get
        if amount_small_tabs % get != 0:
            get2 += 1
        for j in range(get2):
            new = show[get * j]
            for i in range(1, get):
                if i + get * j >= amount_small_tabs:
                    new = np.hstack((new, black2))
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

def klein_roostergetal(show, get, get2):
    xdim = 0
    ydim = 0
    x = len(show)
    if x == 0:
        return 0, 0
    while x > 0:
        x -= 2 * get2 + ydim
        xdim += 1
        if x > 0:
            x -= 2* get + xdim
            ydim += 1
    return xdim, ydim