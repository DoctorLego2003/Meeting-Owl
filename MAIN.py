import glob
import os
import numpy

import cv2
import time
import copy
import dlib

import yaml

import multiprocessing
from multiprocessing import Process, freeze_support
import face_recognition



# met deze file ander files callen zodat het wat overzichtelijker blijft

# ------face detection------#
from Programs.BetterTracking import *
# ------face detection------#


# ------lip detection------#
# distancevorige = 0
# from Programs.LipDetection import *
from Programs.LipDetectie2 import *
# ------lip detection------#


# ------face recognition------#
from Programs.face_recog_test_mp import *
# ------face recognition------#


# ------HandGestures------#
from Programs.gestures.HandGesture import *
# ------HandGestures------#



def load_YAML():
    # open YAML config file
    with open("MAIN_CONFIG", "r") as stream:
        try:
            # print(yaml.safe_load(stream))
            yamldata = yaml.safe_load(stream)
            YAML_DATA = copy.deepcopy(yamldata)
            yawn_thresh = YAML_DATA['yawn_threshold']
        except yaml.YAMLError as exc:
            print(exc)

        print(YAML_DATA)
        return YAML_DATA
        # for key in YAML_DATA:
        #     print(key, "=", YAML_DATA[key], "|", end=" ")
        # print("\n")

def display_FPS(ptime, frame):
    # frame = img
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 200, 0), 3)
    # cv2.putText(frame, f'FPS:{str(int(fps))}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # print(fps)
    return ptime

def load_encoding_images(images_path, fac):
    """
    Load encoding images from path
    :param images_path:
    :return:
    """
    # print("fac", fac)
    # dit zijn facenames en encodings die ingeladen zijn uit de map met images
    all_loaded_face_names = []
    new_face_encodings = []
    new_face_names = []
    to_be_removed_names = []

    # Load Images
    images_path = glob.glob(os.path.join(images_path, "*.*"))

    #print("{} encoding images found.".format(len(images_path)))

    # Store image encoding and names
    for img_path in images_path:

        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the filename only from the initial file path.
        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)

        all_loaded_face_names.append(filename)
        # print("fac", fac)
        # we hebben dus minstens 1 nieuwe naam gevonden -> naam en enc toevoegen aan resp list
        if filename not in fac:
            new_face_names.append(filename)
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            new_face_encodings.append(img_encoding)
            # print("new filename", filename)


    # checken of er namen verwijdert zijn
    for i, known_name in enumerate(fac):
        # print(known_name)
        if known_name not in all_loaded_face_names:
            # niet toevoegen aan lijst, rechtstreeks verwijderen uit die lijst
            to_be_removed_names.append(known_name)
            # fac.remove(known_name)
            # enc.remove(enc[i])


    # enc += new_face_encodings
    # fac += new_face_names
    # print(enc, fac)

    # return enc, fac
    return new_face_names, new_face_encodings, to_be_removed_names

def capture_vid(streamer):
    try:
        cap = cv2.VideoCapture(1)
        ret, img = cap.read()
        assert len(img) > 0
    except:
        cap = cv2.VideoCapture(0)
    # cap.release()
    # print("external_camera_bool", external_camera_bool)

    # for i in range(1,20):
    #     try:
    #         print(i)
    #         cap = cv2.VideoCapture(i)
    #         ret, img = cap.read()
    #         assert len(img) > 0
    #         print(len(img)>0)
    #         break
    #         # print(i)
    #     except:
    #     #     cap = cv2.VideoCapture(0)
    #         continue

    # if external_camera_bool == True:
    #     cap = cv2.VideoCapture(1)
    # else:
    #     cap = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture(1)

    video_capture = cap
    # video_capture = cv2.VideoCapture(0)

    print(video_capture)
    # print("frame")
    while True:
        ret, img = video_capture.read()
        if ret:
            streamer.send(img)

def face_reco(connectie, event, lock, stream, zoomed_reciever, zoomed_event):
    while True:
        lock.acquire()
        frame = stream.recv()
        lock.release()

        # test om shit te ontvangen met de pipe
        # test_data = testconn1reciever.recv()
        # print("test_data", test_data)
        #
        # zoomed = test_data
        # print("face_recog loop")

        zoomed_coords = []
        # zoomed = []

        # if testevent.is_set():
        # recieved_data = testconn1reciever.recv()
        # print("recieved_data", recieved_data)
        # testevent.set()

        zoomed_event.set()
        zoomed = zoomed_reciever.recv()
        # print("recieved_zoomed", zoomed)



        # print("face_reco")
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # face_locations = face_recognition.face_locations(rgb_frame)
        # face_locations2 = face_locations
        # print(face_locations)
        # gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print("zoomed", zoomed_coords)
        # print(zoomed)
        # main_tracking(frame, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model)
        for persoon in zoomed:
            # print("personen", persoon)
            zoomed_coords.append(tuple(persoon[0]))
            # print("zoomed2", zoomed_coords)

        face_locations2 = zoomed_coords
        # print("facelocations", face_locations)  # in (x1, y1, x2, y2)
        # print("facelocations2", face_locations2)  # in (x, y, w, h)
        # (x, y, w, h) = face_locations2[0]
        face_locations2 = [(y, h + x, w + y, x) for (x,y,w,h) in face_locations2]

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations2)

        # print("1", face_locations2)
        connectie.send(zip(face_locations2, face_encodings))
        # connectie.send(zip(zoomed_coords, face_encodings))  # hier behoudt die de [x,y,w,h] coord stijl
        # print("sent face enc")
        event.clear()
        event.wait()

def load_new_face_encodings(new_face_enc_sender, new_face_enc_event, YAML_DATA, enc_fac_reciever, enc_fac_event):
    wait_time = YAML_DATA['new_face_encoding_pull_time']
    while True:
        # if cv2.waitKey(2) & 0xff == ord('r'):
        print("tis gerefreshed")


        enc_fac_event.set()
        # [enc, fac] = list(enc_fac_reciever.recv())
        # dat = list(enc_fac_reciever.recv())  # lijst met 2 lijsten in
        # fac = [data[1] for data in dat]
        # enc = [data[0] for data in dat]
        # enc = dat[0]
        # fac = dat[1]

        fac = enc_fac_reciever.recv()

        # we willen enkel de nieuwe encodings inladen, en niet de oude opnieuw berekenen


        new_face_names, new_face_encodings, to_be_removed_names = load_encoding_images("Programs/images/", fac)
        # print(known_face_names)
        print("new_face_names", new_face_names)

        # time.sleep(wait_time)  # neemt als argument miliseconden


        # enkel nieuwe encodings en namen doorsturen OPGELET ER IS EEN LIMIET VAN HET AANTAL ENCODINGS DOOR DE PIPE
        new_face_enc_sender.send((new_face_names, new_face_encodings, to_be_removed_names))

        new_face_enc_event.clear()
        new_face_enc_event.wait()

def MAIN(YAML_DATA, ptime, known_face_encodings, known_face_names):
    streamer, stream = multiprocessing.Pipe()
    cam = Process(target=capture_vid, args=(streamer, ))
    cam.start()

    face_encodings = []
    # face_locations = []
    face_data = []
    # sent_zoomed = []

    fac = []
    enc = []

    lock = multiprocessing.RLock()
    event = multiprocessing.Event()
    event.set()

    new_face_enc_event = multiprocessing.Event()
    new_face_enc_event.set()

    # test, MOET ALTIJD RUNNEN, DUS NOOIT IN EEN IF STATEMENT ZETTEN
    zoomed_reciever, zoomed_sender = multiprocessing.Pipe()
    zoomed_event = multiprocessing.Event()
    zoomed_event.set()

    enc_fac_reciever, enc_fac_sender = multiprocessing.Pipe()
    enc_fac_event = multiprocessing.Event()
    enc_fac_event.set()


    if YAML_DATA['import_new_face_encodings'] == True:
        new_face_enc_reciever, new_face_enc_sender = multiprocessing.Pipe()
        # face_enc_loader = Process(target=load_new_face_encodings, args=(new_face_enc_sender, known_face_names) )
        face_enc_loader_process = Process(target=load_new_face_encodings, args=(new_face_enc_sender, new_face_enc_event, YAML_DATA, enc_fac_reciever, enc_fac_event))
        face_enc_loader_process.start()

    if YAML_DATA['display_face_recognition'] == True:
        face_data_reciever, face_data_sender = multiprocessing.Pipe()
        # print(face_data_reciever, face_data_sender)
        proces = Process(target=face_reco, args=(face_data_sender, event, lock, stream, zoomed_reciever, zoomed_event))
        proces.start()


    # lock.acquire()
    # frame = stream.recv()
    # # rgb_frame = frame[:, :, ::-1]
    # lock.release()


    # MAIN TRUE LOOP VAN HET PROGRAMMA
    while True:
        lock.acquire()
        frame = stream.recv()
        lock.release()

        # print("zoomed in main voor change", zoomed)


        # ------TRACKING------#
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        main_tracking(frame, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model, zoomed_sender)
        # ------TRACKING------#

        # zoomed has been changed
        # print("zoomed in main na change", zoomed)

        # IS NIET NODIG WANT DIE PAST GWN DE GLOBALE VAR AAN

        if YAML_DATA['import_new_face_encodings'] == True:
            if not new_face_enc_event.is_set():
                dat = list(new_face_enc_reciever.recv())
                # met dat dat de namen en
                new_face_enc_event.set()
                # print("dat", dat)
                # ALS HET HIER ERRORT, TEVEEL IMAGES INGELADEN
                nfnam, nfenc, tbrn = dat[0], dat[1], dat[2]

                removed_counter = 0
                tbrn.sort()
                for i, removed_name in enumerate(tbrn):
                    fac.remove(removed_name)
                    enc.remove(enc[i - removed_counter])
                    removed_counter += 1

                # print("fac 1", fac)
                fac += nfnam
                enc += nfenc
                # print("fac 2", fac)

                known_face_names = fac
                known_face_encodings = enc

                # checken of er namen verwijdert zijn
                # for i, known_name in enumerate(fac):
                #     # print(known_name)
                #     if known_name not in known_face_names:
                #         # niet toevoegen aan lijst, rechtstreeks verwijderen uit die lijst
                #         # removed_names.append(known_name)
                #         print("removing", known_name, enc[i])
                #         fac.remove(known_name)
                #         enc.remove(enc[i])
                #
                # fac += nfnam
                # enc += nfenc
                #
                # print("fac, enc", fac)

                # known_face_names = fac
                # known_face_encodings = enc

                # print(enc, fac)



        if YAML_DATA['display_face_recognition'] == True:
            if not event.is_set():
                face_data = list(face_data_reciever.recv())  # hier ontvangt die de locations van de gezichten en de encodings die hij gaat matchen aan shit

                event.set()
                # print("heeft facedetectie gedaan")
                # for data in face_data:
                #     print(data)
                #     face_encodings.append(list(data[1]))
                #     face_locations.append(list(data[0]))
                    # face_data = [data[0] for data in face_data]
                    # face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]

                # print(face_encodings)
                # print(face_locations)

                face_encodings = [data[1] for data in face_data]
                face_data = [data[0] for data in face_data]
                # print(face_locations, face_data)
                # print("1", face_data)
                # print(face_data)

                # face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]
                face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]
                # print("2", face_data)
                # print(face_data)


            name_coord_linking_list = []
            # face_names = []
            # print("len face_data_rec = ", len(face_data))
            for i, face_location in enumerate(face_data):
                # print(face_location)
                (x, y, w, h) = face_location  # in [x,y,w,h] coordinaten

                # HIER WORDEN MATCHES VERGELEKEN MET ELKAAR
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i])
                # name = "Unknown"
                name = str()
                # print(matches)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                name_coord_linking_list.append((face_location, name))


                # print(name_coord_linking_list)
                # print(name)

                if YAML_DATA['display_delayed_detection']:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            #print(name_coord_linking_list)

            for i in range(len(zoomed)):
                for j in range(len(name_coord_linking_list)):

                    # print(name_coord_linking_list[j][0])
                    # (face_location, name)
                    # print(name_coord_linking_list[j][0])
                    if zoomed[i][5] == list(name_coord_linking_list[j][0]):
                        zoomed[i][3] = name_coord_linking_list[j][1]

        # print("last_print", zoomed)


        # if cv2.waitKey(2) & 0xff == ord('r'):
        #     print("waiting for 10 seconds")
        #     time.sleep(10)


        if enc_fac_event.is_set() and cv2.waitKey(4) & 0xff == ord('r'):
            # enc = copy.deepcopy(known_face_encodings)
            # fac = copy.deepcopy(known_face_names)
            # enc_fac_sender.send((enc, fac))
            # enkel namen van huidige personen doorsturen



            fac = copy.deepcopy(known_face_names)
            enc = copy.deepcopy(known_face_encodings)
            enc_fac_sender.send(fac)
            enc_fac_event.clear()

        if zoomed_event.is_set():
            sent_zoomed = copy.deepcopy(zoomed)  # verzonden zoomed coordinaten
            zoomed_sender.send(sent_zoomed)
            zoomed_event.clear()

            # print("sent_zoomed", sent_zoomed)
            for i in range(len(zoomed)):
                zoomed[i][5] = sent_zoomed[i][0]  # dit add de fixed coodinates voordat de face_recognition gedaan wordt

            #
            # for i in range(len(zoomed)):
            #     zoomed[i][5] = zoomed[i][0]  # dit add de fixed coodinates voordat de face_recognition gedaan wordt

            # testevent.set()
        # testevent.clear()
        # testevent.wait()



        # ---------FPS------------#
        if YAML_DATA['display_FPS'] == True:
            # print("triggered")
            ptime_new = display_FPS(ptime, frame)
            ptime = ptime_new
        # ---------FPS------------#


        # ------Tracking------#
        # main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, distancevorige, face_model, landmark_model)
        # main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model)
        # ------Tracking------#


        cv2.imshow('Live: ', frame)

        # print("---------------")


        # k = cv2.waitKey(30) & 0xff
        # if k == ord('q'):
        #     break
        if cv2.waitKey(4) & 0xff == ord('q') or cv2.waitKey(4) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ptime = 0

    prev = []
    zoomed = []
    YAML_DATA = {}

    # xml files
    face_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier("Programs/xml/haarcascade_frontalface_alt2.xml")
    profile_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_profileface.xml')

    # face shit voor lip detection
    face_model = dlib.get_frontal_face_detector()
    landmark_model = dlib.shape_predictor('Programs/dat/shape_predictor_68_face_landmarks.dat')

    distancevorige = 0
    breedtemondvorige = 1
    zerocount = 0
    talklist = 0
    Talking = False

    known_face_encodings = []
    known_face_names = []



    YAML_DATA = load_YAML()


    # runnen omdat we sowieso 1 keer deze shit nodig hebben
    default_face_names, default_face_encodings, Default_to_be_removed_names = load_encoding_images("Programs/images/", known_face_names)
    known_face_names = default_face_names
    known_face_encodings = default_face_encodings
    # print(known_face_names)
    print("Encoding images loaded")



    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()
    # frame = img
    # external_camera_bool = False
    try:
        cap = cv2.VideoCapture(1)
        ret, img = cap.read()
        assert len(img) > 0
        # external_camera_bool = True
        # cv2.imshow('Live: ', img)
        # cv2.waitKey()
    except:
        cap = cv2.VideoCapture(0)
        # external_camera_bool = False
    cap.release()
    # print("external_camera_bool", external_camera_bool)

    # face_locations = face_recognition.face_locations(frame)
    # face_encodings = face_recognition.face_encodings(frame, face_locations)
    # cap.release()


    # print("run MAIN")
    MAIN(YAML_DATA, ptime, known_face_encodings, known_face_names)
