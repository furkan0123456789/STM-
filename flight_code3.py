from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
import math
import time
from threading import Thread
import queue

import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

liste = []

start_channel = "6"
M = 1
i = 5

#**********DEFINITIONS**********

connection_string = "/dev/ttyACM0"

point1 = LocationGlobalRelative(-35.363261, 149.1652299, 5) # Center (turkey)
point2 = LocationGlobalRelative(38.0255560, 32.5079850, 5)
point3 = LocationGlobalRelative(38.0254950, 32.5080450, 5)
point4 = LocationGlobalRelative(38.0255550, 32.5081150, 5)
point5 = LocationGlobalRelative(38.0256250, 32.5080350, 5)

#**********CONNECTION**********

print("Connecting to vehicle on:%s" % (connection_string))
vehicle = connect(connection_string, wait_ready=True)

print("mode : %s " % vehicle.mode.name)
print("Get some vehicle attribute values:")
print("GPS: %s" % vehicle.gps_0)
print("Battery: %s" % vehicle.battery)
print("Last Heartbeat: %s" % vehicle.last_heartbeat)
print("Is Armable?: %s" % vehicle.is_armable)
print("System status: %s" % vehicle.system_status.state)
print(vehicle.location.global_relative_frame)

#**********LOCATION DISTANCE**********

def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    return LocationGlobal(newlat, newlon, original_location.alt)

def get_distance_metres(aLocation1, aLocation2):
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

#**********ARM AND TAKE OFF**********

def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not (vehicle.is_armable):
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not (vehicle.armed):
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while (True):
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if (vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95):
            print("Reached target altitude")
            break
        time.sleep(1)

#**********GO TO**********

def goto(point, gotoFunction=vehicle.simple_goto):
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, point.lat, point.lon)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    # gotoFunction(targetLocation)
    vehicle.groundspeed = 1  # m/s
    vehicle.airspeed = 1
    print("Speed 1 m/s ")
    vehicle.simple_goto(point)  # ,  groundspeed = 1
    # print "DEBUG: targetLocation: %s" % targetLocation
    # print "DEBUG: targetLocation: %s" % targetDistance

    while (vehicle.mode.name == "GUIDED"):
        currentLocation = vehicle.location.global_relative_frame
        remainingDistance = get_distance_metres(currentLocation, point)
        print("Distance to target: ", remainingDistance)
        if (remainingDistance <= targetDistance * 0.05):
            print("Reached target")
            break
        time.sleep(2)

#**********LANDING**********

def landing():
    print("LAND Mode Active!Pls wait")
    vehicle.mode = VehicleMode("LAND")
    while (vehicle.armed):
        print("Waiting for landing and disarm!")
        time.sleep(1)
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
    print("Vehicle has landed.")

#*********MANUEL MODE CHECK**********

def Manual_Mode_Check():
    if vehicle.mode.name == "GUIDED":
        print("Manuel_Check_Failed")
    else:
        print("Manual Mode has been activated")
        exit()

#********** IMPORT TENSORFLOW MODEL **********

def Tensorflow_model():
    MODEL_NAME = 'inference_graph2'

    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
    PATH_TO_VIDEO = 0
    NUM_CLASSES = 5

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    return PATH_TO_VIDEO,category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

#********** OBJECT DETECTION **********

def object_detection(liste, PATH_TO_VIDEO, category_index, sess, out_queue, image_tensor, detection_boxes, detection_scores,
                     detection_classes, num_detections, confirm2, confirm = 0, id = None):

    t1_wait = time.time()
    video = cv2.VideoCapture(PATH_TO_VIDEO)

    while(video.isOpened()):
        t1 = time.time()
        print("*********************************************************")
        ret, original_frame = video.read()
        frame2 = original_frame.copy()

        frame_expanded = np.expand_dims(frame2, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame2,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)

        for i, value in enumerate(classes[0]):
            if scores[0, i] > 0.8:
                print("*********************************************************")
                print("Object{} : ".format(i + 1), value, category_index.get(value)["name"])

                liste.append(value)

                if len(liste) == 3:
                    print("Liste : ", liste)
                    liste_count = []
                    a = liste.count(2)      # stm
                    liste_count.append(a)
                    b = liste.count(3)      # metu
                    liste_count.append(b)
                    c = liste.count(4)      # heliport
                    liste_count.append(c)
                    d = liste.count(5)      # ort
                    liste_count.append(d)
                    print(liste_count)

                    maximum_iterations = max(liste_count)
                    print("maksimum_iterations : ", maximum_iterations)
                    index_value = liste_count.index(maximum_iterations)
                    print("index_value : ", index_value)

                    if (index_value == 0):
                        id = 2              # stm
                    elif (index_value == 1):
                        id = 3              # metu
                    elif (index_value == 2):
                        id = 4              # heliport
                    else:
                        id = 5              # ort

                    liste.clear()
                    print("****** CLEARED LIST ******")
                    confirm = 1

        #cv2.imshow("Original Frame", original_frame)
        #cv2.imshow("Object detector", frame2)

        t2 = time.time()
        FPS = (1/(t2-t1))
        print("FPS : ", FPS)

        t2_wait = time.time()
        Time = (t2_wait - t1_wait)

        if ((cv2.waitKey(1) == ord('q')) or (confirm == 1) or (confirm2 == 1) or Time > 30):
            print("****** BREAK ******")
            break

    video.release()
    cv2.destroyAllWindows()
    print("id : ", id)
    out_queue.put(id)

def stm(list_point_and_id, index_id):
    index = list_point_and_id.index(index_id)
    goto(list_point_and_id[index - 1], vehicle.simple_goto)
    landing()
    time.sleep(2)
    arm_and_takeoff(5)
    time.sleep(2)

#********** STARTING **********

def starts(liste, PATH_TO_VIDEO, category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):

    my_queue = queue.Queue()

    arm_and_takeoff(5)
    time.sleep(1)

    Manual_Mode_Check()

    print("Starting go to")

    print("Drone is going to point2")
    goto(point2, vehicle.simple_goto)
    time.sleep(1)

    t1 = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                        detection_classes, num_detections, confirm2 = 0))
    t1.start()
    t1.join()
    id1 = my_queue.get()
    print("id1 : ", id1)

    Manual_Mode_Check()

    print("Drone is going to point3")
    goto(point3, vehicle.simple_goto)
    time.sleep(1)

    t2 = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                        detection_classes, num_detections, confirm2 = 0))
    t2.start()
    t2.join()
    id2 = my_queue.get()
    print("id2 : ", id2)

    Manual_Mode_Check()

    print("Drone is going to point4")
    goto(point4, vehicle.simple_goto)
    time.sleep(1)

    t3 = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                        detection_classes, num_detections, confirm2 = 0))
    t3.start()
    t3.join()
    id3 = my_queue.get()
    print("id3 : ", id3)

    Manual_Mode_Check()

    print("Drone is going to point5")
    goto(point5, vehicle.simple_goto)
    time.sleep(1)

    t4 = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                        detection_classes, num_detections, confirm2 = 0))
    t4.start()
    t4.join()
    id4 = my_queue.get()
    print("id4 : ", id4)

    time.sleep(1)

    Manual_Mode_Check()
    list_point_and_id = [point2, id1, point3, id2, point4, id3, point5, id4]
    print("list_point_and_id : ", list_point_and_id)

    print("*** Checking ids ***")
    for i in range(1, 6, 2):
        for j in range(i + 2, 8, 2):
            print(i, j)
            if (list_point_and_id[i] == list_point_and_id[j]):
                print("Similar elements : {}, {}".format(list_point_and_id[i], list_point_and_id[j]))
                goto(list_point_and_id[i - 1], vehicle.simple_goto)
                time.sleep(1)

                t_try = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                            detection_classes, num_detections, confirm2=0))
                t_try.start()
                t_try.join()
                idx = my_queue.get()
                print("idx : ", idx)

                list_point_and_id[i] = idx

                k = list_point_and_id.count(2)
                l = list_point_and_id.count(3)
                m = list_point_and_id.count(4)
                n = list_point_and_id.count(5)

                if (k == 0):
                    list_point_and_id[j] = 2
                elif (l == 0):
                    list_point_and_id[j] = 3
                elif (m == 0):
                    list_point_and_id[j] = 4
                elif(n == 0):
                    list_point_and_id[j] = 5

                print("New list_point_and_id : ", list_point_and_id)

            else:
                print("All of the objects were detected.")

    value = list_point_and_id.count(1)
    if (value != 0):
        print("*** Id edited ***")
        index = list_point_and_id.index(1)

        if (list_point_and_id.count(2) == 0):
            list_point_and_id[index] = 2
        elif (list_point_and_id.count(3) == 0):
            list_point_and_id[index] = 3
        elif (list_point_and_id.count(4) == 0):
            list_point_and_id[index] = 4
        else:
            list_point_and_id[index] = 5

        print("New list_point_and_id : ", list_point_and_id)

    print("Drone is going to STM")
    stm(list_point_and_id, 2)
    Manual_Mode_Check()

    print("Drone is going to METU")
    stm(list_point_and_id, 3)
    Manual_Mode_Check()

    print("Drone is going to ORT")
    stm(list_point_and_id, 4)
    Manual_Mode_Check()

    print("Drone is going to HELİPORT")
    stm(list_point_and_id, 5)
    Manual_Mode_Check()

    print("Drone is going to TURKEY")
    goto(point1, vehicle.simple_goto)
    Manual_Mode_Check()
    landing()

PATH_TO_VIDEO, category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = Tensorflow_model()
print("MODEL INCLUDED")

my_queue = queue.Queue()
t_start = Thread(target=object_detection(liste, PATH_TO_VIDEO, category_index, sess, my_queue, image_tensor, detection_boxes, detection_scores,
                                         detection_classes, num_detections, confirm2=1))
t_start.start()
t_start.join()

while True:
    if (1300 <= vehicle.channels[start_channel] <= 1700):
        print("Channel 6 is activated")
        if (M is 1):
            while (i > 0):
                print("STARTING IN", i, " SEC")
                i = i - 1
                time.sleep(1)

            starts(liste, PATH_TO_VIDEO, category_index, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)

            M = 2
            time.sleep(1)
            print("Drone is waiting user interface reboot")
            time.sleep(1)

    else:
        M = 1
        print("Waiting for User Interface..!")
    print("Start : Channel 6")
    print(" GPS: %s" % vehicle.gps_0)

    if (1701 <= vehicle.channels[start_channel] < 2100):
        # **********CLOSE VEHICLE**********
        vehicle.close()
        break
    time.sleep(1)

