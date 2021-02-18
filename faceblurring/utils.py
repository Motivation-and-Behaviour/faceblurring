import numpy as np
import cv2
from faceblurring.settings import *


def draw_blur(
    frame, conf, left, top, right, bottom, incl_box=INCL_BOX, incl_conf=INCL_CONF,
):

    if incl_box:
        # Draw a rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    if incl_conf:
        # Add confidence to image
        text = "{:.2f}".format(conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, label_size[1])
        cv2.putText(
            frame,
            text,
            (left, top - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    # Add the blurring in
    roi = frame[top:bottom, left:right]

    # Blur the coloured image
    blur = cv2.GaussianBlur(roi, (101, 101), 0)

    # Insert the blurred section back into image
    frame[top:bottom, left:right] = blur


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = (
        left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1
    )

    right = right + margin

    return left, top, right, bottom

def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        draw_blur(frame, confidences[i], left, top, right, bottom)
    return final_boxes

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]