import cv2

from . import settings, utils


def process_and_save_frame(frame, img_id, faces, output_dir, part_id, vid_name):
    processed_frame = blur_faces(frame, faces, settings.DEBUG)
    output_path = utils.make_out_name(output_dir, part_id, vid_name, img_id)
    cv2.imwrite(output_path, processed_frame)

    cv2.putText(
        processed_frame,
        f"Frame: {img_id:05}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )

    return (img_id, processed_frame)


def blur_faces(frame, faces, debug=False):
    if not len(faces):
        return frame

    frame_h, frame_w, _ = frame.shape

    for face in faces:
        x1, y1, x2, y2 = resize_box(face["bbox"], frame_h, frame_w)
        conf = face["det_score"]

        if debug:
            # Include the rect and conf
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{:.2f}".format(conf)
            # Display the label at the top of the bounding box
            label_size, base_line = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 2, 1
            )
            top = max(y2, label_size[1])
            cv2.putText(
                frame,
                text,
                (x1, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Add the blurring in
        roi = frame[y1:y2, x1:x2]

        try:
            # Blur the coloured image
            blur = cv2.GaussianBlur(roi, (101, 101), 0)
            # Insert the blurred section back into image
            frame[y1:y2, x1:x2] = blur
        except Exception as e:
            if debug:
                print(f"[ERROR] Blurring failed.")
                print(e)

    return frame


def resize_box(box, frame_h, frame_w):
    x1, y1, x2, y2 = (
        max(int(box[0]), 0),
        max(int(box[1]), 0),
        min(int(box[2]), frame_w),
        min(int(box[3]), frame_h),
    )

    return (x1, y1, x2, y2)
