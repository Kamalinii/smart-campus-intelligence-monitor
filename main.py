import cv2
import time
import math
from collections import defaultdict, deque

from ultralytics import solutions


# ============================================================
# VIDEO SOURCE
# ============================================================

SOURCE = "classroom_entry.mp4"


# ============================================================
# CAMERA-SPECIFIC CONFIGURATION
# ============================================================
#
# door_line is used internally for Entry / Exit.
# It is NOT displayed.
#
# reverse_direction fixes cameras where ObjectCounter's
# IN/OUT direction is opposite to the real room direction.
# ============================================================

CAMERA_CONFIGS = {

    "classroom_entry.mp4": {

        "door_line": [
            (0.18, 0.08),
            (0.18, 0.72)
        ],

        "reverse_direction": True
    },


    "video7.mp4": {

        "door_line": [
            (0.10, 0.20),
            (0.10, 0.75)
        ],

        "reverse_direction": False
    }
}


DEFAULT_CONFIG = {

    "door_line": [
        (0.15, 0.15),
        (0.15, 0.75)
    ],

    "reverse_direction": False
}


# ============================================================
# YOLO SETTINGS
# ============================================================

MODEL = "yolo11s.pt"

CONFIDENCE = 0.15

IMAGE_SIZE = 640

IOU = 0.5


# ============================================================
# MOTION SETTINGS
# ============================================================

MIN_MOTION_AREA = 3500

TOTAL_MOTION_AREA_THRESHOLD = 7000


# ============================================================
# FRAME QUALITY
# ============================================================

BLUR_THRESHOLD = 100


# ============================================================
# SEATED / STANDING-MOVING
# ============================================================

PERSON_MOVEMENT_THRESHOLD = 10

POSITION_HISTORY_LENGTH = 8

STANDING_ASPECT_RATIO = 1.55


# ============================================================
# ENTRY / EXIT MESSAGE
# ============================================================

EVENT_DISPLAY_DURATION = 35


# ============================================================
# PRESENT COUNT SMOOTHING
# ============================================================
#
# Current detections can fluctuate:
#
# 12 -> 11 -> 13 -> 10 -> 12
#
# because somebody may be temporarily occluded.
#
# We keep a short history and use the highest recent
# reliable detection count.
#
# This is STILL continuously updated.
# ============================================================

PRESENT_HISTORY_LENGTH = 12


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    SOURCE
)


if not cap.isOpened():

    print(
        f"ERROR: Could not open {SOURCE}"
    )

    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30


frame_delay = (
    1.0 / fps
)


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# ============================================================
# CURRENT CAMERA CONFIG
# ============================================================

camera_config = CAMERA_CONFIGS.get(
    SOURCE,
    DEFAULT_CONFIG
)


normalized_line = camera_config[
    "door_line"
]


REVERSE_ENTRY_EXIT = camera_config[
    "reverse_direction"
]


# ============================================================
# CONVERT NORMALIZED LINE TO PIXELS
# ============================================================

door_line = []


for x_percent, y_percent in normalized_line:

    x = int(
        x_percent
        *
        frame_width
    )


    y = int(
        y_percent
        *
        frame_height
    )


    door_line.append(
        (x, y)
    )


# ============================================================
# OBJECT COUNTER
# ============================================================

counter = solutions.ObjectCounter(

    model=MODEL,

    region=door_line,

    classes=[0],

    tracker="bytetrack.yaml",

    conf=CONFIDENCE,

    imgsz=IMAGE_SIZE,

    iou=IOU,

    show=False,

    show_in=False,

    show_out=False,

    show_conf=False,

    show_labels=False,

    verbose=False,

    line_width=1
)


# ============================================================
# VARIABLES
# ============================================================

previous_gray = None


position_history = defaultdict(

    lambda: deque(
        maxlen=POSITION_HISTORY_LENGTH
    )

)


# Short history for continuous attendance
present_history = deque(
    maxlen=PRESENT_HISTORY_LENGTH
)


currently_present = 0


# ============================================================
# ENTRY / EXIT VARIABLES
# ============================================================

previous_in = 0

previous_out = 0


current_in = 0

current_out = 0


raw_in = 0

raw_out = 0


entry_exit_status = (
    "No entry/exit"
)


event_display_frames = 0


# ============================================================
# UNIQUE ENTRY IDS
# ============================================================
#
# ObjectCounter provides cumulative IN count.
# We use corrected IN as total unique entries observed.
# ============================================================

unique_entries = 0


# ============================================================
# RESPONSIVE TEXT
# ============================================================

def draw_fitted_text(
    image,
    text,
    x,
    y,
    max_width,
    preferred_scale=0.50
):

    scale = preferred_scale


    while scale > 0.27:

        text_size, _ = cv2.getTextSize(

            text,

            cv2.FONT_HERSHEY_SIMPLEX,

            scale,

            1

        )


        if text_size[0] <= max_width:

            break


        scale -= 0.02


    cv2.putText(

        image,

        text,

        (x, y),

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        (255, 255, 255),

        1,

        cv2.LINE_AA

    )


# ============================================================
# START INFORMATION
# ============================================================

print()
print("======================================")
print("iCloudEMS Smart Campus")
print("======================================")

print(
    "Source:",
    SOURCE
)

print(
    "Resolution:",
    frame_width,
    "x",
    frame_height
)

print(
    "Continuous attendance: ENABLED"
)

print(
    "Reverse Entry/Exit:",
    REVERSE_ENTRY_EXIT
)

print()
print("Press Q to quit.")
print("======================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    loop_start = time.time()


    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "Video finished."
        )

        break


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )


    # ========================================================
    # MOTION DETECTION
    # ========================================================

    motion_gray = cv2.GaussianBlur(

        gray,

        (21, 21),

        0

    )


    motion_status = (
        "No motion"
    )


    if previous_gray is not None:

        difference = cv2.absdiff(

            previous_gray,

            motion_gray

        )


        _, threshold = cv2.threshold(

            difference,

            30,

            255,

            cv2.THRESH_BINARY

        )


        threshold = cv2.dilate(

            threshold,

            None,

            iterations=2

        )


        contours, _ = cv2.findContours(

            threshold,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )


        total_motion_area = 0


        for contour in contours:

            area = cv2.contourArea(
                contour
            )


            if area < MIN_MOTION_AREA:

                continue


            total_motion_area += area


        if (
            total_motion_area
            >
            TOTAL_MOTION_AREA_THRESHOLD
        ):

            motion_status = (
                "Motion detected"
            )


    previous_gray = motion_gray


    # ========================================================
    # FRAME QUALITY
    # ========================================================

    blur_score = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()


    if blur_score < BLUR_THRESHOLD:

        frame_quality = (
            "Blurry - flag for review"
        )

    else:

        frame_quality = (
            "Clear"
        )


    # ========================================================
    # YOLO + BYTETRACK + OBJECTCOUNTER
    # ========================================================

    count_results = counter(
        frame.copy()
    )


    # ========================================================
    # RAW IN / OUT
    # ========================================================

    raw_in = int(
        count_results.in_count
    )


    raw_out = int(
        count_results.out_count
    )


    # ========================================================
    # CORRECT ENTRY / EXIT DIRECTION
    # ========================================================

    if REVERSE_ENTRY_EXIT:

        current_in = raw_out

        current_out = raw_in


    else:

        current_in = raw_in

        current_out = raw_out


    # ========================================================
    # GET CURRENT TRACKS
    # ========================================================

    track_ids = getattr(
        counter,
        "track_ids",
        []
    )


    boxes = getattr(
        counter,
        "boxes",
        []
    )


    if track_ids is None:

        track_ids = []


    if boxes is None:

        boxes = []


    # ========================================================
    # CONVERT TRACK DATA
    # ========================================================

    try:

        if hasattr(
            track_ids,
            "cpu"
        ):

            ids_list = (
                track_ids
                .cpu()
                .tolist()
            )

        else:

            ids_list = list(
                track_ids
            )


        if hasattr(
            boxes,
            "cpu"
        ):

            boxes_list = (
                boxes
                .cpu()
                .tolist()
            )

        else:

            boxes_list = list(
                boxes
            )


    except Exception:

        ids_list = []

        boxes_list = []


    # ========================================================
    # CONTINUOUS CURRENT ATTENDANCE
    # ========================================================
    #
    # THIS IS THE IMPORTANT CHANGE.
    #
    # We no longer:
    #
    # initial_count + entries - exits
    #
    # Instead YOLO/ByteTrack continuously tells us how many
    # people are currently visible/tracked.
    # ========================================================

    visible_people = len(
        ids_list
    )


    # Save each new current-frame count
    present_history.append(
        visible_people
    )


    # ========================================================
    # SHORT-TERM STABILIZATION
    # ========================================================
    #
    # Take the maximum detection count seen over the recent
    # short window.
    #
    # Example:
    #
    # 13, 13, 12, 13, 11, 13
    #
    # currently_present = 13
    #
    # This prevents one momentary occlusion from instantly
    # dropping attendance.
    # ========================================================

    if len(present_history) > 0:

        currently_present = max(
            present_history
        )

    else:

        currently_present = (
            visible_people
        )


    # ========================================================
    # ENTRY EVENT
    # ========================================================

    if current_in > previous_in:

        entry_exit_status = (
            "Entry detected"
        )


        event_display_frames = (
            EVENT_DISPLAY_DURATION
        )


        print(
            "ENTRY DETECTED"
        )


    # ========================================================
    # EXIT EVENT
    # ========================================================

    if current_out > previous_out:

        entry_exit_status = (
            "Exit detected"
        )


        event_display_frames = (
            EVENT_DISPLAY_DURATION
        )


        print(
            "EXIT DETECTED"
        )


    previous_in = current_in

    previous_out = current_out


    # ========================================================
    # UNIQUE ENTRIES
    # ========================================================
    #
    # Corrected cumulative IN count.
    # ========================================================

    unique_entries = current_in


    # ========================================================
    # ENTRY / EXIT MESSAGE TIMER
    # ========================================================

    if event_display_frames > 0:

        event_display_frames -= 1

    else:

        entry_exit_status = (
            "No entry/exit"
        )


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    annotated_frame = (
        frame.copy()
    )


    # ========================================================
    # SEATED / STANDING-MOVING
    # ========================================================

    seated_count = 0

    standing_moving_count = 0


    # ========================================================
    # PROCESS PEOPLE
    # ========================================================

    for box, track_id in zip(
        boxes_list,
        ids_list
    ):

        track_id = int(
            track_id
        )


        x1 = int(
            box[0]
        )

        y1 = int(
            box[1]
        )

        x2 = int(
            box[2]
        )

        y2 = int(
            box[3]
        )


        box_width = (
            x2 - x1
        )


        box_height = (
            y2 - y1
        )


        if box_width <= 0:

            continue


        # ====================================================
        # CENTER
        # ====================================================

        center_x = (
            x1 + x2
        ) // 2


        center_y = (
            y1 + y2
        ) // 2


        # ====================================================
        # MOVEMENT HISTORY
        # ====================================================

        position_history[
            track_id
        ].append(
            (
                center_x,
                center_y
            )
        )


        history = (
            position_history[
                track_id
            ]
        )


        movement_distance = 0


        if len(history) >= 2:

            old_x, old_y = (
                history[0]
            )


            new_x, new_y = (
                history[-1]
            )


            movement_distance = math.sqrt(

                (
                    new_x
                    -
                    old_x
                ) ** 2

                +

                (
                    new_y
                    -
                    old_y
                ) ** 2

            )


        # ====================================================
        # ASPECT RATIO
        # ====================================================

        aspect_ratio = (

            box_height

            /

            box_width

        )


        # ====================================================
        # STANDING / MOVING
        # ====================================================

        if (
            movement_distance
            >
            PERSON_MOVEMENT_THRESHOLD
        ):

            person_state = (
                "Standing/Moving"
            )


            standing_moving_count += 1


            # ORANGE
            box_color = (
                0,
                165,
                255
            )


        elif (
            aspect_ratio
            >=
            STANDING_ASPECT_RATIO
        ):

            person_state = (
                "Standing/Moving"
            )


            standing_moving_count += 1


            # ORANGE
            box_color = (
                0,
                165,
                255
            )


        else:

            person_state = (
                "Seated"
            )


            seated_count += 1


            # GREEN
            box_color = (
                0,
                255,
                0
            )


        # ====================================================
        # PERSON BOX
        # ====================================================

        cv2.rectangle(

            annotated_frame,

            (x1, y1),

            (x2, y2),

            box_color,

            2

        )


        # ====================================================
        # PERSON LABEL
        # ====================================================

        label = (
            f"ID {track_id} - "
            f"{person_state}"
        )


        person_font = max(

            0.28,

            min(

                0.43,

                frame_width / 1800

            )

        )


        cv2.putText(

            annotated_frame,

            label,

            (
                x1,

                max(
                    y1 - 6,
                    15
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            person_font,

            box_color,

            1,

            cv2.LINE_AA

        )


    # ========================================================
    # ROOM STATUS
    # ========================================================
    #
    # This is also continuously determined.
    # ========================================================

    if currently_present > 0:

        room_status = (
            "Occupied"
        )

    else:

        room_status = (
            "Empty"
        )


    # ========================================================
    # TRANSPARENT DASHBOARD
    # ========================================================

    panel_height = max(

        120,

        int(
            frame_height
            *
            0.20
        )

    )


    panel_height = min(

        panel_height,

        int(
            frame_height
            *
            0.28
        )

    )


    panel_top = (

        frame_height

        -

        panel_height

    )


    overlay = (
        annotated_frame.copy()
    )


    cv2.rectangle(

        overlay,

        (
            0,
            panel_top
        ),

        (
            frame_width,
            frame_height
        ),

        (
            0,
            0,
            0
        ),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.30,

        annotated_frame,

        0.70,

        0,

        annotated_frame

    )


    # ========================================================
    # DASHBOARD POSITIONS
    # ========================================================

    left_x = int(
        frame_width
        *
        0.025
    )


    right_x = int(
        frame_width
        *
        0.52
    )


    left_max_width = int(
        frame_width
        *
        0.45
    )


    right_max_width = int(
        frame_width
        *
        0.45
    )


    row_gap = (
        panel_height
        /
        4
    )


    row1_y = int(
        panel_top
        +
        row_gap * 0.75
    )


    row2_y = int(
        panel_top
        +
        row_gap * 1.75
    )


    row3_y = int(
        panel_top
        +
        row_gap * 2.75
    )


    row4_y = int(
        panel_top
        +
        row_gap * 3.75
    )


    preferred_font = max(

        0.36,

        min(

            0.55,

            frame_width / 1350

        )

    )


    # ========================================================
    # 1. ENTRY / EXIT
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Entry/Exit: {entry_exit_status}",

        left_x,

        row1_y,

        left_max_width,

        preferred_font

    )


    # ========================================================
    # 2. MOTION
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Motion: {motion_status}",

        right_x,

        row1_y,

        right_max_width,

        preferred_font

    )


    # ========================================================
    # 3. CURRENTLY PRESENT
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"{currently_present} currently present",

        left_x,

        row2_y,

        left_max_width,

        preferred_font

    )


    # ========================================================
    # 4. UNIQUE ENTRIES
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"{unique_entries} unique entries",

        right_x,

        row2_y,

        right_max_width,

        preferred_font

    )


    # ========================================================
    # 5. SEATED
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Seated: {seated_count}",

        left_x,

        row3_y,

        left_max_width,

        preferred_font

    )


    # ========================================================
    # 5. STANDING / MOVING
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Standing/Moving: {standing_moving_count}",

        right_x,

        row3_y,

        right_max_width,

        preferred_font

    )


    # ========================================================
    # 6. QUALITY
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Quality: {frame_quality}",

        left_x,

        row4_y,

        left_max_width,

        preferred_font

    )


    # ========================================================
    # 7. ROOM STATUS
    # ========================================================

    draw_fitted_text(

        annotated_frame,

        f"Room: {room_status}",

        right_x,

        row4_y,

        right_max_width,

        preferred_font

    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "iCloudEMS Smart Campus",

        annotated_frame

    )


    # ========================================================
    # PLAYBACK TIMING
    # ========================================================

    processing_time = (

        time.time()

        -

        loop_start

    )


    remaining_time = (

        frame_delay

        -

        processing_time

    )


    if remaining_time > 0:

        time.sleep(
            remaining_time
        )


    # ========================================================
    # QUIT
    # ========================================================

    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("======================================")
print("Processing stopped")
print("======================================")

print(
    "Last detected present:",
    currently_present
)

print(
    "Unique entries:",
    unique_entries
)

print(
    "IN:",
    current_in
)

print(
    "OUT:",
    current_out
)

print("======================================")