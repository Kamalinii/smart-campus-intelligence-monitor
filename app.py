import cv2
import time
import math
import tempfile
import threading
from pathlib import Path
from collections import defaultdict, deque

import av
import streamlit as st
from ultralytics import solutions
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Campus Monitor",
    page_icon="🎥",
    layout="wide"
)


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 14px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("Smart Campus Classroom Monitor")

st.caption(
    "Live classroom analytics using YOLO11, ByteTrack and OpenCV"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL = "yolo11s.pt"

CONFIDENCE = 0.15

IMAGE_SIZE = 960

IOU = 0.5

TARGET_FPS = 8


# Motion
MIN_MOTION_AREA = 3500
TOTAL_MOTION_AREA_THRESHOLD = 7000


# Blur
BLUR_THRESHOLD = 100


# Seated / Standing-Moving
PERSON_MOVEMENT_THRESHOLD = 10
POSITION_HISTORY_LENGTH = 8
STANDING_ASPECT_RATIO = 1.55


# Present count smoothing
PRESENT_HISTORY_LENGTH = 12


# Entry / Exit message
EVENT_DISPLAY_SECONDS = 2.0


# ============================================================
# CAMERA CONFIGURATIONS
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
# SHARED RESULT STORAGE
# ============================================================
#
# Webcam processing happens in a WebRTC callback.
# The callback and Streamlit UI therefore need shared state.
# ============================================================

class SharedResults:

    def __init__(self):

        self.lock = threading.Lock()

        self.currently_present = 0
        self.unique_entries = 0

        self.entry_exit = "No entry/exit"
        self.motion = "No motion"

        self.seated = 0
        self.standing_moving = 0

        self.quality = "Waiting"
        self.room = "Empty"


shared_results = SharedResults()


# ============================================================
# CV PROCESSOR
# ============================================================

class ClassroomProcessor:

    def __init__(
        self,
        width,
        height,
        camera_config
    ):

        self.width = width
        self.height = height

        self.camera_config = camera_config

        self.reverse_entry_exit = (
            camera_config[
                "reverse_direction"
            ]
        )


        # ----------------------------------------------------
        # Convert normalized entrance line to pixels
        # ----------------------------------------------------

        normalized_line = camera_config[
            "door_line"
        ]


        self.door_line = []


        for x_percent, y_percent in normalized_line:

            x = int(
                x_percent
                *
                width
            )

            y = int(
                y_percent
                *
                height
            )

            self.door_line.append(
                (x, y)
            )


        # ----------------------------------------------------
        # Object Counter
        # ----------------------------------------------------

        self.counter = solutions.ObjectCounter(

            model=MODEL,

            region=self.door_line,

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


        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.previous_gray = None


        self.position_history = defaultdict(

            lambda: deque(
                maxlen=POSITION_HISTORY_LENGTH
            )

        )


        self.present_history = deque(
            maxlen=PRESENT_HISTORY_LENGTH
        )


        self.previous_in = 0
        self.previous_out = 0

        self.current_in = 0
        self.current_out = 0

        self.last_event_time = 0

        self.entry_exit_status = (
            "No entry/exit"
        )


    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process(self, frame):

        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )


        # ====================================================
        # MOTION DETECTION
        # ====================================================

        motion_gray = cv2.GaussianBlur(

            gray,

            (21, 21),

            0

        )


        motion_status = (
            "No motion"
        )


        if self.previous_gray is not None:

            difference = cv2.absdiff(

                self.previous_gray,

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


        self.previous_gray = motion_gray


        # ====================================================
        # FRAME QUALITY
        # ====================================================

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


        # ====================================================
        # YOLO + BYTETRACK + OBJECT COUNTER
        # ====================================================

        count_results = self.counter(
            frame.copy()
        )


        raw_in = int(
            count_results.in_count
        )


        raw_out = int(
            count_results.out_count
        )


        # ====================================================
        # CORRECT CAMERA DIRECTION
        # ====================================================

        if self.reverse_entry_exit:

            self.current_in = raw_out
            self.current_out = raw_in

        else:

            self.current_in = raw_in
            self.current_out = raw_out


        # ====================================================
        # GET TRACKS
        # ====================================================

        track_ids = getattr(
            self.counter,
            "track_ids",
            []
        )


        boxes = getattr(
            self.counter,
            "boxes",
            []
        )


        if track_ids is None:

            track_ids = []


        if boxes is None:

            boxes = []


        # ====================================================
        # CONVERT TRACK DATA
        # ====================================================

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


        # ====================================================
        # CONTINUOUS ATTENDANCE
        # ====================================================

        visible_people = len(
            ids_list
        )


        self.present_history.append(
            visible_people
        )


        if len(
            self.present_history
        ) > 0:

            currently_present = max(
                self.present_history
            )

        else:

            currently_present = (
                visible_people
            )


        # ====================================================
        # ENTRY EVENT
        # ====================================================

        if (
            self.current_in
            >
            self.previous_in
        ):

            self.entry_exit_status = (
                "Entry detected"
            )

            self.last_event_time = (
                time.time()
            )


        # ====================================================
        # EXIT EVENT
        # ====================================================

        if (
            self.current_out
            >
            self.previous_out
        ):

            self.entry_exit_status = (
                "Exit detected"
            )

            self.last_event_time = (
                time.time()
            )


        self.previous_in = (
            self.current_in
        )

        self.previous_out = (
            self.current_out
        )


        # ====================================================
        # CLEAR EVENT MESSAGE
        # ====================================================

        if (
            time.time()
            -
            self.last_event_time
            >
            EVENT_DISPLAY_SECONDS
        ):

            self.entry_exit_status = (
                "No entry/exit"
            )


        # ====================================================
        # UNIQUE ENTRIES
        # ====================================================

        unique_entries = (
            self.current_in
        )


        # ====================================================
        # ANNOTATED FRAME
        # ====================================================

        annotated_frame = (
            frame.copy()
        )


        seated_count = 0
        standing_moving_count = 0


        # ====================================================
        # PROCESS EACH TRACKED PERSON
        # ====================================================

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


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2


            center_y = (
                y1 + y2
            ) // 2


            # ------------------------------------------------
            # POSITION HISTORY
            # ------------------------------------------------

            self.position_history[
                track_id
            ].append(
                (
                    center_x,
                    center_y
                )
            )


            history = (
                self.position_history[
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


            # ------------------------------------------------
            # ASPECT RATIO
            # ------------------------------------------------

            aspect_ratio = (

                box_height

                /

                box_width

            )


            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            if (
                movement_distance
                >
                PERSON_MOVEMENT_THRESHOLD
            ):

                person_state = (
                    "Standing/Moving"
                )

                standing_moving_count += 1

                # Orange
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

                # Orange
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

                # Green
                box_color = (
                    0,
                    255,
                    0
                )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(

                annotated_frame,

                (x1, y1),

                (x2, y2),

                box_color,

                2

            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"ID {track_id} - "
                f"{person_state}"
            )


            label_scale = max(

                0.32,

                min(

                    0.50,

                    self.width
                    /
                    1700

                )

            )


            cv2.putText(

                annotated_frame,

                label,

                (
                    x1,

                    max(
                        y1 - 7,
                        15
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                label_scale,

                box_color,

                1,

                cv2.LINE_AA

            )


        # ====================================================
        # ROOM OCCUPANCY
        # ====================================================

        if currently_present > 0:

            room_status = (
                "Occupied"
            )

        else:

            room_status = (
                "Empty"
            )


        # ====================================================
        # RESULTS
        # ====================================================

        results = {

            "currently_present":
                currently_present,

            "unique_entries":
                unique_entries,

            "entry_exit":
                self.entry_exit_status,

            "motion":
                motion_status,

            "seated":
                seated_count,

            "standing_moving":
                standing_moving_count,

            "quality":
                frame_quality,

            "room":
                room_status
        }


        return (
            annotated_frame,
            results
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Video Source"
)


source_type = st.sidebar.radio(

    "Select source",

    [
        "Project Video",
        "Upload Video",
        "Webcam"
    ]
)


st.sidebar.divider()


st.sidebar.caption(
    f"Processing target: {TARGET_FPS} FPS"
)

st.sidebar.caption(
    "Detector: YOLO11s"
)

st.sidebar.caption(
    "Tracker: ByteTrack"
)


# ============================================================
# DASHBOARD FUNCTION
# ============================================================

def create_dashboard():

    st.divider()


    c1, c2, c3, c4 = st.columns(
        4
    )


    present = c1.empty()
    entry = c2.empty()
    motion = c3.empty()
    unique = c4.empty()


    c5, c6, c7, c8 = st.columns(
        4
    )


    seated = c5.empty()
    standing = c6.empty()
    quality = c7.empty()
    room = c8.empty()


    return {

        "present": present,

        "entry": entry,

        "motion": motion,

        "unique": unique,

        "seated": seated,

        "standing": standing,

        "quality": quality,

        "room": room
    }


# ============================================================
# UPDATE DASHBOARD
# ============================================================

def update_dashboard(
    dashboard,
    results
):

    dashboard[
        "present"
    ].metric(

        "Currently Present",

        results[
            "currently_present"
        ]

    )


    dashboard[
        "entry"
    ].metric(

        "Entry / Exit",

        results[
            "entry_exit"
        ]

    )


    dashboard[
        "motion"
    ].metric(

        "Motion",

        results[
            "motion"
        ]

    )


    dashboard[
        "unique"
    ].metric(

        "Unique Entries",

        results[
            "unique_entries"
        ]

    )


    dashboard[
        "seated"
    ].metric(

        "Seated",

        results[
            "seated"
        ]

    )


    dashboard[
        "standing"
    ].metric(

        "Standing / Moving",

        results[
            "standing_moving"
        ]

    )


    dashboard[
        "quality"
    ].metric(

        "Frame Quality",

        results[
            "quality"
        ]

    )


    dashboard[
        "room"
    ].metric(

        "Room",

        results[
            "room"
        ]

    )


# ============================================================
# PROJECT VIDEO / UPLOADED VIDEO
# ============================================================

if source_type in [
    "Project Video",
    "Upload Video"
]:

    SOURCE = None
    source_name = None


    # ========================================================
    # PROJECT VIDEO
    # ========================================================

    if source_type == "Project Video":

        available_videos = []


        for video_name in [

            "classroom_entry.mp4",
            "video7.mp4"

        ]:

            if Path(
                video_name
            ).exists():

                available_videos.append(
                    video_name
                )


        if not available_videos:

            st.warning(
                "No project videos were found."
            )


        else:

            source_name = (
                st.sidebar.selectbox(

                    "Choose video",

                    available_videos

                )
            )


            SOURCE = (
                source_name
            )


    # ========================================================
    # UPLOAD VIDEO
    # ========================================================

    else:

        uploaded_file = (
            st.sidebar.file_uploader(

                "Upload classroom video",

                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ]

            )
        )


        if uploaded_file is not None:

            suffix = Path(
                uploaded_file.name
            ).suffix


            temp_file = (
                tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=suffix

                )
            )


            temp_file.write(
                uploaded_file.read()
            )


            temp_file.close()


            SOURCE = (
                temp_file.name
            )


            source_name = (
                uploaded_file.name
            )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    if source_name in CAMERA_CONFIGS:

        camera_config = (
            CAMERA_CONFIGS[
                source_name
            ]
        )

    else:

        camera_config = (
            DEFAULT_CONFIG
        )


    # ========================================================
    # UI
    # ========================================================

    st.subheader(
        "Classroom Feed"
    )


    video_placeholder = (
        st.empty()
    )


    dashboard = (
        create_dashboard()
    )


    initial_results = {

        "currently_present": 0,

        "unique_entries": 0,

        "entry_exit":
            "No entry/exit",

        "motion":
            "No motion",

        "seated": 0,

        "standing_moving": 0,

        "quality":
            "Waiting",

        "room":
            "Empty"
    }


    update_dashboard(
        dashboard,
        initial_results
    )


    # ========================================================
    # START BUTTON
    # ========================================================

    if SOURCE is not None:

        start_video = (
            st.sidebar.button(

                "Start Monitoring",

                type="primary",

                use_container_width=True

            )
        )


        if start_video:

            cap = cv2.VideoCapture(
                SOURCE
            )


            if not cap.isOpened():

                st.error(
                    "Could not open the video."
                )


            else:

                source_fps = cap.get(
                    cv2.CAP_PROP_FPS
                )


                if source_fps <= 0:

                    source_fps = 30


                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )


                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )


                # ============================================
                # PROCESS AT APPROXIMATELY 8 FPS
                # ============================================

                frame_skip = max(

                    1,

                    round(

                        source_fps

                        /

                        TARGET_FPS

                    )

                )


                processor = ClassroomProcessor(

                    width,

                    height,

                    camera_config

                )


                frame_number = 0


                while cap.isOpened():

                    ret, frame = (
                        cap.read()
                    )


                    if not ret:

                        break


                    frame_number += 1


                    # ----------------------------------------
                    # Sampling
                    # ----------------------------------------

                    if (
                        frame_number
                        %
                        frame_skip
                        != 0
                    ):

                        continue


                    start_time = (
                        time.time()
                    )


                    # ----------------------------------------
                    # Process one incoming frame
                    # ----------------------------------------

                    annotated_frame, results = (
                        processor.process(
                            frame
                        )
                    )


                    # ----------------------------------------
                    # BGR -> RGB
                    # ----------------------------------------

                    display_frame = (
                        cv2.cvtColor(

                            annotated_frame,

                            cv2.COLOR_BGR2RGB

                        )
                    )


                    # ----------------------------------------
                    # Update video
                    # ----------------------------------------

                    video_placeholder.image(

                        display_frame,

                        channels="RGB",

                        use_container_width=True

                    )


                    # ----------------------------------------
                    # Update dashboard
                    # ----------------------------------------

                    update_dashboard(

                        dashboard,

                        results

                    )


                    # ----------------------------------------
                    # Approximate target FPS
                    # ----------------------------------------

                    elapsed = (

                        time.time()

                        -

                        start_time

                    )


                    delay = (

                        1.0
                        /
                        TARGET_FPS

                    )


                    if elapsed < delay:

                        time.sleep(
                            delay - elapsed
                        )


                cap.release()


                st.success(
                    "Video processing completed."
                )


# ============================================================
# WEBCAM
# ============================================================

elif source_type == "Webcam":

    st.subheader(
        "Live Webcam"
    )


    st.info(
        "Click START below and allow camera access "
        "when your browser asks for permission."
    )


    # ========================================================
    # WEBCAM PROCESSOR STORAGE
    # ========================================================

    webcam_state = {

        "processor": None,

        "last_processed_time": 0
    }


    webcam_lock = (
        threading.Lock()
    )


    # ========================================================
    # VIDEO FRAME CALLBACK
    # ========================================================

    def webcam_callback(
        frame
    ):

        image = (
            frame.to_ndarray(
                format="bgr24"
            )
        )


        height, width = (
            image.shape[:2]
        )


        with webcam_lock:

            # -----------------------------------------------
            # Create processor on first webcam frame
            # -----------------------------------------------

            if (
                webcam_state[
                    "processor"
                ]
                is None
            ):

                webcam_state[
                    "processor"
                ] = ClassroomProcessor(

                    width,

                    height,

                    DEFAULT_CONFIG

                )


            processor = (
                webcam_state[
                    "processor"
                ]
            )


            # -----------------------------------------------
            # Limit expensive inference to approximately
            # TARGET_FPS.
            # -----------------------------------------------

            current_time = (
                time.time()
            )


            minimum_interval = (
                1.0
                /
                TARGET_FPS
            )


            if (
                current_time
                -
                webcam_state[
                    "last_processed_time"
                ]
                <
                minimum_interval
            ):

                return av.VideoFrame.from_ndarray(

                    image,

                    format="bgr24"

                )


            webcam_state[
                "last_processed_time"
            ] = current_time


            # -----------------------------------------------
            # Process live frame
            # -----------------------------------------------

            annotated_frame, results = (
                processor.process(
                    image
                )
            )


            # -----------------------------------------------
            # Save results for Streamlit dashboard
            # -----------------------------------------------

            with shared_results.lock:

                shared_results.currently_present = (
                    results[
                        "currently_present"
                    ]
                )

                shared_results.unique_entries = (
                    results[
                        "unique_entries"
                    ]
                )

                shared_results.entry_exit = (
                    results[
                        "entry_exit"
                    ]
                )

                shared_results.motion = (
                    results[
                        "motion"
                    ]
                )

                shared_results.seated = (
                    results[
                        "seated"
                    ]
                )

                shared_results.standing_moving = (
                    results[
                        "standing_moving"
                    ]
                )

                shared_results.quality = (
                    results[
                        "quality"
                    ]
                )

                shared_results.room = (
                    results[
                        "room"
                    ]
                )


            return av.VideoFrame.from_ndarray(

                annotated_frame,

                format="bgr24"

            )


    # ========================================================
    # WEBRTC STREAM
    # ========================================================

    webrtc_ctx = webrtc_streamer(

        key="smart-campus-webcam",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=webcam_callback,

        media_stream_constraints={

            "video": True,

            "audio": False

        },

        async_processing=True

    )


    # ========================================================
    # WEBCAM DASHBOARD
    # ========================================================

    st.divider()

    st.subheader(
        "Live Analytics"
    )


    st.caption(
        "The webcam video above is processed live. "
        "The analytics below update from the same stream."
    )


    # --------------------------------------------------------
    # Read latest webcam results
    # --------------------------------------------------------

    with shared_results.lock:

        webcam_present = (
            shared_results.currently_present
        )

        webcam_unique = (
            shared_results.unique_entries
        )

        webcam_entry = (
            shared_results.entry_exit
        )

        webcam_motion = (
            shared_results.motion
        )

        webcam_seated = (
            shared_results.seated
        )

        webcam_standing = (
            shared_results.standing_moving
        )

        webcam_quality = (
            shared_results.quality
        )

        webcam_room = (
            shared_results.room
        )


    # ========================================================
    # FIRST ROW
    # ========================================================

    w1, w2, w3, w4 = st.columns(
        4
    )


    w1.metric(
        "Currently Present",
        webcam_present
    )


    w2.metric(
        "Entry / Exit",
        webcam_entry
    )


    w3.metric(
        "Motion",
        webcam_motion
    )


    w4.metric(
        "Unique Entries",
        webcam_unique
    )


    # ========================================================
    # SECOND ROW
    # ========================================================

    w5, w6, w7, w8 = st.columns(
        4
    )


    w5.metric(
        "Seated",
        webcam_seated
    )


    w6.metric(
        "Standing / Moving",
        webcam_standing
    )


    w7.metric(
        "Frame Quality",
        webcam_quality
    )


    w8.metric(
        "Room",
        webcam_room
    )


    # ========================================================
    # IMPORTANT WEBCAM NOTE
    # ========================================================

    st.caption(
        "Entry/Exit uses the default entrance boundary for "
        "the webcam. For accurate doorway counting, the "
        "boundary should be configured for the camera's "
        "actual doorway position."
    )