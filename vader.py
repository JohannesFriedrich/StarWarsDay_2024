import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import urllib.request


def load_image_from_URL(url: str): 
    req = urllib.request.urlopen(url)
    encoded = np.asarray(bytearray(req.read()), dtype="uint8")
    image_bgra = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return image_bgra

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    # https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background

def video_frame_callback(frame):
        
    if selected_mask == "None":
        return frame
    else:
        img = frame.to_ndarray(format="bgr24")
        faces = faceData.detectMultiScale(img)
        mask = masks[selected_mask]

        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0))
            # resize mask to a new var called small_mask
            small_mask = cv2.resize(mask, (int(mask_scale_factor*w), int(mask_scale_factor*h)))
            add_transparent_image(
                img, 
                small_mask, 
                max(0, x-int(abs(mask_scale_factor-1)/2*w)), 
                max(0, y-int(abs(mask_scale_factor-1)/2*h))
            )
        if x_wing:
            add_transparent_image(
                img, 
                masks["x_wing"]
            )
        if millenium_falcon:
            add_transparent_image(
                img, 
                masks["millenium_falcon"]
            )

        # rebuild a VideoFrame, preserving timing information
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

faceData = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# #Loading vader_mask asset
masks = {
    "Darth Vader"      : load_image_from_URL("https://www.pngall.com/wp-content/uploads/9/Darth-Vader-Mask-PNG-High-Quality-Image.png"),
    "Chewbacca"        : load_image_from_URL("https://www.pngall.com/wp-content/uploads/9/Chewbacca-Face-PNG-Clipart.png"),
    "Storm Trooper"    : load_image_from_URL("https://www.pngall.com/wp-content/uploads/13/Stormtrooper-Imperial-PNG-Photo.png"),
    "Yoda"             : load_image_from_URL("https://pngimg.com/d/yoda_PNG60.png"),
    "Mandalorian"      : load_image_from_URL("https://pngimg.com/uploads/mandalorian/mandalorian_PNG19.png"),
    "x_wing"           : load_image_from_URL("https://www.pngall.com/wp-content/uploads/2016/03/Star-Wars-Ship-PNG.png"),
    "millenium_falcon" : load_image_from_URL("https://www.pngall.com/wp-content/uploads/2016/03/Star-Wars-Ship-Vector-PNG.png")
}

## UI -----
st.title("Star Wars Day 2024")

webrtc_streamer(
    key="StarWarsDay_2024",
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={ 
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_frame_callback=video_frame_callback
    )

st.sidebar.header("StarWars Day 2024")
selected_mask     = st.sidebar.radio("Mask", ("None", "Darth Vader", "Chewbacca", "Storm Trooper", "Yoda", "Mandalorian"), index = 0)
mask_scale_factor = st.sidebar.slider("Mask scale factor", min_value=0.7, max_value=2.0,  value=1.5)
# mask_x_offset     = st.sidebar.slider("Mask x-offset (left/right)", min_value=-50, max_value=50, value=0)
# mask_y_offset     = st.sidebar.slider("Mask y-offset (top/down)", min_value=-50, max_value=50, value=0)
x_wing            = st.sidebar.checkbox("Add X-Wing")
millenium_falcon  = st.sidebar.checkbox("Add Millenium Falcon")

