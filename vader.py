import cv2
import numpy as np
import streamlit as st

st.title("Star Wars Day 2024")

scale_factor = st.slider("PNG scale factor", min_value=0.1, max_value=5.0)
st.text(f"Slider value is {scale_factor}")

# https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
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

# faceData = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# DOWNSCALE = 3
# UPSCALE_PNG = 1.2

# #OpenCV boiler plate
webcam = cv2.VideoCapture(0)
cv2.namedWindow("StarWars Face Recognation")

# #Loading vader_mask asset
# # vader_mask = cv2.imread('/images/Chewbacca.png', cv2.IMREAD_UNCHANGED)

# # ratio = glasses.shape[1] / glasses.shape[0]
# if webcam.isOpened(): # try to get the first frame
#     rval, frame = webcam.read()
# else:
#     rval = False

# #Main loop
# while rval:
#     # detect eyes and draw glasses
#     minisize = (int(frame.shape[1]/DOWNSCALE),int(frame.shape[0]/DOWNSCALE))
#     miniframe = cv2.resize(frame, minisize)
#     faces = faceData.detectMultiScale(miniframe)

#     # for face in faces:
#     #     x, y, w, h = [v * DOWNSCALE for v in face]

#     #     # resize vade mask to a new var called small_vader_mask
#     #     small_vader_mask = cv2.resize(vader_mask, (int(UPSCALE_PNG*w), int(UPSCALE_PNG*h)))
#     #     add_transparent_image(frame, small_vader_mask, max(0, x-int(abs(UPSCALE_PNG-1)/2*w)), max(0,y-int(abs(UPSCALE_PNG-1)/2*h)))

#     cv2.imshow("Webcam Glasses Tracking", frame)

#     # get next frame
#     rval, frame = webcam.read()

#     key = cv2.waitKey(20)
#     if key in [27, ord('Q'), ord('q')]: # exit on ESC
#         cv2.destroyWindow("Webcam Face Tracking")
#         break
