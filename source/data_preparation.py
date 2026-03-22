import os
import cv2
import math
import os
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def split_video(input_path, output_dir, segment_duration_s=2, max_segments=2):
    """split the input video into segments of segment_duration_s seconds"""
    # - load video
    video = VideoFileClip(input_path)
    duration_s = video.duration # in seconds
    print(f" Splitting video {input_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # - number of segments
    num_segments = math.ceil(duration_s / segment_duration_s)
    if num_segments < max_segments:
        max_segments = num_segments
    print(f" - {max_segments} segments of duration  {segment_duration_s}")

    # - for each segment
    for i in range(max_segments):
        start_time = i * segment_duration_s
        end_time = min((i + 1) * segment_duration_s, duration_s) # make sure it doesn't exceed video length

        # split clip
        new_clip = video.subclip(start_time, end_time)

        # save it to output dir
        output_filename = f"{output_dir}/segment_{i+1}.mp4"
        new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    # Close the video reader to free up memory
    video.close()
    print("Splitting complete!")


def extract_frames(video_path, output_dir):
    """
    Reads a video file and saves each frame as a JPEG image.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")

    # Initialize the video capture object
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    count = 0
    success = True

    print("Extraction started...")

    while success:
        # Read the next frame
        success, frame = video.read()

        if success:
            count += 1
            if count % 100 == 0:
                # Save frame every 100 frames
                frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Extracted {count} frames...")

    # Close the video file
    video.release()
    print(f"Done! {count} images saved to '{output_dir}'.")


def create_text_clip(text, duration, color=(255, 255, 255), font_size=60):
    # Create a transparent image
    img = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Position text in center
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((1280-w)/2, (720-h)/2), text, font=font, fill=color)

    # Convert PIL image to a format MoviePy understands (numpy array)
    img_array = np.array(img.convert('RGB'))
    return ImageClip(img_array).set_duration(duration)


def extract_and_combine(video_path, moments, output_filename, query):
    video = VideoFileClip(video_path)
    duration_s = video.duration # in seconds
    clips = []
    intro_clip = create_text_clip(f"Query {query}", 3)
    clips.append(intro_clip)

    for start, end in moments:
        # Extract the subclip
        if start < duration_s and end <= duration_s:
            # create an insert image
            insert_clip = create_text_clip(f"Moment {start} - {end} s", 3)
            clips.append(insert_clip)

            # add clip
            clip = video.subclip(start, end)
            clips.append(clip)
        else:
            print(f"WArning, moment {start} - {end} outside of video duration {duration_s}s")

    # Stitch all clips together
    final_video = concatenate_videoclips(clips)

    # Write to disk  
    final_video.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    # Clean up memory
    video.close()



# --------------------------------------------------


if __name__ == "__main__":

    # - extract a few frames as images from a video
    # extract_frames("../data/Janja_Garnbret.mp4", "../data/extracted_images")

    # - split the video into segments
    split_video("../data/Janja_Garnbret.mp4", "../data/video_segments_350/", segment_duration_s=350, max_segments=2)

