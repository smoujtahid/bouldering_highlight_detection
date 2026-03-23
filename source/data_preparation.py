import os
import cv2
import math
import os
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def split_video(input_path, output_dir, segment_duration_s=2, max_segments=2):
    """
    split the input video into segments of segment_duration_s seconds
    Args:
        input_path: str, path to video
        output_dir: str, path to output directory
        segment_duration_s: float, segment duration
        max_segments: int, number of max segments to extract
    """
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
    Args:
        video_path: str, path to video
        output_dir: str, path to output directory
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


def create_text_clip(text, duration, color=(255, 255, 255), font_size=50):
    """
    create a clip with text for video inserts
    Args:
        text: str, text to insert
        duration: float, duration of clip in seconds
        color: tuple, color in RGB for text
        font_size: int, font size
    """
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
    """
    extract clips based on moments and combine them into a single video
    Args:
        video_path: str, path to original video
        moments: list, moments with start, end in seconds
        output_filename: str, output file for vide
        query: str, current query
    """
    video = VideoFileClip(video_path)
    duration_s = video.duration # in seconds
    clips = []
    intro_clip = create_text_clip(f"Query = {query}", 3)
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


def merge_moments(moments):
    """
    merge moments that intersect
    Args:
        moments: list, moments with start, end in seconds
    Returns:
        merged_moments: list, moments with start, end in seconds

    """
    if not moments:
        return []

    # Sort moments by start time
    sorted_moments = sorted(moments, key=lambda x: x[0])

    merged = [list(sorted_moments[0])]

    for current_start, current_end in sorted_moments[1:]:
        # Get the 'end' of the last moment we added to the merged list
        last_end = merged[-1][1]

        # If current start is <= last end, they overlap or touch
        if current_start <= last_end:
            # Update the end of the last moment to the maximum of both
            merged[-1][1] = max(last_end, current_end)
        else:
            # No overlap, add as a new moment
            merged.append([current_start, current_end])

    # Convert back to list of tuples
    return [tuple(m) for m in merged]



# --------------------------------------------------


if __name__ == "__main__":

    # - extract a few frames as images from a video
    # extract_frames("../data/Janja_Garnbret.mp4", "../data/extracted_images")

    # - split the video into segments
    split_video("../data/Janja_Garnbret.mp4", "../data/video_segments_350/", segment_duration_s=350, max_segments=2)

