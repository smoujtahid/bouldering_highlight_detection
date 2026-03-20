import os
import cv2
from moviepy import VideoFileClip
import math
import os


def split_video(input_path, output_dir, segment_duration_s=2, max_segments=20):
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
        new_clip = video.subclipped(start_time, end_time)

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



# --------------------------------------------------


if __name__ == "__main__":

    # - extract a few frames as images from a video
    # extract_frames("../data/Janja_Garnbret.mp4", "../data/extracted_images")

    # - split the video into segments
    split_video("../data/Janja_Garnbret.mp4", "../data/video_segments/", segment_duration_s=2, max_segments=20)

