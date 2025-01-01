import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def create_directory(path):
    """Utility function to create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_device_selection():
    """Prompt the user to select a device for processing."""
    print("Choose your device (enter the number):")
    print("1) CPU")
    print("2) GPU")
    print("3) MPS")
    device = input().strip()
    if device == "1":
        return "cpu"
    elif device == "2":
        return "gpu"
    elif device == "3":
        return "mps"
    else:
        raise ValueError("Invalid device selection")

def get_processing_mode():
    """Prompt the user to select the processing mode."""
    print("Choose your processing mode:")
    print("1) Process 1 video")
    print("2) Process 1 video from a specific frame")
    print("3) Process all videos")
    return input("Enter the number: ").strip()

def get_video_list(input_videos_dir, mode):
    """Get a list of videos to process based on the selected mode."""
    if mode == "1":
        video_name = input("Enter the name of the video file (e.g., video1.mp4): ").strip()
        return [video_name]
    elif mode == "2":
        video_name = input("Enter the name of the video file (e.g., video1.mp4): ").strip()
        starting_frame = int(input("Enter the starting frame number: ").strip())
        return [(video_name, starting_frame)]
    elif mode == "3":
        return [f for f in os.listdir(input_videos_dir) if f.lower().endswith((".mkv", ".mp4", ".avi", ".mov"))]
    else:
        raise ValueError("Invalid processing mode selected.")

def process_video(video_file, input_videos_dir, output_frames_dir, output_crops_dir, model, device, starting_frame=0):
    """Process a single video to extract frames and crop detected faces."""
    input_video_path = os.path.join(input_videos_dir, video_file)
    if not os.path.exists(input_video_path):
        print(f"Video file {video_file} not found. Skipping...")
        return

    video_name = os.path.splitext(video_file)[0]
    video_frames_dir = os.path.join(output_frames_dir, video_name)
    video_crops_dir = os.path.join(output_crops_dir, video_name)
    create_directory(video_frames_dir)
    create_directory(video_crops_dir)

    print(f"Processing video: {video_file}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {frame_rate}, Total Frames: {frame_count}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    frame_idx = starting_frame

    with tqdm(total=frame_count - starting_frame, desc=f"Processing frames in {video_file}", leave=False) as frame_pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_file = os.path.join(video_frames_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_file, frame)

            results = model.predict(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes

            if boxes is not None:
                for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                    box_size = max(x2 - x1, y2 - y1) * 2  # Ensure 2x size

                    # Ensure 1:1 aspect ratio with padding
                    x1_exp = x_center - box_size // 2
                    y1_exp = y_center - box_size // 2
                    x2_exp = x_center + box_size // 2
                    y2_exp = y_center + box_size // 2

                    # Calculate padding if the box exceeds frame boundaries
                    pad_top = max(0, -y1_exp)
                    pad_bottom = max(0, y2_exp - frame.shape[0])
                    pad_left = max(0, -x1_exp)
                    pad_right = max(0, x2_exp - frame.shape[1])

                    # Add padding to the frame
                    padded_frame = cv2.copyMakeBorder(
                        frame,
                        pad_top, pad_bottom, pad_left, pad_right,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)  # Black padding
                    )

                    # Adjust coordinates for padded frame
                    x1_exp += pad_left
                    x2_exp += pad_left
                    y1_exp += pad_top
                    y2_exp += pad_top

                    cropped_face = padded_frame[y1_exp:y2_exp, x1_exp:x2_exp]

                    # Adjust to enforce square aspect ratio after padding
                    cropped_height, cropped_width = cropped_face.shape[:2]
                    if cropped_height > cropped_width:
                        diff = cropped_height - cropped_width
                        pad_left = diff // 2
                        pad_right = diff - pad_left
                        cropped_face = cv2.copyMakeBorder(
                            cropped_face, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                        )
                    elif cropped_width > cropped_height:
                        diff = cropped_width - cropped_height
                        pad_top = diff // 2
                        pad_bottom = diff - pad_top
                        cropped_face = cv2.copyMakeBorder(
                            cropped_face, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                        )

                    crop_file = os.path.join(video_crops_dir, f"frame_{frame_idx:04d}_face_{i}.jpg")
                    cv2.imwrite(crop_file, cropped_face)

            frame_idx += 1
            frame_pbar.update(1)

    cap.release()
    print(f"Finished processing video: {video_file}")

def main():
    input_videos_dir = "videos/"
    output_frames_dir = "frames/"
    output_crops_dir = "crops/"

    create_directory(output_frames_dir)
    create_directory(output_crops_dir)

    device = get_device_selection()
    processing_mode = get_processing_mode()
    model = YOLO("./models/yolov8x6_animeface.pt")
    model.to(device)

    if processing_mode == "2":
        video_list = get_video_list(input_videos_dir, processing_mode)
        for video, start_frame in video_list:
            process_video(video, input_videos_dir, output_frames_dir, output_crops_dir, model, device, start_frame)
    else:
        video_list = get_video_list(input_videos_dir, processing_mode)
        for video in video_list:
            process_video(video, input_videos_dir, output_frames_dir, output_crops_dir, model, device)

    print("All tasks completed!")

if __name__ == "__main__":
    main()
