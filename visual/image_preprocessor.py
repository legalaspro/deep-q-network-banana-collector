import cv2
import numpy as np
from collections import deque


class DQNPreprocessor:
    def __init__(self, frame_height=84, frame_width=84, num_frames=4):
        """
        Initializes the frame stack and specifies frame dimensions and number of stacked frames.
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)  # Store the last `num_frames` frames

    def preprocess_frame(self, frame):
        """
        Process the raw input frame by converting to grayscale and resizing.
        """
        # Remove the singleton dimension (1, 84, 84, 3) -> (84, 84, 3)
        frame = frame.squeeze(0)

        # Convert to 8-bit if the frame is not already in that format
        if frame.dtype != np.uint8:
             frame = (frame*255).astype(np.uint8)  # Scale and convert to uint8 if needed

        # Convert frame to grayscale (using luminosity method)      
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(gray_frame)

        normalized_frame = enhanced_frame / 255.0
        
        # # Apply a slight Gaussian blur to reduce noise
        # blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        # # Normalize pixel values to be in range [0, 1]
        # normalized_frame = blurred_frame / 255.0

        return normalized_frame

    def get_state(self, frame):
        """
        Updates the frame stack with a new preprocessed frame and returns the stacked state.
        """
        processed_frame = self.preprocess_frame(frame)
        self.frames.append(processed_frame)  # Add new frame to the deque

        # Ensure stack has exactly `num_frames` frames; pad if necessary
        if len(self.frames) < self.num_frames:
            while len(self.frames) < self.num_frames:
                self.frames.append(processed_frame)

        # Stack frames along the first axis (to get a shape like [4, 84, 84])
        state = np.stack(self.frames, axis=0)
        return state
    
