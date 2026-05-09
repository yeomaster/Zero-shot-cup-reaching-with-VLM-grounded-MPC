# camera.py
import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # --- Gets the camera's unique intrinsics ---
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Returns the raw depth_frame object too, because we need it for the 3D math
        return color_image, depth_image, depth_frame

    def get_aligned_frames(self):
        """Returns raw rs.frame objects (color_frame, depth_frame) — needed
        for things like rs.pointcloud.calculate() that operate on frames, not
        numpy arrays."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        return aligned.get_color_frame(), aligned.get_depth_frame()

    def get_intrinsics(self):
        return self.intrinsics

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass  # already stopped or never started — don't mask the real error