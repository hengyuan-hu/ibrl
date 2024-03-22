from abc import abstractmethod
from typing import Optional, Union

import numpy as np

try:
    import cv2

    IMPORTED_CV2 = True
except ImportError:
    IMPORTED_CV2 = False

try:
    import pyrealsense2 as rs

    IMPORTED_PYREALSENSE = True
except ImportError:
    IMPORTED_PYREALSENSE = False


class Camera:
    def __init__(self, width: int, height: int, depth: bool):
        self.width = width
        self.height = height
        self.depth = depth

    @abstractmethod
    def get_frames(self) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def close(self):
        pass


class OpenCVCamera(Camera):
    def __init__(self, id: Optional[Union[int, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self._cap = None

    @property
    def cap(self):
        assert IMPORTED_CV2, "cv2 not imported."
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.id)  # type: ignore
            # self._cap.set(cv2.CAP_PROP_EXPOSURE, -2)
            # values other than default 640x480 have not been tested yet
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self._cap

    def get_frames(self):
        _, image = self.cap.read()
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"": image}

    def close(self):
        self.cap.release()


class RealSenseCamera(Camera):
    def __init__(self, serial_number: str, **kwargs):
        super().__init__(**kwargs)
        self.serial_number = str(serial_number)
        self._pipeline = None
        self.align = None
        self.depth_filters = None

        self.warned = False

    @property
    def has_depth(self):
        return self.depth

    @property
    def pipeline(self):
        if not IMPORTED_PYREALSENSE:
            if not self.warned:
                print("[Camera] Warning: No realsense, use mock camera instead")
                self.warned = True
            return

        assert IMPORTED_PYREALSENSE, "pyrealsense2 not installed."
        if self._pipeline is None:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            if self.depth:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)
                self.depth_filters = [rs.spatial_filter(), rs.temporal_filter()]
            self._pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            # warmup cameras
            for _ in range(2):
                self._pipeline.wait_for_frames()
        return self._pipeline

    def get_frames(self):
        if self.pipeline is None:
            image = np.random.random((self.width, self.height, 3))
            image = (image * 255).astype(np.uint8)
            return {"": image}

        assert self.align is not None
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        image = np.asanyarray(aligned_frames.get_color_frame().get_data())
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frames = {"": image}

        if self.depth:
            assert self.depth_filters is not None
            depth = aligned_frames.get_depth_frame()
            for rs_filter in self.depth_filters:
                depth = rs_filter(depth)
            depth = np.asanyarray(depth.get_data())
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth

        return frames

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()


if __name__ == "__main__":
    import cv2

    def show_image(images, wait_key=0):
        image = np.hstack(images)
        # convert the color because capture_rgb performed conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", image)
        cv2.waitKey(wait_key)

    cams = {
        # "in_hand": RealSenseCamera("241222076578", height=224, width=224, depth=False),
        # "left": RealSenseCamera("042222070680", height=224, width=224, depth=False),
        # "side-left": OpenCVCamera("/dev/video12", height=224, width=224, depth=False),
        # "side-right": OpenCVCamera("/dev/video14", height=224, width=224, depth=False),
        "front": RealSenseCamera("838212072814", height=224, width=224, depth=False),
        # "front": OpenCVCamera("/dev/video16", height=224, width=224, depth=False),
    }

    while True:
        images = []
        for name, cam in cams.items():
            rgb_image = cam.get_frames()[""]
            if name == "front":
                rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            images.append(rgb_image)

        show_image(images, 1)
