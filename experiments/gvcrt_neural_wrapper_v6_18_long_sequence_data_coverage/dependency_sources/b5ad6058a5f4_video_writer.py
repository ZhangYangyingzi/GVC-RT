
import os
import subprocess

from PIL import Image


class PNGWriter():
    def __init__(self, dst_path, width, height, qp=None):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.qp = qp
        self.padding = 5
        self.current_frame_index = 0
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        # rgb: 3xhxw uint8 numpy array
        rgb = rgb.transpose(1, 2, 0)

        frame_idx = self.current_frame_index
        if self.qp is not None:
            png_path = os.path.join(self.dst_path,
                                    f"qp{self.qp}_frame{frame_idx:04d}.png"
                                    )
        else:
            png_path = os.path.join(self.dst_path,
                                    f"im{str(frame_idx + 1).zfill(self.padding)}.png"
                                    )
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class MP4Writer():
    def __init__(self, dst_path, width, height, fps_num, fps_den=1):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps_num}/{fps_den}", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst_path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write_one_frame(self, rgb):
        # rgb: 3xhxw uint8 numpy array
        self.process.stdin.write(rgb.transpose(1, 2, 0).tobytes())

    def close(self):
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {self.dst_path}")


class YUV420Writer():
    def __init__(self, dst_path, width, height):
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
        self.dst_path = dst_path
        self.width = width
        self.height = height

        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, y, uv):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()
