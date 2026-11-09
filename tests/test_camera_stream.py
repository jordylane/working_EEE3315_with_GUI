import unittest

from pc.mjpeg import extract_jpeg_frames


class CameraStreamTests(unittest.TestCase):
    def test_extracts_jpeg_frames_split_across_chunks(self):
        buffer = bytearray(b"multipart preamble\xff")

        self.assertEqual(extract_jpeg_frames(buffer), [])

        buffer.extend(b"\xd8first frame\xff")
        self.assertEqual(extract_jpeg_frames(buffer), [])

        buffer.extend(b"\xd9--boundary--\xff\xd8second frame\xff\xd9")
        self.assertEqual(
            extract_jpeg_frames(buffer),
            [
                b"\xff\xd8first frame\xff\xd9",
                b"\xff\xd8second frame\xff\xd9",
            ],
        )
        self.assertEqual(buffer, b"")

    def test_discards_stream_data_before_a_frame(self):
        buffer = bytearray(
            b"multipart headers\xff\xd8frame data\xff\xd9"
        )

        self.assertEqual(
            extract_jpeg_frames(buffer),
            [b"\xff\xd8frame data\xff\xd9"],
        )
        self.assertEqual(buffer, b"")


if __name__ == "__main__":
    unittest.main()
