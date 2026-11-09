def extract_jpeg_frames(buffer):
    frames = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        end = buffer.find(b"\xff\xd9", 2)
        if end < 0:
            break
        frames.append(bytes(buffer[:end + 2]))
        del buffer[:end + 2]
    return frames
