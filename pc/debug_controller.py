import os
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import pygame


def poll(joystick, previous):
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
            state = "pressed" if event.type == pygame.JOYBUTTONDOWN else "released"
            print(f"button {event.button} {state}", flush=True)
        elif event.type == pygame.JOYDEVICEREMOVED:
            raise pygame.error("controller disconnected")
    values = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    if previous is None or any(abs(v - p) > 0.05 for v, p in zip(values, previous)):
        print("axes:", [round(v, 2) for v in values], flush=True)
        previous = values
    return values, previous


def detect_axis(joystick, prompt, previous):
    print(prompt, flush=True)
    baseline = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    peaks = [0.0] * len(baseline)
    moved, released_at = False, None
    while True:
        values, previous = poll(joystick, previous)
        offsets = [v - b for v, b in zip(values, baseline)]
        peaks = [v if abs(v) > abs(p) else p for v, p in zip(offsets, peaks)]
        if max(map(abs, offsets), default=0.0) > 0.15:
            moved, released_at = True, None
        elif moved:
            released_at = released_at or time.monotonic()
            if time.monotonic() - released_at >= 0.4:
                axis = max(range(len(peaks)), key=lambda i: abs(peaks[i]))
                return axis, peaks, previous
        time.sleep(0.1)


def main():
    pygame.display.init()
    pygame.joystick.init()
    previous = None
    try:
        while True:
            count = pygame.joystick.get_count()
            print(f"controllers: {count}", flush=True)
            if not count:
                print("no controller detected: check the data cable, close Steam and DS4Windows, replug", flush=True)
                time.sleep(2)
                pygame.joystick.quit()
                pygame.joystick.init()
                continue
            joystick = pygame.joystick.Joystick(0)
            try:
                joystick.init()
                print(f"name: {joystick.get_name()}, axes: {joystick.get_numaxes()}, buttons: {joystick.get_numbuttons()}, hats: {joystick.get_numhats()}", flush=True)
                x_axis, _, previous = detect_axis(joystick, "Push the RIGHT stick fully RIGHT, then release", previous)
                y_axis, y_peaks, previous = detect_axis(joystick, "Push the RIGHT stick fully UP, then release", previous)
                direction = "negative" if y_peaks[y_axis] < 0 else "positive"
                print(f"STICK_X_AXIS = {x_axis}\nSTICK_Y_AXIS = {y_axis}")
                print(f"UP reads as {direction} on the Y axis (should be negative).")
                while True:
                    _, previous = poll(joystick, previous)
                    time.sleep(0.1)
            except pygame.error:
                print("controller disconnected, waiting...", flush=True)
                joystick.quit()
                pygame.joystick.quit()
                pygame.joystick.init()
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nController test stopped.", flush=True)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
