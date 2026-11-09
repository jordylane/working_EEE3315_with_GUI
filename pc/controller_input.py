DEADZONE = 0.35


def controller_commands(axes):
    values = list(axes[:4])
    values.extend([0.0] * (4 - len(values)))
    left_x, left_y, right_x, right_y = values
    commands = set()

    if right_x < -DEADZONE:
        commands.add(b"L")
    elif right_x > DEADZONE:
        commands.add(b"R")
    if right_y < -DEADZONE:
        commands.add(b"U")
    elif right_y > DEADZONE:
        commands.add(b"D")

    if left_x < -DEADZONE:
        commands.add(b"a")
    elif left_x > DEADZONE:
        commands.add(b"d")
    if left_y < -DEADZONE:
        commands.add(b"w")
    elif left_y > DEADZONE:
        commands.add(b"s")

    return commands
