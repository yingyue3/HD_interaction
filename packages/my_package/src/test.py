import sys, tty, termios

def get_char_unix():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Example usage
print("Press a key (Unix-like):")
char = get_char_unix()
print(f"You pressed: '{char}'")
