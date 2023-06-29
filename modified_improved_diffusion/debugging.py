import inspect

debug_mode = False

def debug_print(text):
    if debug_mode:
        caller_frame = inspect.currentframe().f_back
        frame_info = inspect.getframeinfo(caller_frame)
        line_number = frame_info.lineno
        file_name = frame_info.filename.split("/")[-1]
        print(f"[{file_name}, Line {line_number}] {text}")

