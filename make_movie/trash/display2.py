import ctypes

def get_refresh_rate():
    class DEVMODE(ctypes.Structure):
        _fields_ = [
            ("dmDeviceName", ctypes.c_wchar * 32),
            ("dmSpecVersion", ctypes.c_ushort),
            ("dmDriverVersion", ctypes.c_ushort),
            ("dmSize", ctypes.c_ushort),
            ("dmDriverExtra", ctypes.c_ushort),
            ("dmFields", ctypes.c_ulong),
            ("dmPosition_x", ctypes.c_long),
            ("dmPosition_y", ctypes.c_long),
            ("dmDisplayOrientation", ctypes.c_ulong),
            ("dmDisplayFixedOutput", ctypes.c_ulong),
            ("dmColor", ctypes.c_short),
            ("dmDuplex", ctypes.c_short),
            ("dmYResolution", ctypes.c_short),
            ("dmTTOption", ctypes.c_short),
            ("dmCollate", ctypes.c_short),
            ("dmFormName", ctypes.c_wchar * 32),
            ("dmLogPixels", ctypes.c_ushort),
            ("dmBitsPerPel", ctypes.c_ulong),
            ("dmPelsWidth", ctypes.c_ulong),
            ("dmPelsHeight", ctypes.c_ulong),
            ("dmDisplayFlags", ctypes.c_ulong),
            ("dmDisplayFrequency", ctypes.c_ulong),
        ]

    ENUM_CURRENT_SETTINGS = -1
    user32 = ctypes.windll.user32
    dm = DEVMODE()
    dm.dmSize = ctypes.sizeof(DEVMODE)

    if user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(dm)):
        return dm.dmDisplayFrequency
    else:
        return None

rate = get_refresh_rate()
print(f"Current Refresh Rate: {rate} Hz")
