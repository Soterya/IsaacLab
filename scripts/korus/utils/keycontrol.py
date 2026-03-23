# -------- Keyboard ----------
import omni.appwindow
import carb
from carb.input import KeyboardEventType, KeyboardInput

class KeyControl:
    """Hold/press to move the prismatic joint:
       Up/W = inflate (+), Down/S = deflate (-), Space = stop.
    """
    def __init__(self, step: float = 0.002):
        self.step = float(step)
        self.inc  = False
        self.dec  = False
        self._sub_id   = None
        self._input    = None
        self._keyboard = None

    def subscribe(self):
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            carb.log_warn("No app window; keyboard control disabled.")
            return
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub_id = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_input)

    def unsubscribe(self):
        try:
            if self._input and self._keyboard and self._sub_id:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_id)
        finally:
            self._sub_id = None
            self._keyboard = None
            self._input = None

    def on_keyboard_input(self, e):
        if e.input in (KeyboardInput.W, KeyboardInput.UP):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.inc = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.inc = False
        elif e.input in (KeyboardInput.S, KeyboardInput.DOWN):
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.dec = True
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.dec = False
        elif e.input == KeyboardInput.SPACE:
            if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
                self.inc = False
                self.dec = False