from supervisely import Application

from src.ui.ui import layout
import src.globals as g


app = Application(layout=layout, static_dir=g.TEMP_DATA_PATH)
