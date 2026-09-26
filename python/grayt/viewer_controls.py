"""Native Qt settings for the desktop observer; imported only by the viewer."""

from PySide6 import QtCore, QtGui, QtWidgets

from .emission import PageThorneDisk
from .matter import ThinDisk
from .spacetime import BlackHole, QMetric


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.closing = False
        self.setCentralWidget(viewer.canvas.native)
        self.resize(*viewer.canvas.size)
        self.controls = ViewerControls(viewer)
        self.dock = QtWidgets.QDockWidget("Settings", self)
        self.dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
                                  QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(scroll)
        body_layout.addWidget(self.controls.actions)
        self.dock.setWidget(body)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.toggle = self.dock.toggleViewAction()
        self.toggle.setText("Settings [H]")
        self.toggle.setShortcut(QtGui.QKeySequence("H"))
        toolbar = self.addToolBar("Viewer")
        toolbar.setMovable(False)
        toolbar.addAction(self.toggle)
        self.setStyleSheet("""
            QMainWindow, QDockWidget, QScrollArea, QWidget#viewerSettings {
                background: #101216; color: #e9e5dc;
            }
            QLabel, QCheckBox { color: #e9e5dc; }
            QToolBar { background: #181c22; border: 0; spacing: 8px; }
            QDoubleSpinBox, QComboBox {
                background: #242a33; color: #f5f0e4; padding: 4px;
            }
            QPushButton, QToolButton {
                background: #303845; color: #f5f0e4; padding: 6px 10px;
                border: 1px solid #485363; border-radius: 4px;
            }
            QPushButton:hover, QToolButton:hover { background: #414c5c; }
        """)

    def closeEvent(self, event):
        self.closing = True
        self.viewer.canvas.close()
        super().closeEvent(event)


class ViewerControls(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.setObjectName("viewerSettings")
        self.viewer = viewer
        self.dirty = False
        self.sky_path = None
        self.fields = {}
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.form = form
        self.number("spin", "Spin a", -0.999, 0.999, 0.01, 3)
        self.number("q", "Quadrupole q", -0.99, 10, 0.01, 3)
        self.fields["spin"].setEnabled(isinstance(viewer.spacetime, BlackHole))
        self.fields["q"].setEnabled(isinstance(viewer.spacetime, QMetric))
        self.number("radius", "Observer radius / M", 2.02, 1e6, 10)
        self.number("theta", "Inclination °", 3.2, 176.8, 1)
        self.number("phi", "Azimuth °", 0, 360, 1)
        self.number("fov_x", "View half-width / M", 0.25, 1000, 1)
        self.number("fov_y", "View half-height / M", 0.25, 1000, 1)
        disk = QtWidgets.QCheckBox("Show disk")
        self.fields["disk_on"] = disk
        form.addRow(disk)
        disk.toggled.connect(self.changed)
        self.number("r_out", "Disk outer radius / M", 0.1, 1e5, 1)
        self.number("l0", "Disk angular momentum", -100, 100, 0.1)
        supported = (isinstance(viewer.spacetime, BlackHole) and
                     (viewer.disk is None or isinstance(viewer.disk, (ThinDisk, PageThorneDisk))))
        for name in ("disk_on", "r_out"):
            self.fields[name].setEnabled(supported)
        self.fields["l0"].setEnabled(supported and not isinstance(viewer.disk, PageThorneDisk))
        if viewer.disk is not None and not supported:
            self.fields["spin"].setEnabled(False)
        background = QtWidgets.QComboBox()
        for label, value in [("Black", "black"), ("Celestial map", "celestial"),
                             ("Diagnostic grid", "grid")]:
            background.addItem(label, value)
        self.fields["background"] = background
        form.addRow("Background", background)
        background.currentIndexChanged.connect(self.changed)
        self.map_label = QtWidgets.QLabel()
        self.map_label.setWordWrap(True)
        form.addRow("Sky image", self.map_label)
        image_buttons = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Choose image…")
        nasa = QtWidgets.QPushButton("NASA map")
        load.clicked.connect(self.choose_image)
        nasa.clicked.connect(self.use_nasa)
        image_buttons.addWidget(load)
        image_buttons.addWidget(nasa)
        layout.addLayout(image_buttons)
        self.number("sky_longitude", "Sky rotation °", -360, 360, 5)
        self.number("sky_gain", "Background brightness", 0.01, 100, 0.25)
        self.number("brightness", "Disk brightness", 1 / 32, 32, 0.25, 3)
        hint = QtWidgets.QLabel("Sky images use an equirectangular map.\nDistances are in GM/c².\nDrag to orbit · Scroll to zoom · H to hide settings")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.actions = QtWidgets.QWidget()
        action_layout = QtWidgets.QVBoxLayout(self.actions)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        action_layout.addWidget(self.status)
        apply = QtWidgets.QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self.apply)
        action_layout.addWidget(apply)
        reset = QtWidgets.QPushButton("Reset camera")
        reset.clicked.connect(self.reset_camera)
        action_layout.addWidget(reset)
        save = QtWidgets.QPushButton("Save PNG…")
        save.clicked.connect(self.save)
        action_layout.addWidget(save)
        layout.addStretch()
        self.setMinimumWidth(290)
        self.sync()

    def number(self, name, label, minimum, maximum, step, decimals=2):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setKeyboardTracking(False)
        self.fields[name] = box
        self.form.addRow(label, box)
        box.valueChanged.connect(self.changed)

    def changed(self, *_):
        self.dirty = True
        self.status.setText("Press Apply to update the view.")

    def sync(self):
        if self.dirty:
            return
        viewer, camera = self.viewer, self.viewer.state.camera
        disk = viewer.disk or viewer._disk_template
        with QtCore.QSignalBlocker(self.fields["radius"]):
            self.fields["radius"].setMinimum(viewer.spacetime.capture_radius + 0.02)
            self.fields["radius"].setMaximum(viewer.escape_radius - 0.01 if viewer.escape_radius else 1e6)
        values = {
            "spin": getattr(viewer.spacetime, "a", 0), "q": getattr(viewer.spacetime, "q", 0),
            "radius": camera.r, "theta": camera.theta, "phi": camera.phi,
            "fov_x": (camera.x[1] - camera.x[0]) / 2,
            "fov_y": (camera.y[1] - camera.y[0]) / 2,
            "r_out": getattr(disk, "r_out", 20), "l0": getattr(disk, "l0", 1.8),
            "sky_longitude": getattr(viewer.sky, "longitude", viewer._sky_longitude),
            "sky_gain": getattr(viewer.sky, "gain", viewer._sky_gain), "brightness": viewer.brightness,
        }
        for name, value in values.items():
            with QtCore.QSignalBlocker(self.fields[name]):
                self.fields[name].setValue(value)
        with QtCore.QSignalBlocker(self.fields["disk_on"]):
            self.fields["disk_on"].setChecked(viewer.disk is not None)
        box = self.fields["background"]
        with QtCore.QSignalBlocker(box):
            box.setCurrentIndex(box.findData(viewer.background))
        self.map_label.setText(getattr(viewer.sky, "name", "NASA Deep Star Maps"))

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose sky map", "",
                                                      "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.sky_path = path
            self.map_label.setText(path)
            self.fields["background"].setCurrentIndex(1)
            self.changed()

    def use_nasa(self):
        self.sky_path = ""
        self.map_label.setText("NASA Deep Star Maps")
        self.fields["background"].setCurrentIndex(1)
        self.changed()

    def apply(self):
        try:
            values = {name: field.value() for name, field in self.fields.items()
                      if isinstance(field, QtWidgets.QDoubleSpinBox)}
            self.viewer.update_parameters(
                **{name: values[name] for name in ("radius", "theta", "phi", "fov_x", "fov_y",
                                                  "sky_longitude", "sky_gain", "brightness")},
                spin=values["spin"] if self.fields["spin"].isEnabled() else None,
                q=values["q"] if self.fields["q"].isEnabled() else None,
                disk_on=self.fields["disk_on"].isChecked() if self.fields["disk_on"].isEnabled() else None,
                r_out=values["r_out"] if self.fields["r_out"].isEnabled() else None,
                l0=values["l0"] if self.fields["l0"].isEnabled() else None,
                background=self.fields["background"].currentData(), sky_path=self.sky_path,
            )
        except (ValueError, RuntimeError, OSError) as error:
            self.status.setText(str(error))
            return
        self.sky_path = None
        self.dirty = False
        self.sync()
        self.status.setText("View updated.")
        self.viewer.canvas.native.setFocus()

    def reset_camera(self):
        self.dirty = False
        self.sky_path = None
        self.viewer.state.reset()
        self.viewer.render(refine=True)
        self.sync()

    def save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save current view",
                                                      "gravtracer_view.png", "PNG image (*.png)")
        if path:
            self.viewer.save(path)
