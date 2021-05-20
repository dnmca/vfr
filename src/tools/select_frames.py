
import cv2
import sys
import typing

import numpy as np

from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class FrameCache:
    def __init__(self, max_size=500):

        self.max_size = max_size
        self.cache = {}
        self.log = []

    def add(self, key, value):
        if len(self.cache) > self.max_size:
            self.clear()
        self.cache[key] = value
        self.log.append(key)

    def get(self, key):
        return self.cache[key]

    def clear(self, n=1):
        to_be_deleted = self.log[:n]
        del self.log[:n]
        for x in to_be_deleted:
            del self.cache[int(x)]

    def __contains__(self, item):
        return item in self.cache


class Model:

    def __init__(self):
        self.video_id = None
        self.video_path = None
        self.video_type = None
        self.video_height = None
        self.video_width = None
        self.data_path = None

        self.cap = None
        self.cap_pos = 0
        self.frame_count = 0
        self.seek_interval = 1
        self.saved_frames = 0

        self.current_frame = None
        self.cache = FrameCache()

    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def open_video(self, video_path: Path):

        video_folder = video_path.parent
        self.video_id = video_path.stem

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            print(f'Could not open capture: {video_path}')
            return False

        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.data_path = Path(video_folder) / self.video_id
        self.data_path.mkdir(exist_ok=True)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, self.current_frame = self.cap.read()

        self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.saved_frames = self.count_saved_frames()

        return True

    def count_saved_frames(self):
        count = 0
        for path in self.data_path.iterdir():
            if path.is_file():
                count += 1
        return count

    def construct_destination(self):
        cap_pos_len = len(str(int(self.cap_pos)))
        padded_cap_pos = '0' * (7 - cap_pos_len) + str(int(self.cap_pos))
        destination = self.data_path / f'{self.video_id}_{padded_cap_pos}.png'
        destination.resolve()
        return destination

    def current_saved(self):
        destination = self.construct_destination()
        return destination.exists()

    def save_frame(self):
        if self.cap is not None:
            destination = self.construct_destination()
            cv2.imwrite(str(destination), self.current_frame)
            self.saved_frames += 1

    def delete_frame(self):
        if self.cap is not None:
            destination = self.construct_destination()
            if destination.exists():
                destination.unlink()
                self.saved_frames -= 1

    def get_cap_pos(self):
        return self.cap_pos

    def get_frame(self, frame_idx):
        return self.current_frame

    def set_seek_interval(self, seek_interval):
        self.seek_interval = seek_interval

    def seek_forward(self):
        dest = self.cap_pos + self.seek_interval
        if dest in self.cache:
            self.current_frame = self.cache.get(dest)
            self.cap_pos = dest
        else:
            if dest < self.frame_count:
                if self.seek_interval <= 50:

                    diff = int(self.cap_pos - self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if 50 > diff >= 0:
                        for i in range(self.seek_interval - 1 + diff):
                            _ = self.cap.read()
                        _, self.current_frame = self.cap.read()
                        self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cache.add(self.cap_pos, self.current_frame)
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, dest)
                        _, self.current_frame = self.cap.read()
                        self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cache.add(self.cap_pos, self.current_frame)
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, dest - 1)
                    _, self.current_frame = self.cap.read()
                    self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.cache.add(self.cap_pos, self.current_frame)
            else:
                self.cap_pos = self.frame_count - 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap_pos)
                _, self.current_frame = self.cap.read()
                self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cache.add(self.cap_pos, self.current_frame)

    def seek_backward(self):
        dest = self.cap_pos - self.seek_interval
        if dest in self.cache:
            self.current_frame = self.cache.get(dest)
            self.cap_pos = dest
        else:
            if dest >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, dest - 1)
                _, self.current_frame = self.cap.read()
                self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cache.add(self.cap_pos, self.current_frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, self.current_frame = self.cap.read()
                self.cap_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cache.add(self.cap_pos, self.current_frame)


class Controller:
    def __init__(self, model):
        self.model = model
        self.views = []

    def add_view(self, view):
        view.set_model(self.model)
        view.set_controller(self)
        self.views.append(view)

    def open_video(self, video_path: Path):
        status = self.model.open_video(video_path)
        if status:
            self.on_frame_updated()

    def update_views(self, hint, data=None):

        for view in self.views:
            view.update_view(hint, data)

    def on_frame_updated(self):
        self.update_views("frame", data={
            'frame_idx': self.model.get_cap_pos(),
            'frame_count': self.model.frame_count,
            'saved_frames': self.model.saved_frames
        })

    def seek_forward(self):

        if not self.model.is_open():
            return
        self.model.seek_forward()
        self.on_frame_updated()

    def seek_backward(self):
        if not self.model.is_open():
            return
        self.model.seek_backward()
        self.on_frame_updated()

    def set_interval(self, interval):
        self.model.set_seek_interval(interval)

    def save_frame(self):
        self.model.save_frame()

    def delete_frame(self):
        self.model.delete_frame()


class FrameView(QLabel):
    def __init__(self, parent):
        super(FrameView, self).__init__(parent)
        self.model = None
        self.controller = None

        self.true_w, self.true_h = None, None
        self.visual_w, self.visual_h = None, 720

        self.current_frame = None

    def set_model(self, model):
        self.model = model

    def set_controller(self, controller):
        self.controller = controller

    def update_view(self, hint, data):
        if hint == "frame":
            frame_idx = self.model.get_cap_pos()
            frame = self.model.get_frame(frame_idx)

            self.true_h, self.true_w, _ = frame.shape
            scale = self.true_h / self.visual_h
            self.visual_w = int(np.round(self.true_w / scale))
            self.current_frame = cv2.resize(frame, (self.visual_w, self.visual_h))
            self.set_image(self.current_frame)

    def set_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bytes_per_line = 3 * self.visual_w
        q_image = QImage(image.data, self.visual_w, self.visual_h, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(q_pixmap)
        self.setFixedSize(q_pixmap.size())


class Application(QApplication):
    key_pressed_signal = pyqtSignal(object)

    def __init__(self, argv: typing.List[str]):
        super().__init__(argv)

    def notify(self, receiver, e):
        if e.type() == QEvent.KeyPress:
            if e.key() == Qt.Key_Left:
                self.key_pressed_signal.emit(e)
                return True
            elif e.key() == Qt.Key_Right:
                self.key_pressed_signal.emit(e)
                return True
            elif e.key() == Qt.Key_Space:
                self.key_pressed_signal.emit(e)
                return True

        return QApplication.notify(self, receiver, e)


class UI(QObject):

    def setupUI(self, main_window):

        main_window.setWindowTitle("Select frames")
        main_window.resize(1920, 1080)

        self.central_widget = QWidget(main_window)
        self.central_widget.setObjectName('central_widget')

        self.central_layout = QHBoxLayout()

        self.main = QWidget(self.central_widget)
        self.main.setObjectName('main')

        self.main_layout = QVBoxLayout(self.main)

        # defining left widgets:
        self.menu = QWidget(self.main)
        self.menu.setMaximumHeight(65)
        self.menu.setObjectName('tools')
        # self.menu_layout = QHBoxLayout(self.menu)

        # defining counters
        self.frameCounter = QLabel(f'Frame: {-1} ({np.round(0.0, 2)}%)', self.menu)
        self.frameCounter.setGeometry(QRect(50, 0, 250, 30))
        self.frameCounter.setObjectName('frameCounter')

        self.saved_counter = QLabel(f'Saved: {0}', self.menu)
        self.saved_counter.setGeometry(QRect(50, 30, 250, 30))
        self.saved_counter.setObjectName('saved_counter')

        # defining buttons
        self.open_video_button = QPushButton(self.menu)
        self.open_video_button.setGeometry(QRect(400, 0, 120, 60))
        self.open_video_button.setObjectName('open_video_button')

        self.interval_button = QComboBox(self.menu)
        self.interval_button.setGeometry(QRect(690, 0, 120, 60))
        self.interval_button.setObjectName("interval_button")
        self.interval_button.addItems(
            ['1', '5', '10', '15', '20', '25', '50', '100', '200', '500', '1000', '2000'])

        self.prev_frame = QPushButton(self.menu)
        self.prev_frame.setGeometry(QRect(820, 0, 120, 60))
        self.prev_frame.setObjectName("prev_frame")
        self.prev_frame.setToolTip('Key Left')

        self.next_frame = QPushButton(self.menu)
        self.next_frame.setGeometry(QRect(950, 0, 120, 60))
        self.next_frame.setObjectName("next_frame")
        self.next_frame.setToolTip('Key Right')

        self.save = QPushButton(self.menu)
        self.save.setGeometry(QRect(1350, 0, 120, 60))
        self.save.setObjectName("save")
        self.save.setToolTip('Key Space')

        self.delete = QPushButton(self.menu)
        self.delete.setGeometry(QRect(1480, 0, 120, 60))
        self.delete.setObjectName('delete')

        font = QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(14)

        self.menu.setFont(font)

        self.annotation_layout = QHBoxLayout(self.main)
        self.annotation = QWidget(self.main)

        # defining frame view
        self.frame_view = FrameView(self.main)
        self.frame_view.setObjectName('frame_view')

        for x in [
            self.frame_view
        ]:
            self.annotation_layout.addWidget(x)

        self.annotation.setLayout(self.annotation_layout)

        for x in [
            self.menu,
            self.annotation
        ]:
            self.main_layout.addWidget(x)

        for x in [
            self.main
        ]:
            self.central_layout.addWidget(x)

        self.central_widget.setLayout(self.central_layout)
        main_window.setCentralWidget(self.central_widget)

        self.retranslate_ui()

        QMetaObject.connectSlotsByName(main_window)

    def retranslate_ui(self):
        _translate = QCoreApplication.translate
        self.open_video_button.setText(_translate("Ball Annotator", "Open video"))
        self.interval_button.setWindowIconText(_translate("Ball Annotator", "Interval"))
        self.next_frame.setText(_translate("Ball Annotator", "Next"))
        self.prev_frame.setText(_translate("Ball Annotator", "Previous"))
        self.save.setText(_translate("Ball Annotator", "Save"))
        self.delete.setText(_translate("Ball Annotator", "Delete"))


class MainWindowUI(UI):

    def __init__(self):
        self.model = Model()
        self.controller = Controller(self.model)

        super().__init__()

    def setupUI(self, main_window):
        super().setupUI(main_window)
        self.ui = main_window

        self.open_video_button.clicked.connect(self.open_video_clicked)

        self.interval_button.currentIndexChanged.connect(self.interval_clicked)
        self.next_frame.clicked.connect(self.next_clicked)
        self.prev_frame.clicked.connect(self.prev_clicked)
        self.save.clicked.connect(self.save_clicked)
        self.delete.clicked.connect(self.delete_clicked)

        self.controller.add_view(self)
        self.controller.add_view(self.frame_view)

        self.delete.setVisible(False)

    def open_video_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_path, _ = QFileDialog.getOpenFileName(self.ui, 'Open video file', '', 'Video file (*.mp4 *.mkv *.webm)')
        if video_path:
            self.controller.open_video(Path(video_path))

    def interval_clicked(self):
        current_interval = self.interval_button.currentText()
        self.controller.set_interval(int(current_interval))

    def next_clicked(self):
        self.controller.seek_forward()

    def prev_clicked(self):
        self.controller.seek_backward()

    def save_clicked(self):
        self.controller.save_frame()

        self.saved_counter.setText(
            f'Saved: {self.model.count_saved_frames()}'
        )
        self.delete.setVisible(True)

    def delete_clicked(self):
        self.controller.delete_frame()
        self.saved_counter.setText(
            f'Saved: {self.model.count_saved_frames()}'
        )
        self.delete.setVisible(False)

    def key_press_event(self, e):
        if e.key() == Qt.Key_Left:
            self.prev_clicked()
        elif e.key() == Qt.Key_Right:
            self.next_clicked()
        elif e.key() == Qt.Key_Space:
            self.save_clicked()

    def set_model(self, model):
        pass

    def set_controller(self, controller):
        pass

    def update_view(self, hint, data=None):
        if hint == 'frame' and data is not None:
            self.frameCounter.setText(
                f'Frame: {int(data["frame_idx"])} ({np.round((data["frame_idx"] + 1) / data["frame_count"] * 100.0, 2)}%)'
            )
            self.saved_counter.setText(
                f'Saved: {data["saved_frames"]}'
            )
            if self.model.current_saved():
                self.delete.setVisible(True)
            else:
                self.delete.setVisible(False)


def main():
    app = Application(sys.argv)
    app.setStyle('Oxygen')
    main_window = QMainWindow()
    ui = MainWindowUI()
    ui.setupUI(main_window)
    app.key_pressed_signal.connect(ui.key_press_event)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()