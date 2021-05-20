
import cv2
import sys
import json
import typing

import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5 import QtGui

from src.core.constants import KeyPoint, KEYPOINTS, CONNECTIONS
from src.core.utils import initialize_field_template

# TODO: add counter


class KeypointTracker:

    def __init__(self):

        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def interpolate(self, prev_frame, curr_frame, prev_annotation):

        prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        height, width, _ = prev_frame.shape

        prev_points = np.array([[pos[0], pos[1]] for name, pos in prev_annotation.items()]).astype(np.float32)

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_grey,
            nextImg=curr_grey,
            prevPts=prev_points,
            nextPts=None,
            **self.lk_params
        )

        interpolated = {}

        for index, name in enumerate(prev_annotation.keys()):
            x, y = next_points[index]
            x = int(np.round(x))
            x = int(np.round(x))

            if x >= width or y >= height or x < 0 or y < 0:
                interpolated[name] = prev_annotation[name]
            else:
                interpolated[name] = [x, y]

        return interpolated


class Model:

    def __init__(self):

        self.data_path = None
        self.anno_path = None

        self.idx2name = {}
        self.idx2anno = {}
        self.idx2frame = {}
        self.current_idx = 0

        self.keypoint_id = None

    def get_current_idx(self):
        return self.current_idx

    def reset_model(self):
        self.data_path = None
        self.anno_path = None

        self.idx2name = {}
        self.idx2anno = {}
        self.idx2frame = {}
        self.current_idx = 0
        self.keypoint_id = None

    def open_folder(self, data_path):
        self.reset_model()
        self.data_path = Path(data_path)
        self.anno_path = self.data_path / 'annotation'

        if not self.anno_path.exists():
            self.anno_path.mkdir(exist_ok=True)

        image_paths = sorted(self.data_path.glob('*.png'))

        if len(image_paths) == 0:
            return False
        
        for index, image_path in enumerate(image_paths):
            img_name = image_path.stem
            self.idx2name[index] = img_name

            image = cv2.imread(str(image_path))
            self.idx2frame[index] = image

            anno = self.read_anno_by_idx(index)
            if anno is not None:
                self.idx2anno[index] = anno

        self.current_idx = 0
        return True

    def read_anno_by_idx(self, idx):
        anno_file = self.anno_path / f'{self.idx2name[idx]}.json'
        if anno_file.exists():
            with open(str(anno_file), 'r') as file:
                anno = json.load(file)
                return anno
        return None

    def all_changes_saved(self, idx):

        anno = self.read_anno_by_idx(idx)
        if anno is not None:
            if anno == self.idx2anno[idx]:
                return True
            else:
                return False
        else:
            return False

    def save_annotation(self):
        anno_file = self.anno_path / f'{self.idx2name[self.current_idx]}.json'
        with open(str(anno_file), 'w') as file:
            json.dump(self.idx2anno[self.current_idx], file)

    def get_annotation(self, idx, reset=False):
        if idx not in self.idx2anno or reset:
            height, width, _ = self.idx2frame[idx].shape
            annotation = initialize_field_template((width, height))
            self.idx2anno[idx] = annotation
        return self.idx2anno[idx]

    def set_annotation(self, idx, anno):
        self.idx2anno[idx] = anno

    def get_frame(self, idx):
        return self.idx2frame[idx]
    
    def get_frames(self, idx, offset=3):
        return {i: frame for i, frame in self.idx2frame.items() if abs(idx - i) <= 3}

    def next(self):
        if self.current_idx + 1 in self.idx2frame:
            self.current_idx += 1

    def prev(self):
        if self.current_idx - 1 in self.idx2frame:
            self.current_idx -= 1


class SequenceView(QGraphicsView):
    def __init__(self, parent):
        QGraphicsView.__init__(self, parent=parent)
        self.model: Model = None
        self.controller: Controller = None
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setStyleSheet("background: transparent")

    def set_model(self, model: Model):
        self.model= model

    def set_controller(self, controller):
        self.controller = controller

    def add_padding(self, img, color, size):
        return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=color)

    def plot_text(self, image, x, y, text, font_size):
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('resources/UbuntuMono-R.ttf', font_size)
        draw.text((x, y), text, fill=(255, 255, 255), anchor="lb", font=font, align='left')
        return np.array(img)

    def update_view(self, hint, data):
        if hint == 'frame':
            current_idx = self.model.get_current_idx()
            idx_to_image = self.model.get_frames(current_idx)

            h, w, c = idx_to_image[current_idx].shape
            divisor = h / 100
            w_new = int(np.round(w / divisor))

            resized_images = []

            for index in range(current_idx - 3, current_idx + 4):
                if index in idx_to_image:
                    img = idx_to_image[index]
                    img = cv2.resize(img, dsize=(w_new, 100))

                    img = self.plot_text(img, x=10, y=90, text=f'{index+1}', font_size=30)

                    if self.model.all_changes_saved(index):
                        gray = np.zeros_like(img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray[:, :, 0] = img
                        gray[:, :, 1] = img
                        gray[:, :, 2] = img
                        img = gray
                else:
                    img = np.ones(shape=(100, w_new, 3), dtype=np.uint8) * 255

                border_color = [153, 153, 0] if index == current_idx else [255, 255, 255]
                img = self.add_padding(img, color=border_color, size=5)
                resized_images.append(img)

            seq_img = cv2.hconcat(resized_images)
            self.set_image(seq_img)

    def set_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.scene.clear()
        pixmap = self.scene.addPixmap(QPixmap.fromImage(qimage))
        self.ensureVisible(self.scene.sceneRect())
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        pixmap.setTransformationMode(Qt.SmoothTransformation)


class ImageLabel(QLabel):
    def __init__(self, parent):
        super(ImageLabel, self).__init__(parent)

        self.tracker = KeypointTracker()

        self.model: Model = None
        self.controller = None
        self.image = None
        self.draw: ImageDraw = None

        self.copy_cache = None

        self.keypoints = {}

        self.curr_idx = None
        self.curr_keypoint = None

        self.true_h, self.true_w = None, None
        self.visual_h, self.visual_w = 720, None

    def get_keypoint(self, x, y):
        for name, pos in self.keypoints.items():
            x_pos, y_pos = pos
            if x_pos - 10 <= x <= x_pos + 10 and y_pos - 10 <= y <= y_pos + 10:
                return name
        return None

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.buttons() == Qt.LeftButton:

            x = ev.localPos().x()
            y = ev.localPos().y()
            keypoint = self.get_keypoint(x, y)

            if keypoint is not None:
                if keypoint == self.curr_keypoint:
                    self.curr_keypoint = None
                else:
                    self.curr_keypoint = keypoint
            else:
                if self.curr_keypoint is not None:
                    self.keypoints[self.curr_keypoint] = [x, y]
                    self.upload_anno()

            self.update_view(hint='frame', data=None)

        elif ev.buttons() == Qt.RightButton:
            pass

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        pass

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        pass

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        pass

    def set_model(self, model: Model):
        self.model= model

    def set_controller(self, controller):
        self.controller = controller

    def plot_text(self, text, x, y, font_size, color=(255, 255, 255)):
        font = ImageFont.truetype('resources/UbuntuMono-R.ttf', font_size)
        self.draw.text((x, y), text, fill=color, anchor="ms", font=font)

    def plot_key_point(self, x, y, focus):
        self.draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=None, outline='red', width=2)
        if focus:
            self.draw.rectangle((x - 12, y - 12, x + 12, y + 12), fill=None, outline=(51, 255, 153), width=2)

    def plot_connection(self, a, b):
        x_a, y_a = a
        x_b, y_b = b
        self.draw.line(xy=[x_a, y_a, x_b, y_b], fill=(255, 255, 255), width=1)

    def plot_grid(self):
        for connection in CONNECTIONS:
            a, b = connection
            if a in self.keypoints and b in self.keypoints:
                self.plot_connection(self.keypoints[a], self.keypoints[b])

    def load_anno(self, reset=False):
        annotation = self.model.get_annotation(self.curr_idx, reset)
        self.keypoints.clear()

        for name, pos in annotation.items():
            x, y = pos
            x_scaled = int(np.round(x * self.visual_w / self.true_w))
            y_scaled = int(np.round(y * self.visual_h / self.true_h))
            self.keypoints[name] = [x_scaled, y_scaled]

    def upload_anno(self):
        rescaled = {}
        for name, pos in self.keypoints.items():
            x, y = pos
            x_scaled = int(np.round(x * self.true_w / self.visual_w))
            y_scaled = int(np.round(y * self.true_h / self.visual_h))
            rescaled[name] = [x_scaled, y_scaled]
        self.model.set_annotation(self.curr_idx, rescaled)

    def plot_anno(self):
        image = Image.fromarray(self.image)
        self.draw = ImageDraw.Draw(image)

        for name, pos in self.keypoints.items():
            x, y = pos
            focus = True if name == self.curr_keypoint else False
            self.plot_key_point(x, y, focus)
            self.plot_text(f'{name}', x, y - 30, font_size=20)

        self.plot_grid()
        self.image = np.array(image)

    def update_view(self, hint, data):
        if hint == "frame":
            self.curr_idx = self.model.get_current_idx()

            image = self.model.get_frame(self.curr_idx)
            self.true_h, self.true_w, _ = image.shape
            scale = self.true_h / self.visual_h
            self.visual_w = int(np.round(self.true_w / scale))
            self.image = cv2.resize(image, (self.visual_w, self.visual_h))

            self.load_anno()
            self.plot_anno()
            self.set_image(self.image)
        elif hint == 'delete_keypoint':
            if self.curr_keypoint is not None:
                del self.keypoints[self.curr_keypoint]
                self.curr_keypoint = None
            self.upload_anno()
        elif hint == 'copy_annotation':
            self.copy_cache = self.keypoints.copy()
        elif hint == 'paste_annotation':
            if self.copy_cache is not None:
                self.keypoints = self.copy_cache
                self.upload_anno()
        elif hint == 'reset_annotation':
            self.load_anno(reset=True)
            self.upload_anno()
        elif hint == 'infer_annotation':
            if self.curr_idx > 0:
                prev_image = self.model.get_frame(self.curr_idx - 1)
                prev_anno = self.model.get_annotation(self.curr_idx - 1)
                curr_image = self.model.get_frame(self.curr_idx)

                keypoints = self.tracker.interpolate(
                    prev_frame=prev_image,
                    curr_frame=curr_image,
                    prev_annotation=prev_anno
                )

                self.keypoints = {}

                for name, pos in keypoints.items():
                    x, y = pos
                    x_scaled = int(np.round(x * self.visual_w / self.true_w))
                    y_scaled = int(np.round(y * self.visual_h / self.true_h))
                    self.keypoints[name] = [x_scaled, y_scaled]

                self.upload_anno()

    def set_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bytes_per_line = 3 * self.visual_w
        q_image = QImage(image.data, self.visual_w, self.visual_h, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(q_pixmap)
        self.setFixedSize(q_pixmap.size())


class Controller:

    def __init__(self, model):
        self.model: Model = model
        self.views = []

    def add_view(self, view):
        view.set_model(self.model)
        view.set_controller(self)
        self.views.append(view)

    def on_frame_update(self):
        self.update_views('frame', None)

    def open_folder(self, folder_path):
        status = self.model.open_folder(folder_path)
        if status:
            self.on_frame_update()

    def update_views(self, hint, data=None):
        for view in self.views:
            view.update_view(hint, data)

    def next_frame(self):
        self.model.next()
        self.on_frame_update()

    def prev_frame(self):
        self.model.prev()
        self.on_frame_update()

    def delete_keypoint(self):
        self.update_views('delete_keypoint', None)
        self.on_frame_update()

    def copy_annotation(self):
        self.update_views('copy_annotation', None)

    def paste_annotation(self):
        self.update_views('paste_annotation', None)
        self.on_frame_update()

    def reset_annotation(self):
        self.update_views('reset_annotation', None)
        self.on_frame_update()

    def infer_annotation(self):
        self.update_views('infer_annotation', None)
        self.on_frame_update()

    def save_annotation(self):
        self.model.save_annotation()


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
            elif e.key() == Qt.Key_Delete:
                self.key_pressed_signal.emit(e)
                return True

        return QApplication.notify(self, receiver, e)


class UI(QObject):

    def setupUI(self, MainWindow):

        MainWindow.setWindowTitle("Keypoint Annotation")
        MainWindow.resize(1920, 1080)

        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName('central_widget')

        self.centralLayout = QHBoxLayout()

        self.main = QWidget(self.centralWidget)
        self.main.setObjectName('main')

        self.main_layout = QVBoxLayout(self.main)

        self.menu = QWidget(self.main)
        self.menu.setMinimumHeight(50)
        self.menu.setObjectName('tools')
        # self.menu_layout = QHBoxLayout(self.menu)

        # defining buttons
        self.open_folder = QPushButton(self.menu)
        self.open_folder.setGeometry(QRect(20, 0, 80, 45))
        self.open_folder.setObjectName('open_folder')

        self.prev_frame = QPushButton(self.menu)
        self.prev_frame.setGeometry(QRect(120, 0, 80, 45))
        self.prev_frame.setObjectName("prev_frame")
        self.prev_frame.setToolTip('Key Left')

        self.next_frame = QPushButton(self.menu)
        self.next_frame.setGeometry(QRect(220, 0, 80, 45))
        self.next_frame.setObjectName("next_frame")
        self.next_frame.setToolTip('Key Right')

        self.save = QPushButton(self.menu)
        self.save.setGeometry(QRect(320, 0, 80, 45))
        self.save.setObjectName("save")
        self.save.setToolTip('Key Space')

        self.curr_keypoint = QLabel(self.menu)
        self.curr_keypoint.setGeometry(QRect(620, 0, 80, 45))
        self.curr_keypoint.setObjectName("curr_keypoint")

        font = QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(14)

        self.menu.setFont(font)

        # sequence visualization
        self.sequence_view = SequenceView(self.main)
        self.sequence_view.setObjectName('sequence_view')
        # self.sequence_view.setMaximumHeight(100)

        self.annotation_layout = QHBoxLayout(self.main)

        self.annotation = QWidget(self.main)

        # defining frame view
        self.frame_view = ImageLabel(self.main)
        self.frame_view.setObjectName('frame_view')

        for x in [
            self.frame_view
        ]:
            self.annotation_layout.addWidget(x)

        self.annotation.setLayout(self.annotation_layout)

        for x in [
            self.menu,
            self.sequence_view,
            self.annotation
        ]:
            self.main_layout.addWidget(x)

        for w in [
            self.main,
        ]:
            self.centralLayout.addWidget(w)

        self.copy_annotation = QShortcut(QKeySequence('Ctrl+c'), self.centralWidget)
        self.paste_annotation = QShortcut(QKeySequence('Ctrl+v'), self.centralWidget)
        self.reset_annotation = QShortcut(QKeySequence('Ctrl+r'), self.centralWidget)
        self.infer_annotation = QShortcut(QKeySequence('Ctrl+s'), self.centralWidget)

        self.centralWidget.setLayout(self.centralLayout)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslate_ui()

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self):
        _translate = QCoreApplication.translate
        self.open_folder.setText(_translate("Ball Annotator", "Open"))
        self.prev_frame.setText(_translate("Ball Annotator", "<"))
        self.next_frame.setText(_translate("Ball Annotator", ">"))
        self.save.setText(_translate("Ball Annotator", "Save"))


class MainWindowUI(UI):

    def __init__(self):
        self.model = Model()
        self.controller = Controller(self.model)

        super().__init__()

    def setupUI(self, MainWindow):
        super().setupUI(MainWindow)
        self.ui = MainWindow

        self.open_folder.clicked.connect(self.open_folder_clicked)
        self.prev_frame.clicked.connect(self.prev_frame_clicked)
        self.next_frame.clicked.connect(self.next_frame_clicked)
        self.save.clicked.connect(self.save_clicked)

        self.controller.add_view(self)
        self.controller.add_view(self.sequence_view)
        self.controller.add_view(self.frame_view)

        self.copy_annotation.activated.connect(self.on_copy)
        self.paste_annotation.activated.connect(self.on_paste)
        self.reset_annotation.activated.connect(self.on_reset)
        self.infer_annotation.activated.connect(self.on_infer)

    def open_folder_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly
        options |= QFileDialog.DontResolveSymlinks
        folder_path = QFileDialog.getExistingDirectory(
            self.ui,
            "Open Folder",
            "/home/andrii/Desktop/diploma_data",
            options=options)

        if folder_path:
            self.controller.open_folder(folder_path)

    @pyqtSlot()
    def on_copy(self):
        self.controller.copy_annotation()

    @pyqtSlot()
    def on_paste(self):
        self.controller.paste_annotation()

    @pyqtSlot()
    def on_reset(self):
        self.controller.reset_annotation()

    @pyqtSlot()
    def on_infer(self):
        self.controller.infer_annotation()

    def next_frame_clicked(self):
        self.controller.next_frame()

    def prev_frame_clicked(self):
        self.controller.prev_frame()

    def save_clicked(self):
        self.controller.save_annotation()

    def delete_clicked(self):
        self.controller.delete_keypoint()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.prev_frame_clicked()
        elif e.key() == Qt.Key_Right:
            self.next_frame_clicked()
        elif e.key() == Qt.Key_Space:
            self.save_clicked()
        elif e.key() == Qt.Key_Delete:
            self.delete_clicked()

    def mouseClickEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            pass

    def set_model(self, model):
        pass

    def set_controller(self, controller):
        pass

    def update_view(self, hint, data=None):
        pass


def main():

    app = Application(sys.argv)
    app.setStyle('Oxygen')
    MainWindow = QMainWindow()
    ui = MainWindowUI()
    ui.setupUI(MainWindow)
    app.key_pressed_signal.connect(ui.keyPressEvent)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()