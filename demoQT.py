import sys
import cv2
import numpy as np
import datetime
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy, QMessageBox, QDialog,
    QSlider, QColorDialog, QGridLayout, QDoubleSpinBox, QLabel as QLabel2
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
from collections import Counter
import configparser


CONFIG_FILE = "config.ini"


class SettingsDialog(QDialog):
    def __init__(self, parent=None, box_color=(0, 173, 181), box_thickness=2, font_scale=0.8, model_path="", conf_threshold=0.7):
        super().__init__(parent)
        self.setWindowTitle("检测设置")
        self.setFixedSize(400, 360)

        self.box_color = box_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        layout = QGridLayout()

        # 颜色选择按钮
        self.color_btn = QPushButton("选择框颜色")
        self.color_display = QLabel()
        self.color_display.setFixedSize(50, 25)
        self.color_display.setStyleSheet(f"background-color: rgb{self.box_color}; border: 1px solid black;")

        layout.addWidget(self.color_btn, 0, 0)
        layout.addWidget(self.color_display, 0, 1)

        # 线宽滑条
        layout.addWidget(QLabel("线宽："), 1, 0)
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(10)
        self.thickness_slider.setValue(self.box_thickness)
        layout.addWidget(self.thickness_slider, 1, 1)

        # 字体大小滑条
        layout.addWidget(QLabel("文字大小："), 2, 0)
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setMinimum(5)    # 实际缩放 0.5
        self.font_slider.setMaximum(30)   # 3.0
        self.font_slider.setValue(int(self.font_scale * 10))
        layout.addWidget(self.font_slider, 2, 1)

        # 置信度设置
        layout.addWidget(QLabel2("置信度阈值："), 3, 0)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(self.conf_threshold)
        layout.addWidget(self.conf_spinbox, 3, 1)

        # 选择模型权重按钮和标签
        self.model_btn = QPushButton("选择模型权重")
        self.model_label = QLabel(os.path.basename(self.model_path) if self.model_path else "未选择模型")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_btn, 4, 0)
        layout.addWidget(self.model_label, 4, 1)

        # 按钮
        self.btn_ok = QPushButton("确定")
        self.btn_cancel = QPushButton("取消")
        layout.addWidget(self.btn_ok, 5, 0)
        layout.addWidget(self.btn_cancel, 5, 1)

        self.setLayout(layout)

        # 信号
        self.color_btn.clicked.connect(self.choose_color)
        self.model_btn.clicked.connect(self.choose_model)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.box_color = (color.red(), color.green(), color.blue())
            self.color_display.setStyleSheet(f"background-color: rgb{self.box_color}; border: 1px solid black;")

    def choose_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择模型权重文件", "", "模型文件 (*.pt *.pth)")
        if fname:
            self.model_path = fname
            self.model_label.setText(os.path.basename(fname))

    def get_values(self):
        return (
            self.box_color,
            self.thickness_slider.value(),
            self.font_slider.value() / 10.0,
            self.model_path,
            self.conf_spinbox.value()
        )


class FishDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐟 鱼类检测系统 - YOLOv11")
        self.setGeometry(100, 100, 1000, 720)

        # 读取配置
        self.config = configparser.ConfigParser()
        self.load_config()

        # 载入模型，优先加载配置文件指定路径
        self.load_model(self.model_path)

        # 参数初始化
        self.box_color = self.config_gettuple("Settings", "box_color", (0, 173, 181))
        self.box_thickness = self.config_getint("Settings", "box_thickness", 2)
        self.font_scale = self.config_getfloat("Settings", "font_scale", 0.8)
        self.conf_threshold = self.config_getfloat("Settings", "conf_threshold", 0.7)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("🐟目标检测")
        self.image_label.setFont(QFont("微软雅黑", 60, QFont.Bold))
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #222831;
                color: #EEEEEE;
                border: 2px solid #00ADB5;
                border-radius: 10px;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_fish_count = QLabel("检测数量：0")
        self.label_fish_count.setFont(QFont("微软雅黑", 16))
        self.label_fish_count.setStyleSheet("color: #00ADB5;")
        self.label_fish_count.setAlignment(Qt.AlignCenter)

        button_style = """
            QPushButton {
                background-color: #00ADB5;
                color: white;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #019ca1;
            }
            QPushButton:disabled {
                background-color: #6c6c6c;
                color: #cccccc;
            }
        """

        self.btn_load_image = QPushButton("导入图片")
        self.btn_load_image.setStyleSheet(button_style)

        self.btn_load_video = QPushButton("导入视频")
        self.btn_load_video.setStyleSheet(button_style)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setStyleSheet(button_style)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.setStyleSheet(button_style)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.setStyleSheet(button_style)

        self.btn_settings = QPushButton("设置")
        self.btn_settings.setStyleSheet(button_style)

        self.btn_pause.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_save.setEnabled(False)

        h_layout1 = QHBoxLayout()
        h_layout1.setSpacing(15)
        h_layout1.addWidget(self.btn_load_image)
        h_layout1.addWidget(self.btn_load_video)
        h_layout1.addWidget(self.btn_pause)
        h_layout1.addWidget(self.btn_clear)
        h_layout1.addWidget(self.btn_settings)

        h_layout2 = QHBoxLayout()
        h_layout2.addStretch()
        h_layout2.addWidget(self.btn_save)
        h_layout2.addStretch()

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(15, 15, 15, 15)
        v_layout.setSpacing(20)
        v_layout.addWidget(self.image_label, stretch=8)
        v_layout.addWidget(self.label_fish_count, stretch=1)
        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear.clicked.connect(self.clear_display)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_settings.clicked.connect(self.open_settings_dialog)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_paused = False
        self.detected_fish_counter = Counter()
        self.video_writer = None
        self.video_save_path = None

        self.current_frame = None
        self.video_saved = False
        self.current_mode = None

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            self.config.read(CONFIG_FILE)
        else:
            self.config['Settings'] = {}

        self.model_path = self.config['Settings'].get('model_path', r"E:\DSX\UnderwaterFishDetection\runs\detect\yolo11\weights\best.pt")

    def save_config(self):
        # 保存当前设置
        if 'Settings' not in self.config:
            self.config['Settings'] = {}
        self.config['Settings']['model_path'] = self.model_path
        self.config['Settings']['box_color'] = ",".join(map(str, self.box_color))
        self.config['Settings']['box_thickness'] = str(self.box_thickness)
        self.config['Settings']['font_scale'] = str(self.font_scale)
        self.config['Settings']['conf_threshold'] = str(self.conf_threshold)

        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)

    def config_gettuple(self, section, option, default):
        try:
            val = self.config.get(section, option)
            return tuple(map(int, val.split(',')))
        except:
            return default

    def config_getint(self, section, option, default):
        try:
            return self.config.getint(section, option)
        except:
            return default

    def config_getfloat(self, section, option, default):
        try:
            return self.config.getfloat(section, option)
        except:
            return default

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.class_names = self.model.names
            self.model_path = path
            self.statusBar().showMessage(f"已加载模型：{os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "加载模型失败", f"加载模型权重失败：{e}")
            self.model = None
            self.class_names = {}

    def detect_and_draw(self, frame):
        self.detected_fish_counter.clear()
        if not self.model:
            return frame
        results = self.model(frame, stream=True)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                if conf[i] >= self.conf_threshold:
                    cls_id = int(class_[i])
                    cls_name = self.class_names.get(cls_id, "未知")
                    self.detected_fish_counter[cls_name] += 1

                    x1, y1, x2, y2 = map(int, boxes[i])
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2

                    cv2.circle(frame, (mid_x, mid_y), 5, self.box_color, -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
                    cv2.putText(frame, f"{cls_name} : {conf[i]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.box_color, self.box_thickness)

        count_text = "检测数量： " + ", ".join(f"{k}: {v}" for k, v in self.detected_fish_counter.items())
        if not self.detected_fish_counter:
            count_text = "检测数量：0"
        self.label_fish_count.setText(count_text)
        return frame

    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qimg.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开图片文件", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.current_mode = "image"
            self.cap = None
            self.timer.stop()
            self.is_paused = False
            img = cv2.imread(fname)
            if img is None:
                return
            img = self.detect_and_draw(img)
            self.display_image(img)
            self.current_frame = img.copy()
            self.btn_pause.setEnabled(False)
            self.btn_clear.setEnabled(True)
            self.btn_save.setEnabled(True)

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if fname:
            self.current_mode = "video"
            self.cap = cv2.VideoCapture(fname)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "打开视频失败", "无法打开视频文件！")
                return
            self.is_paused = False
            self.btn_pause.setEnabled(True)
            self.btn_pause.setText("暂停")
            self.btn_clear.setEnabled(True)
            self.btn_save.setEnabled(False)  # 视频保存等待手动开始保存
            self.timer.start(30)
            self.video_save_path = None
            self.video_writer = None
            self.video_saved = False

    def update_frame(self):
        if self.cap is None or self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.btn_pause.setEnabled(False)
            self.btn_save.setEnabled(self.video_saved)
            return

        frame = self.detect_and_draw(frame)

        # 保存视频
        if self.video_writer is not None:
            self.video_writer.write(frame)
            self.video_saved = True
            self.btn_save.setEnabled(True)

        self.current_frame = frame.copy()
        self.display_image(frame)

    def toggle_pause(self):
        if self.cap is None:
            return
        if self.is_paused:
            self.is_paused = False
            self.timer.start(30)
            self.btn_pause.setText("暂停")
            # 开始保存视频（如果之前没保存）
            if self.video_writer is None:
                self.start_video_save()
        else:
            self.is_paused = True
            self.timer.stop()
            self.btn_pause.setText("继续")

    def clear_display(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image_label.clear()
        self.image_label.setText("🐟目标检测")
        self.label_fish_count.setText("检测数量：0")
        self.btn_pause.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.current_frame = None
        self.is_paused = False
        self.video_save_path = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.video_saved = False

    def save_result(self):
        if self.current_frame is None:
            return

        if self.current_mode == "image":
            fname, _ = QFileDialog.getSaveFileName(self, "保存图片", f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
            if fname:
                cv2.imwrite(fname, self.current_frame)
                QMessageBox.information(self, "保存成功", f"图片已保存到:\n{fname}")

        elif self.current_mode == "video":
            if self.video_save_path and os.path.exists(self.video_save_path):
                QMessageBox.information(self, "保存成功", f"视频已保存到:\n{self.video_save_path}")
            else:
                QMessageBox.warning(self, "保存失败", "视频保存失败或未生成。")

    def start_video_save(self):
        if self.cap is None or self.current_frame is None:
            return
        # 自动生成视频保存路径
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存视频文件夹", os.getcwd())
        if not dir_path:
            return
        self.video_save_path = os.path.join(dir_path, f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w, _ = self.current_frame.shape
        self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, 25, (w, h))

    def open_settings_dialog(self):
        dlg = SettingsDialog(
            self,
            box_color=self.box_color,
            box_thickness=self.box_thickness,
            font_scale=self.font_scale,
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
        )
        if dlg.exec():
            self.box_color, self.box_thickness, self.font_scale, model_path, conf_threshold = dlg.get_values()
            self.conf_threshold = conf_threshold
            if model_path and model_path != self.model_path:
                self.load_model(model_path)
            self.save_config()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FishDetectorApp()
    window.show()
    sys.exit(app.exec())
