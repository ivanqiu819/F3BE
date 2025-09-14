import sys
import subprocess
import os
import json
import tqdm
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QDialog,
    QFileDialog, QMessageBox, QStackedWidget, QTextEdit, QCheckBox,
    QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QFont, QPainter, QColor, QBrush

# UI Constants for consistent styling
MAIN_LAYOUT_SPACING = 20
ACTION_BUTTON_SPACING = 15
BUTTON_MIN_HEIGHT = 45
BUTTON_MIN_WIDTH = 120
HELP_BUTTON_SIZE = (100, 35)

GLOBAL_STYLESHEET = """
QWidget, QDialog {
    background-color: #f7f7f7; font-family: Arial, sans-serif;
}
QLineEdit, QPushButton, QLabel, QTextEdit, QSpinBox { font-size: 14px; }
QLineEdit, QTextEdit, QSpinBox {
    background-color: #ffffff; border: 1px solid #ccc; border-radius: 4px; padding: 8px;
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus { border-color: #0078d7; }
QLineEdit:read-only { background-color: #e9e9e9; }
QPushButton {
    background-color: #e1e1e1; color: #111; border: none;
    border-radius: 5px; padding: 10px 15px;
    font-weight: bold;
}
QPushButton:hover { background-color: #d1d1d1; }
QPushButton:pressed { background-color: #c1c1c1; }
QPushButton#HelpBtn {
    background-color: #6c757d; color: white;
}
QPushButton#HelpBtn:hover { background-color: #5a6268; }
QPushButton#ActionBtnGreen {
    background-color: #28a745; color: white;
}
QPushButton#ActionBtnGreen:hover { background-color: #218838; }
QPushButton#ActionBtnBlue {
    background-color: #007bff; color: white;
}
QPushButton#ActionBtnBlue:hover { background-color: #0069d9; }
"""


# --- Custom Toggle Switch Widget ---
class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.setChecked(not self.isChecked())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        handle_radius = height / 2 - 2
        if self.isChecked():
            bg_color = QColor("#007bff")
            handle_pos = QPoint(int(width - handle_radius) - 2, int(height / 2))
        else:
            bg_color = QColor("#ccc")
            handle_pos = QPoint(int(handle_radius + 2), int(height / 2))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(0, 0, width, height, height / 2, height / 2)
        painter.setBrush(QBrush(Qt.white))
        painter.drawEllipse(handle_pos, handle_radius, handle_radius)


class ConfigManager:
    def get_default_structure(self, cam_num):
        structure = {"calibration":
                         {"intrinsic": {}, "extrinsic": {}},
                     "experiment": {}}
        for i in range(cam_num):
            device_key = f"device{i}"
            structure["experiment"][device_key] = {"left_camera": {}, "right_camera": {}, "annotation": {}}
            structure["calibration"]["intrinsic"][device_key] = {"left": {}, "right": {}}
            structure["calibration"]["extrinsic"][device_key] = {}
        return structure

    def create_project_structure(self, base_path, structure_dict):
        for name, content in structure_dict.items():
            current_path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(current_path, exist_ok=True)
                self.create_project_structure(current_path, content)

    def generate_config(self, project_name, experimenter, cam_num):
        ref = {}
        for i in range(cam_num):
            left = (i - 1) % cam_num
            right = (i + 1) % cam_num
            ref[f"device{i}"] = [f"device{left}", f"device{right}"]
        config_data = {
            "Env_Path": "/path/to/python",
            "Project_Info": {
                "Project_Name": project_name,
                "Experimenter": experimenter,
                "Number_of_Devices": cam_num
            },
            "Calibration_Info": {
                "Intrinsic": {
                    "Checkerboard_Corner_Spacing": [30, 30],
                    "Inner_Corner_Array_Size": [11, 8]
                },
                "Extrinsic": {
                    "April_Grid_Size": 6,
                    "April_Size": 0.0352,
                    "April_Interval": 0.01056,
                    "April_Family": "t36h11"
                },
            },
            "Depth_Computation": {
                "Env_Path": "/path/to/python",
                "Ckpt_Path": "/path/to/pretrained_model.pth"
            },
            "Animal_Detection": {
                "Env_Path": "/path/to/python",
                "Ckpt_Path": "/path/to/pretrained_model.pth"
            },
            "Embedding_Extraction": {
                "Env_Path": "/path/to/python",
                "Ckpt_Path": "/path/to/pretrained_model.pth",
                "mask_mix_ratio": 1.0,
                "rotate_num": 16,
                "centralize": "true",
                "normalize": "true" 
            },
            "Reference_Devices": ref
        }
        return config_data


# --- "Extrinsic Calibration" Page ---
class ExtrinsicCalibPage(QWidget):
    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        file_select_layout = QHBoxLayout()
        self.file_path_display = QLineEdit()
        self.file_path_display.setPlaceholderText("Please select extrinsic calibration folder...")
        self.file_path_display.setReadOnly(True)
        self.select_file_btn = QPushButton("Select folder")
        self.select_file_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        file_select_layout.addWidget(QLabel("Extrinsic calibration folder:"))
        file_select_layout.addWidget(self.file_path_display)
        file_select_layout.addWidget(self.select_file_btn)
        
        layout.addLayout(file_select_layout)
        layout.addStretch()
        
        action_layout = QHBoxLayout()
        action_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.help_btn = QPushButton("Help")
        self.help_btn.setObjectName("HelpBtn")
        self.help_btn.setMinimumSize(*HELP_BUTTON_SIZE)
        action_layout.addWidget(self.help_btn)
        action_layout.addStretch()
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setObjectName("ActionBtnGreen")
        self.calibrate_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        action_layout.addWidget(self.calibrate_btn)
        layout.addLayout(action_layout)
        self.select_file_btn.clicked.connect(self.select_file)
        self.calibrate_btn.clicked.connect(self.run_calibration)
        self.help_btn.clicked.connect(self.show_help)

    def select_file(self):
        file_path = QFileDialog.getExistingDirectory(self, "Select extrinsic calibration folder", "")
        if file_path:
            self.file_path_display.setText(file_path)

    def run_calibration(self):
        selected_file = self.file_path_display.text()
        if not selected_file:
            QMessageBox.warning(self, "Warning", "Please select a folder before extrinsic calibration!")
            return
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            config = json.load(open(os.path.join(self.project_path, "config.json"), "r", encoding='utf-8-sig'))
            python_interpreter = config["Env_Path"]
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsic_calibration.py")
            result = subprocess.run([python_interpreter, script_path, selected_file], check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    creationflags=creation_flags)
            output = result.stdout
            QMessageBox.information(self, "Success",
                                    f"Extrinsic calibration for folder '{selected_file}' completed!\nOutput:\n{output}")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Extrinsic calibration failed!\nError:\n{e.stderr}")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Execution failed!\nError: 'python' command or 'extrinsic_calibration.py' script not found.")

    def show_help(self):
        QMessageBox.information(self, "Extrinsic Calibration Help",
                                "This page is for extrinsic calibration.\n\n1."
                                " Put images of AprilTag Grid from different cameras into the extrinsic folder.\n2."
                                " Click 'Select folder' to choose the folder.\n3. Click 'Calibrate' to process and write results to file.")


# --- "Intrinsic Calibration" Page ---
class IntrinsicCalibPage(QWidget):
    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.project_path = project_path

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        intrinsic_file_select_layout = QHBoxLayout()
        self.intrinsic_file_path_display = QLineEdit()
        self.intrinsic_file_path_display.setPlaceholderText("Please select intrinsic calibration folder...")
        self.intrinsic_file_path_display.setReadOnly(True)
        self.intrinsic_select_file_btn = QPushButton("Select folder")
        self.intrinsic_select_file_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        intrinsic_file_select_layout.addWidget(QLabel("Intrinsic calibration folder:"))
        intrinsic_file_select_layout.addWidget(self.intrinsic_file_path_display)
        intrinsic_file_select_layout.addWidget(self.intrinsic_select_file_btn)
        
        layout.addLayout(intrinsic_file_select_layout)
        layout.addStretch()
        
        intrinsic_action_layout = QHBoxLayout()
        intrinsic_action_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.intrinsic_help_btn = QPushButton("Help")
        self.intrinsic_help_btn.setObjectName("HelpBtn")
        self.intrinsic_help_btn.setMinimumSize(*HELP_BUTTON_SIZE)
        intrinsic_action_layout.addWidget(self.intrinsic_help_btn)
        intrinsic_action_layout.addStretch()
        self.intrinsic_calibrate_btn = QPushButton("Calibrate")
        self.intrinsic_calibrate_btn.setObjectName("ActionBtnGreen")
        self.intrinsic_calibrate_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        intrinsic_action_layout.addWidget(self.intrinsic_calibrate_btn)
        layout.addLayout(intrinsic_action_layout)
        self.intrinsic_select_file_btn.clicked.connect(self.select_intrinsic_file)
        self.intrinsic_calibrate_btn.clicked.connect(self.run_intrinsic_calibration)
        self.intrinsic_help_btn.clicked.connect(self.show_intrinsic_help)

    def select_intrinsic_file(self):
        file_path = QFileDialog.getExistingDirectory(self, "Select intrinsic calibration folder", "")
        if file_path:
            self.intrinsic_file_path_display.setText(file_path)

    def run_intrinsic_calibration(self):
        selected_file = self.intrinsic_file_path_display.text()
        if not selected_file:
            QMessageBox.warning(self, "Warning", "Please select a folder before intrinsic calibration!")
            return
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            config = json.load(open(os.path.join(self.project_path, "config.json"), "r", encoding='utf-8-sig'))
            python_interpreter = config["Env_Path"]
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stereo_calibration.py")
            result = subprocess.run([python_interpreter, script_path, selected_file], check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
            output = result.stdout
            QMessageBox.information(self, "Success",
                                    f"Intrinsic calibration for folder '{selected_file}' completed!\nOutput:\n{output}")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Intrinsic calibration failed!\nError:\n{e.stderr}")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Execution failed!\nError: 'python' command or 'stereo_calibration.py' script not found.")

    def show_intrinsic_help(self):
        QMessageBox.information(self, "Intrinsic Calibration Help",
                                "This page is for intrinsic calibration.\n\n1."
                                " Put checkerboard images taken from different angles into a folder.\n2."
                                " Click 'Select folder' to choose the folder.\n3. Click 'Calibrate' to process and write results to file.")


# --- "Depth computation" Page ---
class DepthComputationPage(QWidget):
    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.project_path = project_path

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        path_select_layout = QHBoxLayout()
        self.seq_path_display = QLineEdit()
        self.seq_path_display.setPlaceholderText("Please select sequence path...")
        self.seq_path_display.setReadOnly(True)
        self.select_path_btn = QPushButton("Select path")
        self.select_path_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        path_select_layout.addWidget(QLabel("Sequence path:"))
        path_select_layout.addWidget(self.seq_path_display)
        path_select_layout.addWidget(self.select_path_btn)

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.btn_generate = QPushButton("Rectify")
        self.btn_generate.setObjectName("ActionBtnGreen")
        self.btn_generate.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        self.btn_generate_2 = QPushButton("Compute")
        self.btn_generate_2.setObjectName("ActionBtnGreen")
        self.btn_generate_2.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        action_buttons_layout.addWidget(self.btn_generate)
        action_buttons_layout.addWidget(self.btn_generate_2)
        action_buttons_layout.addStretch()

        layout.addLayout(path_select_layout)
        layout.addLayout(action_buttons_layout)
        layout.addStretch()
        
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.help_btn = QPushButton("Help")
        self.help_btn.setObjectName("HelpBtn")
        self.help_btn.setMinimumSize(*HELP_BUTTON_SIZE)
        bottom_layout.addWidget(self.help_btn)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)
        self.select_path_btn.clicked.connect(self.select_sequence_path)
        self.btn_generate.clicked.connect(self.run_rectify)
        self.btn_generate_2.clicked.connect(self.run_generate_depth_map)
        self.help_btn.clicked.connect(self.show_help)

    def select_sequence_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select sequence path", "")
        if dir_path:
            self.seq_path_display.setText(dir_path)


    def _validate_path(self):
        path = self.seq_path_display.text()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return None
        return path

    def run_rectify(self):
        sequence_dir = self._validate_path()
        if sequence_dir:
            try:
                script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/FoundationStereo/scripts/rectify_and_compute_depth.py"
                script_path = os.path.normpath(script_path)
                stereo_parameter_path = os.path.join(self.project_path, "stereo_parameter.json")
                stereo_parameter_path = os.path.normpath(stereo_parameter_path)
                json_path = os.path.join(self.project_path, "config.json")
                with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                    data = json.load(json_file)
                device_num = data["Project_Info"]["Number_of_Devices"]
                python_interpreter = data["Depth_Computation"]["Env_Path"]
                ckpt_path = data["Depth_Computation"]["Ckpt_Path"]
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

                command_list = [python_interpreter, str(script_path),
                                "--stereo_parameters", str(stereo_parameter_path), "--base_path", str(sequence_dir),
                                "--num_devices", str(device_num), "--rectify", "--ckpt_path", str(ckpt_path),
                                ]
                subprocess.run(command_list, check=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
                QMessageBox.information(self, "Success", "Rectification finish!")
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Error", f"Rectification failed!\nError:\n{e.stderr}")

    def run_generate_depth_map(self):
        sequence_dir = self._validate_path()
        if sequence_dir:
            try:
                script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/FoundationStereo/scripts/rectify_and_compute_depth.py"
                script_path = os.path.normpath(script_path)
                stereo_parameter_path = os.path.join(self.project_path, "stereo_parameter.json")
                stereo_parameter_path = os.path.normpath(stereo_parameter_path)
                json_path = os.path.join(self.project_path, "config.json")
                with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                    data = json.load(json_file)
                device_num = data["Project_Info"]["Number_of_Devices"]
                python_interpreter = data["Depth_Computation"]["Env_Path"]
                ckpt_path = data["Depth_Computation"]["Ckpt_Path"]
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

                command_list = [python_interpreter, str(script_path),
                                "--stereo_parameters", str(stereo_parameter_path), "--base_path", str(sequence_dir),
                                "--num_devices", str(device_num), "--ckpt_path", str(ckpt_path),
                                ]
                subprocess.run(command_list, check=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
                QMessageBox.information(self, "Success", "Depth map generation finish!")
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Error", f"Depth map generation failed!\nError:\n{e.stderr}")


    def show_help(self):
        QMessageBox.information(self, "Depth Computation Help",
                                "This page is for generating depth map\n"
                                "**Compute**: Click to compute depth maps.Choose the floder experiment")


# --- "Animal Detection" Page ---
class AnimalDetectionPage(QWidget):
    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.project_path = project_path

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        path_select_layout = QHBoxLayout()
        self.seq_path_display = QLineEdit()
        self.seq_path_display.setPlaceholderText("Please select sequence path...")
        self.seq_path_display.setReadOnly(True)
        self.select_path_btn = QPushButton("Select path")
        self.select_path_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        path_select_layout.addWidget(QLabel("Sequence path:"))
        path_select_layout.addWidget(self.seq_path_display)
        path_select_layout.addWidget(self.select_path_btn)
        
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.btn_first_frame = QPushButton("Annotate first frame")
        self.btn_first_frame.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        self.btn_mask_extract = QPushButton("Detect")
        self.btn_mask_extract.setObjectName("ActionBtnGreen")
        self.btn_mask_extract.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        action_buttons_layout.addWidget(self.btn_first_frame)
        action_buttons_layout.addWidget(self.btn_mask_extract)
        action_buttons_layout.addStretch()
        
        layout.addLayout(path_select_layout)
        layout.addLayout(action_buttons_layout)
        layout.addStretch()
        
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.help_btn = QPushButton("Help")
        self.help_btn.setObjectName("HelpBtn")
        self.help_btn.setMinimumSize(*HELP_BUTTON_SIZE)
        bottom_layout.addWidget(self.help_btn)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)
        self.select_path_btn.clicked.connect(self.select_sequence_path)
        self.btn_first_frame.clicked.connect(self.run_first_frame_annotation)
        self.btn_mask_extract.clicked.connect(self.run_mask_extraction)
        self.help_btn.clicked.connect(self.show_help)

    def select_sequence_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select sequence path", "")
        if dir_path:
            self.seq_path_display.setText(dir_path)

    def _validate_path(self):
        path = self.seq_path_display.text()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return None
        return path

    def run_first_frame_annotation(self):
        selected_file = self._validate_path()
        if selected_file:
            try:
                json_path = os.path.join(self.project_path, "config.json")
                with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                    data = json.load(json_file)
                python_interpreter = data["Animal_Detection"]["Env_Path"]
                ckpt_path = data["Animal_Detection"]["Ckpt_Path"]
                script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/sam2/tools/annotation.py"
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                device_num = data["Project_Info"]["Number_of_Devices"]
                for device_idx in range(device_num):
                    device_dir = os.path.join(selected_file, f"device{device_idx}")
                    if not os.path.exists(device_dir):
                        QMessageBox.warning(self, "Warning", f"Device directory '{device_dir}' does not exist!")
                        return
                    command_list = [python_interpreter, str(script_path),
                                    "--device_dir", str(device_dir), "--sam2_checkpoint", str(ckpt_path)]
                    subprocess.run(command_list, check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
                QMessageBox.information(self, "Success", f"First frame annotation for '{selected_file}' completed!")

            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Error", f"Calibration failed!\nError:\n{e.stderr}")
            except FileNotFoundError:
                QMessageBox.critical(self, "Error",
                                     "Execution failed!\nError: 'python' command or 'generate_frist_mask.py' script not found.")

        else:
            QMessageBox.warning(self, "Warning", "No path received")

    def run_mask_extraction(self):
        selected_file = self._validate_path()
        if not selected_file:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return
        try:
            json_path = os.path.join(self.project_path, "config.json")
            with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                data = json.load(json_file)
            python_interpreter = data["Animal_Detection"]["Env_Path"]
            ckpt_path = data["Animal_Detection"]["Ckpt_Path"]
            device_num = data["Project_Info"]["Number_of_Devices"]
            stereo_parameter_path = os.path.join(self.project_path, "stereo_parameter.json")
            stereo_parameter_path = os.path.normpath(stereo_parameter_path)
            script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/sam2/tools/easy_refine.py"
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            command_list = [python_interpreter, str(script_path), "--base_path", str(selected_file),
                            "--sam2_checkpoint", str(ckpt_path), "--device_number", str(device_num),
                            "--parameter_file", str(stereo_parameter_path),"--config_file",str(json_path)]
            subprocess.run(command_list, check=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
            QMessageBox.information(self, "Success", "Animal detection executed successfully!")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Animal detection failed!\nError:\n{e.stderr}")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Execution failed!\nError: 'python' command or 'easy_refine.py' script not found.")




    def show_help(self):
        help_text = """This page is for SAM2 related tasks.
        - **Annotate first frame**: Process the first frame of the sequence.
        Instructions:
        - Left mouse button: add point
        - Number keys 1-4: switch mouse ID
        - Key '0': global negative sample
        - Key 'r': reset all points
        - Key 'q': quit and return results

        - **Detect**: Extract mask from images.
        """
        QMessageBox.information(self, "Animal Detection Help", help_text)


# --- "Embedding Extraction" Page ---
class EmbeddingExtractionPage(QWidget):
    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.project_path = project_path

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        # main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        path_select_layout = QHBoxLayout()
        self.seq_path_display = QLineEdit()
        self.seq_path_display.setPlaceholderText("Please select sequence path...")
        self.seq_path_display.setReadOnly(True)
        self.select_path_btn = QPushButton("Select path")
        self.select_path_btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        path_select_layout.addWidget(QLabel("Sequence path:"))
        path_select_layout.addWidget(self.seq_path_display)
        path_select_layout.addWidget(self.select_path_btn)
        
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.btn_gen_pcd = QPushButton("Generate point clouds")
        self.btn_gen_pcd.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        self.btn_embedding_extract = QPushButton("Extract embeddings")
        self.btn_embedding_extract.setObjectName("ActionBtnGreen")
        self.btn_embedding_extract.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        self.btn_visualize = QPushButton("UMAP and Visualize")
        self.btn_visualize.setObjectName("ActionBtnBlue")
        self.btn_visualize.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
        action_buttons_layout.addWidget(self.btn_gen_pcd)
        action_buttons_layout.addWidget(self.btn_embedding_extract)
        action_buttons_layout.addWidget(self.btn_visualize)
        action_buttons_layout.addStretch()
        
        help_layout = QHBoxLayout()
        help_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.btn_help = QPushButton("Help")
        self.btn_help.setObjectName("HelpBtn")
        self.btn_help.setMinimumSize(*HELP_BUTTON_SIZE)
        help_layout.addWidget(self.btn_help)
        help_layout.addStretch()
        
        main_layout.addLayout(path_select_layout)
        main_layout.addLayout(action_buttons_layout)
        main_layout.addStretch()
        main_layout.addLayout(help_layout)
        self.select_path_btn.clicked.connect(self.select_sequence_path)
        self.btn_gen_pcd.clicked.connect(self.run_gen_pcd)
        self.btn_embedding_extract.clicked.connect(self.run_feature_extraction)
        self.btn_visualize.clicked.connect(self.run_umap_visualization)
        self.btn_help.clicked.connect(self.show_help)

    def select_sequence_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select sequence path", "")
        if dir_path:
            self.seq_path_display.setText(dir_path)

    def run_gen_pcd(self):
        base_path = self.seq_path_display.text()
        if not base_path:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return
        try:
            json_path = os.path.join(self.project_path, "config.json")
            with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                data = json.load(json_file)
            python_interpreter = data["Embedding_Extraction"]["Env_Path"]
            ckpt_path = data["Embedding_Extraction"]["Ckpt_Path"]
            num_devices = data["Project_Info"]["Number_of_Devices"]
            mask_mix_ratio = data["Embedding_Extraction"].get("mask_mix_ratio", 0.0)

            parameter_path = os.path.join(self.project_path, "stereo_parameter.json")
            script_path = os.path.dirname(os.path.abspath(__file__)) + "/fuse_pointclouds.py"
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            command_list = [python_interpreter, str(script_path),
                            "--parameter_path", str(parameter_path),
                            "--base_path", str(base_path),
                            "--num_devices", str(num_devices),
                            "--mask_mix_ratio", str(mask_mix_ratio),
                            ]
            subprocess.run(command_list, check=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Point cloud generation failed!\nError:\n{e.stderr}")
        QMessageBox.information(self, "Success", "'Point cloud generation' executed successfully!")

    def run_feature_extraction(self):
        base_path = self.seq_path_display.text()
        if not base_path:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return
        try:
            json_path = os.path.join(self.project_path, "config.json")
            with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                data = json.load(json_file)
            python_interpreter = data["Embedding_Extraction"]["Env_Path"]
            ckpt_path = data["Embedding_Extraction"]["Ckpt_Path"]
            rotate_num = data["Embedding_Extraction"]["rotate_num"]
            centralize = data["Embedding_Extraction"]["centralize"]
            normalize = data["Embedding_Extraction"]["normalize"]

            script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Point-SAM/demo/embedding_extraction.py"
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            command_list = [python_interpreter, str(script_path),
                            "--base_path", str(base_path),
                            "--model_path", str(ckpt_path),
                            "--rotate_num", str(rotate_num),
                            "--centralize", str(centralize),
                            "--normalize", str(normalize)
                            ]
            subprocess.run(command_list, check=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Embedding extraction failed!\nError:\n{e.stderr}")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Execution failed!\nError: 'python' command or 'embedding_extraction.py' script not found.")
        except KeyError as e:
            QMessageBox.critical(self, "Error",
                                 f"Missing configuration key: {e}\nPlease check your config.json file.")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error",
                                 f"Invalid JSON format in config file: {e}\nPlease check your config.json file.")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"An unexpected error occurred: {e}\nPlease check your config.json file.")
        QMessageBox.information(self, "Success", "Embedding extraction executed successfully!")

    def run_umap_visualization(self):
        base_path = self.seq_path_display.text()
        if not base_path:
            QMessageBox.warning(self, "Warning", "Please select 'sequence path' first!")
            return
        try:
            json_path = os.path.join(self.project_path, "config.json")
            with open(json_path, 'r', encoding='utf-8-sig') as json_file:
                data = json.load(json_file)
            python_interpreter = data["Embedding_Extraction"]["Env_Path"]
            rotate_num = data["Embedding_Extraction"]["rotate_num"]
            centralize = data["Embedding_Extraction"]["centralize"]
            normalize = data["Embedding_Extraction"]["normalize"]

            script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Point-SAM/demo/UMAP_and_visualization.py"
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            command_list = [python_interpreter, str(script_path),
                            "--base_path", str(base_path),
                            "--rotate_num", str(rotate_num),
                            "--centralize", str(centralize),
                            "--normalize", str(normalize)]
            subprocess.run(command_list, check=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"UMAP and visualization failed!\nError:\n{e.stderr}")

    def show_help(self):
        QMessageBox.information(self, "Feature Extraction Help",
                                "This page is for feature extraction.\n\n- **Path**: Specify pointcloud file and output path.\n- **Toggle**: Choose whether to rotate, center, normalize, etc.")


# --- Project Workspace Main Interface ---
class ProjectWorkspace(QWidget):
    def __init__(self, project_path):
        super().__init__()
        self.project_path = project_path
        print(f"Workspace opened for: {self.project_path}")
        self.setWindowTitle(f"Workspace - {os.path.basename(project_path)}")
        self.initial_width, self.initial_height = 1000, 700
        self.BASE_FONT_SIZE = 14
        self.setGeometry(200, 200, self.initial_width, self.initial_height)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        top_button_layout = QHBoxLayout()
        top_button_layout.setSpacing(ACTION_BUTTON_SPACING)
        self.page_buttons_info = [
            ("Intrinsic Calibration", IntrinsicCalibPage),
            ("Extrinsic Calibration", ExtrinsicCalibPage),
            ("Depth Computation", DepthComputationPage),
            ("Animal Detection", AnimalDetectionPage),
            ("Embedding Extraction", EmbeddingExtractionPage),
        ]
        self.stacked_widget = QStackedWidget()
        for i, (name, page_class) in enumerate(self.page_buttons_info):
            btn = QPushButton(name)
            btn.setMinimumSize(BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT)
            top_button_layout.addWidget(btn)
            if page_class:
                page = page_class(self.project_path)
            else:
                page = QLabel(f"This is the '{name}' page")
                page.setAlignment(Qt.AlignCenter)
            self.stacked_widget.addWidget(page)
            btn.clicked.connect(lambda checked, index=i: self.stacked_widget.setCurrentIndex(index))
        top_button_layout.addStretch()
        main_layout.addLayout(top_button_layout)
        main_layout.addWidget(self.stacked_widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)



# --- New Project Dialog ---
class NewProjectDialog(QDialog):
    project_created = pyqtSignal(str)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("New Project")
        self.setMinimumSize(550, 400)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(ACTION_BUTTON_SPACING)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.proj_name_input = QLineEdit()
        self.experimenter_input = QLineEdit()
        self.num_cam_spinbox = QSpinBox()
        self.num_cam_spinbox.setMinimum(1)
        self.num_cam_spinbox.setMaximum(16)
        self.num_cam_spinbox.setValue(4)
        self.location_input = QLineEdit()
        self.location_input.setReadOnly(True)

        form_layout.addRow("Project name:", self.proj_name_input)
        form_layout.addRow("Experimenter:", self.experimenter_input)
        form_layout.addRow("Number of cameras:", self.num_cam_spinbox)

        location_layout = QHBoxLayout()
        location_layout.addWidget(self.location_input)
        location_layout.addWidget(QLabel("(Auto-generated based on above)"))
        form_layout.addRow("Location:", location_layout)

        bottom_buttons_layout = QHBoxLayout()
        self.create_btn = QPushButton("Create")
        bottom_buttons_layout.addWidget(QLabel("(Auto-generate folder and config file)"))
        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(self.create_btn)

        main_layout.addLayout(form_layout)
        main_layout.addStretch()
        main_layout.addLayout(bottom_buttons_layout)

        self.proj_name_input.textChanged.connect(self.update_location)
        self.experimenter_input.textChanged.connect(self.update_location)
        self.create_btn.clicked.connect(self.create_project)

    def update_location(self):
        proj_name = self.proj_name_input.text().strip()
        experimenter = self.experimenter_input.text().strip()
        if proj_name and experimenter:
            safe_proj_name = "".join(c for c in proj_name if c.isalnum() or c in (' ', '_')).rstrip()
            safe_experimenter = "".join(c for c in experimenter if c.isalnum() or c in (' ', '_')).rstrip()
            self.location_input.setText(f"{safe_proj_name}_{safe_experimenter}")
        else:
            self.location_input.clear()

    def create_project(self):
        location_name = self.location_input.text().strip()
        proj_name = self.proj_name_input.text().strip()
        experimenter = self.experimenter_input.text().strip()

        if not location_name or not proj_name or not experimenter:
            QMessageBox.warning(self, "Error", "Project Name and Experimenter cannot be empty!")
            return

        num_cameras = self.num_cam_spinbox.value()

        try:
            project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', location_name)
            project_path = os.path.normpath(project_path)

            if os.path.exists(project_path):
                raise FileExistsError(f"Folder '{location_name}' already exists.")

            structure = self.config_manager.get_default_structure(num_cameras)
            self.config_manager.create_project_structure(project_path, structure)

            config_data = self.config_manager.generate_config(proj_name, experimenter, num_cameras)

            with open(os.path.join(project_path, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)

            QMessageBox.information(self, "Success",
                                    f"Project created successfully with {num_cameras} devices!\nPath: {project_path}")
            self.project_created.emit(project_path)
            self.accept()

        except FileExistsError as e:
            QMessageBox.warning(self, "Creation Failed", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")


# --- Main Window ---
class MainWindow(QWidget):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.workspace_window = None
        self.initial_width, self.initial_height = 800, 500
        self.setGeometry(300, 300, self.initial_width, self.initial_height)
        self.BASE_TITLE_FONT_SIZE = 22
        self.BASE_BUTTON_FONT_SIZE = 16
        self.setWindowTitle("XXXGUI")
        self.init_ui()
        self.update_font_sizes()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(MAIN_LAYOUT_SPACING)
        
        self.title_label = QLabel("XXXGUI")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(ACTION_BUTTON_SPACING * 2)  # 更大的间距用于主窗口
        self.create_proj_btn = QPushButton("Create new project")
        self.load_proj_btn = QPushButton("Load project")
        self.create_proj_btn.setMinimumSize(BUTTON_MIN_WIDTH * 1.5, BUTTON_MIN_HEIGHT)  # 主窗口按钮稍大
        self.load_proj_btn.setMinimumSize(BUTTON_MIN_WIDTH * 1.5, BUTTON_MIN_HEIGHT)
        
        button_layout.addStretch(1)
        button_layout.addWidget(self.create_proj_btn, 2)
        button_layout.addWidget(self.load_proj_btn, 2)
        button_layout.addStretch(1)
        
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(button_layout)
        main_layout.addStretch(2)
        self.create_proj_btn.clicked.connect(self.open_new_project_dialog)
        self.load_proj_btn.clicked.connect(self.load_existing_project)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_font_sizes()

    def update_font_sizes(self):
        if self.initial_width == 0: return
        scale_factor = self.width() / self.initial_width
        new_title_size = max(12, int(self.BASE_TITLE_FONT_SIZE * scale_factor))
        new_button_size = max(10, int(self.BASE_BUTTON_FONT_SIZE * scale_factor))
        self.title_label.setFont(QFont("Arial", new_title_size, QFont.Bold))
        button_font = QFont("Arial", new_button_size, QFont.Bold)
        self.create_proj_btn.setFont(button_font)
        self.load_proj_btn.setFont(button_font)

    def open_new_project_dialog(self):
        dialog = NewProjectDialog(self.config_manager, self)
        dialog.project_created.connect(self.switch_to_workspace)
        dialog.exec_()

    def load_existing_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Load project folder", "")
        if dir_path:
            config_path = os.path.join(dir_path, "config.json")
            if os.path.exists(config_path):
                self.switch_to_workspace(dir_path)
            else:
                QMessageBox.warning(self, "Load Error",
                                    f"The selected folder '{os.path.basename(dir_path)}' does not appear to be a valid project (missing config.json).")

    def switch_to_workspace(self, project_path):
        self.workspace_window = ProjectWorkspace(project_path)
        self.workspace_window.show()
        self.close()


# --- Main Entry ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    config_manager = ConfigManager()
    main_win = MainWindow(config_manager)
    main_win.show()
    sys.exit(app.exec_())
