from absl import app
import collections

import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from unitree_api_bindings import unitree_api


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, window_size: int = 100):
        super().__init__()

        # Initialize Unitree API
        network_name = "enx7cc2c647de4f"
        control_rate_us = 100000  # 10 Hz
        self.unitree_driver = unitree_api.UnitreeDriver(
            network_name,
            control_rate_us,
        )
        self.unitree_driver.initialize()

        # Arguments:
        self.window_size = window_size

        # Initialize PyQt5 Window:
        self.setWindowTitle("Unitree Visualizer")
        self.setGeometry(100, 100, 1920, 1080)

        # Main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QGridLayout(main_widget)

        self.plot_widgets = []

        # IMU Data Visualization:
        graphics_layout_widget = pg.GraphicsLayoutWidget(show=True)
        self.plot_widgets.append(graphics_layout_widget)
        main_layout.addWidget(graphics_layout_widget, 0, 0)

        self.imu_main_plot_items = []
        self.imu_buffers = []
        for i, name in enumerate(["Accelerometer", "Gyroscope", "RPY"]):
            for j, axis in enumerate(["X", "Y", "Z"]):
                p = graphics_layout_widget.addPlot(row=i, col=j)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setTitle(f"{name} {axis}", color="k", size="20pt")
                

                x_deque = collections.deque(
                    np.linspace(0, 0, self.window_size),
                    maxlen=self.window_size,
                )
                y_deque = collections.deque(
                    [np.nan] * self.window_size,
                    maxlen=self.window_size,
                )

                plot_item = p.plot(
                    list(x_deque), list(y_deque),
                    pen=pg.mkPen(color=(100, 149, 237), width=2)
                )
                self.imu_main_plot_items.append(plot_item)
                self.imu_buffers.append({'x': x_deque, 'y': y_deque})

                if i == 0 and j == 0:
                    p.setLabel("left", "Acceleration (m/s²)", **{"color": "black", "font-size": "12pt"})
                elif i == 1 and j == 0:
                    p.setLabel("left", "Angular Velocity (rad/s)", **{"color": "black", "font-size": "12pt"})
                elif i == 2 and j == 0:
                    p.setLabel("left", "Angle (rad)", **{"color": "black", "font-size": "12pt"})

        shared_x_label_text = "Time (s)"
        label_item = pg.LabelItem(shared_x_label_text, size="11pt", bold=True, color='k')
        graphics_layout_widget.addItem(label_item, row=3, col=0, colspan=3)

        # Mtor Data Visualization: Joint Positions
        graphics_layout_widget = pg.GraphicsLayoutWidget(show=True)
        self.plot_widgets.append(graphics_layout_widget)
        main_layout.addWidget(graphics_layout_widget, 0, 1)

        self.joint_position_plot_items = []
        self.joint_position_buffers = []
        for i, leg in enumerate(["Front Right", "Front Left", "Hind Right", "Hind Left"]):
            for j, joint in enumerate(["Abduction", "Hip", "Knee"]):
                p = graphics_layout_widget.addPlot(row=i, col=j)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setTitle(f"{leg} {joint}", color="k", size="20pt")
                

                x_deque = collections.deque(
                    np.linspace(0, 0, self.window_size),
                    maxlen=self.window_size,
                )
                y_deque = collections.deque(
                    [np.nan] * self.window_size,
                    maxlen=self.window_size,
                )

                plot_item = p.plot(
                    list(x_deque), list(y_deque),
                    pen=pg.mkPen(color=(100, 149, 237), width=2)
                )
                self.joint_position_plot_items.append(plot_item)
                self.joint_position_buffers.append({'x': x_deque, 'y': y_deque})

                if i == 0 and j == 0:
                    p.setLabel("left", "Position (rad)", **{"color": "black", "font-size": "12pt"})
                elif i == 1 and j == 0:
                    p.setLabel("left", "Position (rad)", **{"color": "black", "font-size": "12pt"})
                elif i == 2 and j == 0:
                    p.setLabel("left", "Position (rad)", **{"color": "black", "font-size": "12pt"})
                elif i == 3 and j == 0:
                    p.setLabel("left", "Position (rad)", **{"color": "black", "font-size": "12pt"})
        
        shared_x_label_text = "Time (s)"
        label_item = pg.LabelItem(shared_x_label_text, size="11pt", bold=True, color='k')
        graphics_layout_widget.addItem(label_item, row=4, col=0, colspan=3)

        # Motor Data Visualization: Joint Velocities
        graphics_layout_widget = pg.GraphicsLayoutWidget(show=True)
        self.plot_widgets.append(graphics_layout_widget)
        main_layout.addWidget(graphics_layout_widget, 1, 0)

        self.joint_velocity_plot_items = []
        self.joint_velocity_buffers = []
        for i, leg in enumerate(["Front Right", "Front Left", "Hind Right", "Hind Left"]):
            for j, joint in enumerate(["Abduction", "Hip", "Knee"]):
                p = graphics_layout_widget.addPlot(row=i, col=j)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setTitle(f"{leg} {joint}", color="k", size="20pt")
                

                x_deque = collections.deque(
                    np.linspace(0, 0, self.window_size),
                    maxlen=self.window_size,
                )
                y_deque = collections.deque(
                    [np.nan] * self.window_size,
                    maxlen=self.window_size,
                )

                plot_item = p.plot(
                    list(x_deque), list(y_deque),
                    pen=pg.mkPen(color=(100, 149, 237), width=2)
                )

                self.joint_velocity_plot_items.append(plot_item)
                self.joint_velocity_buffers.append({'x': x_deque, 'y': y_deque})

                if i == 0 and j == 0:
                    p.setLabel("left", "Velocity (rad/s)", **{"color": "black", "font-size": "12pt"})
                elif i == 1 and j == 0:
                    p.setLabel("left", "Velocity (rad/s)", **{"color": "black", "font-size": "12pt"})
                elif i == 2 and j == 0:
                    p.setLabel("left", "Velocity (rad/s)", **{"color": "black", "font-size": "12pt"})
                elif i == 3 and j == 0:
                    p.setLabel("left", "Velocity (rad/s)", **{"color": "black", "font-size": "12pt"})

        shared_x_label_text = "Time (s)"
        label_item = pg.LabelItem(shared_x_label_text, size="11pt", bold=True, color='k')
        graphics_layout_widget.addItem(label_item, row=4, col=0, colspan=3)

        # Motor Data Visualization: Torque Estimates
        graphics_layout_widget = pg.GraphicsLayoutWidget(show=True)
        self.plot_widgets.append(graphics_layout_widget)
        main_layout.addWidget(graphics_layout_widget, 1, 1)

        self.torque_estimate_plot_items = []
        self.torque_estimate_buffers = []
        for i, leg in enumerate(["Front Right", "Front Left", "Hind Right", "Hind Left"]):
            for j, joint in enumerate(["Abduction", "Hip", "Knee"]):
                p = graphics_layout_widget.addPlot(row=i, col=j)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setTitle(f"{leg} {joint}", color="k", size="20pt")
                

                x_deque = collections.deque(
                    np.linspace(0, 0, self.window_size),
                    maxlen=self.window_size,
                )
                y_deque = collections.deque(
                    [np.nan] * self.window_size,
                    maxlen=self.window_size,
                )

                plot_item = p.plot(
                    list(x_deque), list(y_deque),
                    pen=pg.mkPen(color=(100, 149, 237), width=2)
                )
                self.torque_estimate_plot_items.append(plot_item)
                self.torque_estimate_buffers.append({'x': x_deque, 'y': y_deque})

                if i == 0 and j == 0:
                    p.setLabel("left", "Torque (Nm)", **{"color": "black", "font-size": "12pt"})
                elif i == 1 and j == 0:
                    p.setLabel("left", "Torque (Nm)", **{"color": "black", "font-size": "12pt"})
                elif i == 2 and j == 0:
                    p.setLabel("left", "Torque (Nm)", **{"color": "black", "font-size": "12pt"})
                elif i == 3 and j == 0:
                    p.setLabel("left", "Torque (Nm)", **{"color": "black", "font-size": "12pt"})

        shared_x_label_text = "Time (s)"
        label_item = pg.LabelItem(shared_x_label_text, size="11pt", bold=True, color='k')
        graphics_layout_widget.addItem(label_item, row=4, col=0, colspan=3)

        # Time Initialization:
        self.update_interval_ms = 100
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_interval_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self.current_time = 0.0

    def update_plot(self):
        imu_state = self.unitree_driver.get_imu_state()
        motor_state = self.unitree_driver.get_motor_state()

        # Update Plot Data:
        self.current_time += self.update_interval_ms * 1.0e-3

        # Unpack IMU state:
        accelerometer = np.asarray(imu_state.accelerometer)
        gyroscope = np.asarray(imu_state.gyroscope)
        rpy = np.asarray(imu_state.rpy)

        # Unpack Motor state:
        joint_positions = np.asarray(motor_state.q)
        joint_velocities = np.asarray(motor_state.qd)
        torque_estimates = np.asarray(motor_state.torque_estimate)

        # Update IMU Data:
        sensors = np.concatenate([accelerometer, gyroscope, rpy])
        for sensor, plot_item, buffer in zip(sensors, self.imu_main_plot_items, self.imu_buffers):
            x_deque = buffer['x']
            y_deque = buffer['y']
            x_deque.append(self.current_time)
            y_deque.append(sensor)
            # Update the plot item with the new data
            plot_item.setData(list(x_deque), list(y_deque))

        # Update Motor Data: Joint Positions
        for joint_position, plot_item, buffer in zip(joint_positions, self.joint_position_plot_items, self.joint_position_buffers):
            x_deque = buffer['x']
            y_deque = buffer['y']
            x_deque.append(self.current_time)
            y_deque.append(joint_position)
            # Update the plot item with the new data
            plot_item.setData(list(x_deque), list(y_deque))

        # Update Motor Data: Joint Velocities
        for joint_velocity, plot_item, buffer in zip(joint_velocities, self.joint_velocity_plot_items, self.joint_velocity_buffers):
            x_deque = buffer['x']
            y_deque = buffer['y']
            x_deque.append(self.current_time)
            y_deque.append(joint_velocity)
            # Update the plot item with the new data
            plot_item.setData(list(x_deque), list(y_deque))

        # Update Motor Data: Torque Estimates
        for torque_estimate, plot_item, buffer in zip(torque_estimates, self.torque_estimate_plot_items, self.torque_estimate_buffers):
            x_deque = buffer['x']
            y_deque = buffer['y']
            x_deque.append(self.current_time)
            y_deque.append(torque_estimate)
            # Update the plot item with the new data
            plot_item.setData(list(x_deque), list(y_deque))


def main(argv=None):
    qt_app = QtWidgets.QApplication([])

    # Set global Config:
    pg.setConfigOption('background', 'w')  # 'w' stands for white
    pg.setConfigOption('foreground', 'k')  # 'k' stands for black

    main_window = MainWindow(window_size=100)
    main_window.show()
    qt_app.exec()


if __name__ == "__main__":
    app.run(main)
