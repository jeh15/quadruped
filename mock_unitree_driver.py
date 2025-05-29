from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class MockLowState:
    foot_force: np.ndarray


@dataclass
class MockImuState:
    quaternion: np.ndarray
    gyroscope: np.ndarray
    accelerometer: np.ndarray
    rpy: np.ndarray


@dataclass
class MockMotorState:
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    torque_estimate: np.ndarray


class MockUnitreeDriver:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data

    def update_data(self, mj_data: mujoco.MjData) -> None:
        self.mj_data = mj_data

    def get_low_state(self) -> MockLowState:
        """
            Returns a mock of the Unitree Go2 Low-level Struct.
            This differs from the original in that it returns
            a distance between contacts where negative means penetration.
        """
        contact = self.mj_data.contact
        foot_force = np.concatenate([
            contact[:self.mj_model.ncon].dist,
        ])
        return MockLowState(foot_force=foot_force)

    def get_imu_state(self) -> MockImuState:
        """
            Returns a mock of the Unitree Go2 IMU Struct.
        """
        # Convert Quaternion to RPY:
        def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
            sinr_cosp = 2 * (
                quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]
            )
            cosr_cosp = 1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = np.sqrt(
                1 + 2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1])
            )
            cosp = np.sqrt(1 - sinp ** 2)
            pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

            siny_cosp = 2 * (
                quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]
            )
            cosy_cosp = 1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return np.array([roll, pitch, yaw])

        quaternion = self.mj_data.qpos[3:7]
        gyroscope = self.get_sensor_data(
            self.mj_model, self.mj_data, "imu_gyro"
        )
        accelerometer = self.get_sensor_data(
            self.mj_model, self.mj_data, "imu_acceleration"
        )
        rpy = quaternion_to_rpy(quaternion)

        return MockImuState(
            quaternion=quaternion,
            gyroscope=gyroscope,
            accelerometer=accelerometer,
            rpy=rpy,
        )

    def get_motor_state(self) -> MockMotorState:
        """
            Returns a mock of the Unitree Go2 Motor Struct.
        """
        q = self.mj_data.qpos[7:]
        qd = self.mj_data.qvel[6:]
        qdd = np.zeros_like(qd)
        torque_estimate = self.mj_data.actuator_force

        return MockMotorState(
            q=q,
            qd=qd,
            qdd=qdd,
            torque_estimate=torque_estimate
        )

    @staticmethod
    def get_sensor_data(
        model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
    ) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr: sensor_adr + sensor_dim]
