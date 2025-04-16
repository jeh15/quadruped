from absl import app, flags, logging
import os

import numpy as np

import matplotlib.pyplot as plt


def main(argv=None):
    # Load Simulation Data:
    log_path = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    simulation_gryoscope = np.loadtxt(f'{log_path}/simulation_gyroscope_data.txt', delimiter=',')
    simulation_projected_gravity = np.loadtxt(f'{log_path}/simulation_projected_gravity_data.txt', delimiter=',')
    simulation_quaternion = np.loadtxt(f'{log_path}/simulation_quaternion_data.txt', delimiter=',')
    simulation_joint_position = np.loadtxt(f'{log_path}/simulation_joint_position_data.txt', delimiter=',')
    simulation_joint_velocity = np.loadtxt(f'{log_path}/simulation_joint_velocity_data.txt', delimiter=',')
    simulation_action = np.loadtxt(f'{log_path}/simulation_action_data.txt', delimiter=',')
    simulation_ctrl = np.loadtxt(f'{log_path}/simulation_ctrl_data.txt', delimiter=',')

    # Load Hardware Data:
    hardware_gryoscope = np.loadtxt(f'{log_path}/hardware_gyroscope_data.txt', delimiter=',')
    hardware_projected_gravity = np.loadtxt(f'{log_path}/hardware_projected_gravity_data.txt', delimiter=',')
    hardware_quaternion = np.loadtxt(f'{log_path}/hardware_quaternion_data.txt', delimiter=',')
    hardware_joint_position = np.loadtxt(f'{log_path}/hardware_joint_position_data.txt', delimiter=',')
    hardware_joint_velocity = np.loadtxt(f'{log_path}/hardware_joint_velocity_data.txt', delimiter=',')
    hardware_action = np.loadtxt(f'{log_path}/hardware_action_data.txt', delimiter=',')
    hardware_ctrl = np.loadtxt(f'{log_path}/hardware_ctrl_data.txt', delimiter=',')
    hardware_filtered_gyroscope = np.loadtxt(f'{log_path}/hardware_filtered_gyroscope_data.txt', delimiter=',')
    hardware_filtered_projected_gravity = np.loadtxt(f'{log_path}/hardware_filtered_projected_gravity_data.txt', delimiter=',')
    hardware_filtered_joint_position = np.loadtxt(f'{log_path}/hardware_filtered_joint_position_data.txt', delimiter=',')
    hardware_filtered_joint_velocity = np.loadtxt(f'{log_path}/hardware_filtered_joint_velocity_data.txt', delimiter=',')

    plt.ion()

    # Plot: Gyroscope Comparison
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    for i, ax in enumerate(axs):
        ax.plot(simulation_gryoscope[:, i], label='Simulation')
        ax.plot(hardware_gryoscope[:, i], label='Hardware')
        ax.plot(hardware_filtered_gyroscope[:, i], label='Filtered')
        ax.legend()
    axs[0].set_ylabel('Gyroscope X')
    axs[1].set_ylabel('Gyroscope Y')
    axs[2].set_ylabel('Gyroscope Z')
    axs[2].set_xlabel('Time Step')
    axs[2].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'gyroscope_comparison.png'))
    plt.close(fig)

    # Plot: Projected Gravity Comparison
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    for i, ax in enumerate(axs):
        ax.plot(simulation_projected_gravity[:, i], label='Simulation')
        ax.plot(hardware_projected_gravity[:, i], label='Hardware')
        ax.plot(hardware_filtered_projected_gravity[:, i], label='Filtered')
        ax.legend()
    axs[0].set_ylabel('Projected Gravity X')
    axs[1].set_ylabel('Projected Gravity Y')
    axs[2].set_ylabel('Projected Gravity Z')
    axs[2].set_xlabel('Time Step')
    axs[2].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'projected_gravity_comparison.png'))
    plt.close(fig)

    # Plot: Quaternion Comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    for i, ax in enumerate(axs):
        ax.plot(simulation_quaternion[:, i], label='Simulation')
        ax.plot(hardware_quaternion[:, i], label='Hardware')
        ax.legend()
    axs[0].set_ylabel('Quaternion W')
    axs[1].set_ylabel('Quaternion X')
    axs[2].set_ylabel('Quaternion Y')
    axs[3].set_ylabel('Quaternion Z')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'quaternion_comparison.png'))
    plt.close(fig)

    # Plot: Joint Position Comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    fr_leg_ids = [0, 1, 2]
    fl_leg_ids = [3, 4, 5]
    rr_leg_ids = [6, 7, 8]
    rl_leg_ids = [9, 10, 11]
    leg_ids = [fr_leg_ids, fl_leg_ids, rr_leg_ids, rl_leg_ids]
    for ax, leg_id in zip(axs, leg_ids):
        for i in leg_id:
            ax.plot(simulation_joint_position[:, i], label='Simulation')
            ax.plot(hardware_joint_position[:, i], label='Hardware')
            ax.plot(hardware_filtered_joint_position[:, i], label='Filtered')
            ax.legend()

    axs[0].set_ylabel('Front Right Leg Joint Position')
    axs[1].set_ylabel('Front Left Leg Joint Position')
    axs[2].set_ylabel('Rear Left Leg Joint Position')
    axs[3].set_ylabel('Rear Right Leg Joint Position')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'joint_position_comparison.png'))
    plt.close(fig)

    # Plot: Joint Velocity Comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    for ax, leg_id in zip(axs, leg_ids):
        for i in leg_id:
            ax.plot(simulation_joint_velocity[:, i], label='Simulation')
            ax.plot(hardware_joint_velocity[:, i], label='Hardware')
            ax.plot(hardware_filtered_joint_velocity[:, i], label='Filtered')
            ax.legend()
    axs[0].set_ylabel('Front Right Leg Joint Velocity')
    axs[1].set_ylabel('Front Left Leg Joint Velocity')
    axs[2].set_ylabel('Rear Left Leg Joint Velocity')
    axs[3].set_ylabel('Rear Right Leg Joint Velocity')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'joint_velocity_comparison.png'))
    plt.close(fig)

    # Plot: Action Comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    for ax, leg_id in zip(axs, leg_ids):
        for i in leg_id:
            ax.plot(simulation_action[:, i], label='Simulation')
            ax.plot(hardware_action[:, i], label='Hardware')
            ax.legend()
    axs[0].set_ylabel('Front Right Leg Action')
    axs[1].set_ylabel('Front Left Leg Action')
    axs[2].set_ylabel('Rear Left Leg Action')
    axs[3].set_ylabel('Rear Right Leg Action')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'action_comparison.png'))
    plt.close(fig)

    # Plot: Control Comparison
    fig, axs = plt.subplots(4, 1, figsize=(10, 14))
    for ax, leg_id in zip(axs, leg_ids):
        for i in leg_id:
            ax.plot(simulation_ctrl[:, i], label='Simulation')
            ax.plot(hardware_ctrl[:, i], label='Hardware')
            ax.legend()
    axs[0].set_ylabel('Front Right Leg Control')
    axs[1].set_ylabel('Front Left Leg Control')
    axs[2].set_ylabel('Rear Left Leg Control')
    axs[3].set_ylabel('Rear Right Leg Control')
    axs[3].set_xlabel('Time Step')
    axs[3].legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(log_path, 'control_comparison.png'))
    plt.close(fig)

    # Plot: Joint Position by Leg:
    leg_map = {
        'front_right': [0, 1, 2],
        'front_left': [3, 4, 5],
        'hind_right': [6, 7, 8],
        'hind_left': [9, 10, 11]
    }
    for key, value in leg_map.items():
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))
        for ax, j in zip(axs, value):
            ax.plot(simulation_joint_position[:, j], label='Simulation')
            ax.plot(hardware_joint_position[:, j], label='Hardware')
            ax.plot(hardware_filtered_joint_position[:, j], label='Filtered')
            ax.legend()
        axs[0].set_ylabel('Abduction')
        axs[1].set_ylabel('Hip')
        axs[2].set_ylabel('Knee')
        axs[2].set_xlabel('Time Step')
        axs[2].legend()
        plt.suptitle(f'{key} leg joint position')
        plt.tight_layout()
        plt.draw()
        fig.savefig(os.path.join(log_path, f'joint_position_{key}_leg.png'))
        plt.close(fig)

    # Plot: Joint Velocity by Leg:
    for key, value in leg_map.items():
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))
        for ax, j in zip(axs, value):
            ax.plot(simulation_joint_velocity[:, j], label='Simulation')
            ax.plot(hardware_joint_velocity[:, j], label='Hardware')
            ax.plot(hardware_filtered_joint_velocity[:, j], label='Filtered')
            ax.legend()
        axs[0].set_ylabel('Abduction')
        axs[1].set_ylabel('Hip')
        axs[2].set_ylabel('Knee')
        axs[2].set_xlabel('Time Step')
        axs[2].legend()
        plt.suptitle(f'{key} leg joint velocity')
        plt.tight_layout()
        plt.draw()
        fig.savefig(os.path.join(log_path, f'joint_velocity_{key}_leg.png'))
        plt.close(fig)

    # Plot: Action by Leg:
    for key, value in leg_map.items():
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))
        for ax, j in zip(axs, value):
            ax.plot(simulation_action[:, j], label='Simulation')
            ax.plot(hardware_action[:, j], label='Hardware')
            ax.legend()
        axs[0].set_ylabel('Abduction')
        axs[1].set_ylabel('Hip')
        axs[2].set_ylabel('Knee')
        axs[2].set_xlabel('Time Step')
        axs[2].legend()
        plt.suptitle(f'{key} leg action')
        plt.tight_layout()
        plt.draw()
        fig.savefig(os.path.join(log_path, f'action_{key}_leg.png'))
        plt.close(fig)

    # Plot: Control by Leg:
    for key, value in leg_map.items():
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))
        for ax, j in zip(axs, value):
            ax.plot(simulation_ctrl[:, j], label='Simulation')
            ax.plot(hardware_ctrl[:, j], label='Hardware')
            ax.legend()
        axs[0].set_ylabel('Abduction')
        axs[1].set_ylabel('Hip')
        axs[2].set_ylabel('Knee')
        axs[2].set_xlabel('Time Step')
        axs[2].legend()
        plt.suptitle(f'{key} leg control')
        plt.tight_layout()
        plt.draw()
        fig.savefig(os.path.join(log_path, f'control_{key}_leg.png'))
        plt.close(fig)


if __name__ == "__main__":
    app.run(main)
