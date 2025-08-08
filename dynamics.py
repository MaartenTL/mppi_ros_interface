import torch
from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import numpy as np

mf = model_functions()

class OmnidirectionalPointRobotDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device


    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        x, y, theta = states[:, 0], states[:, 1], states[:, 2]

        new_x = x + actions[:, 0] * self._dt
        new_y = y + actions[:, 1] * self._dt
        new_theta = theta + actions[:, 2] * self._dt

        new_states = torch.stack([new_x, new_y, new_theta], dim=1)
        return new_states, actions

class Kinematic_Bicycle:
    def __init__(self, dt=0.05, device="cpu") -> None:
        self._dt = dt
        self._device = device
        self.counter = 0

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        self.counter = self.counter + 1
        x, y, yaw, vx, vy = states.unbind(dim=1)

        throttle, steer = actions.unbind(dim=1)

        # evaluate steering angle
        steering_angle = mf.steering_2_steering_angle(steer, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)

        # evaluate longitudinal forces
        Fx = + mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self) \
             + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)

        acc_x = Fx / mf.m_self  # acceleration in the longitudinal direction

        # simple bycicle nominal model - using centre of mass as reference point
        w = vx * np.tan(steering_angle) / (mf.lr_self + mf.lf_self)  # angular velocity
        vy = mf.l_COM_self * w

        xdot1 = vx * np.cos(yaw) - vy * np.sin(yaw)
        xdot2 = vx * np.sin(yaw) + vy * np.cos(yaw)
        xdot3 = w
        xdot4 = acc_x

        # 6) Euler integration to get next state
        x = x + xdot1 * self._dt
        y = y + xdot2 * self._dt
        yaw = yaw + xdot3 * self._dt
        vx = vx + xdot4 * self._dt
        vy = vy #s Taken from the ROS simulator

        new_states = torch.stack([x, y, yaw, vx, vy], dim=1)

        return new_states,  actions