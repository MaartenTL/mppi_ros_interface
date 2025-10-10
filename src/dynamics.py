import torch
from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import numpy as np
import sys

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

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        x, y, yaw, vx, vy, w = states.unbind(dim=1)

        throttle, steer = actions.unbind(dim=1)

        # evaluate steering angle
        steering_angle = mf.steering_2_steering_angle(steer, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)

        # evaluate longitudinal forces
        Fx = + mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self) \
             + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)

        acc_x = Fx / mf.m_self  # acceleration in the longitudinal direction

        w = vx * torch.tan(steering_angle) / (mf.lr_self + mf.lf_self)  # angular velocity
        vy = mf.l_COM_self * w

        xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
        xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
        xdot3 = w
        xdot4 = acc_x

        # 6) Euler integration to get next state
        x = x + xdot1 * self._dt
        y = y + xdot2 * self._dt
        yaw = yaw + xdot3 * self._dt
        vx = vx + xdot4 * self._dt
        vy = vy #s Taken from the ROS simulator
        w = w

        new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)

        return new_states,  actions

class Dynamic_Bicycle:
    def __init__(self, dt=0.05, device="cpu") -> None:
        self._dt = dt
        self._device = device

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        # extract states
        x, y, yaw, vx, vy, w = states.unbind(dim=1)

        throttle, steer = actions.unbind(dim=1)

        # evaluate steering angle
        steering_angle = mf.steering_2_steering_angle(steer, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
                                                      mf.e_s_self)

        # # evaluate longitudinal forces
        Fx_wheels = + mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self) \
                    + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self) \
                    + mf.F_friction_due_to_steering(steering_angle, vx, mf.a_stfr_self, mf.b_stfr_self, mf.d_stfr_self,
                                                    mf.e_stfr_self)

        c_front = (mf.m_front_wheel_self) / mf.m_self
        c_rear = (mf.m_rear_wheel_self) / mf.m_self

        # redistribute Fx to front and rear wheels according to normal load
        Fx_front = Fx_wheels * c_front
        Fx_rear = Fx_wheels * c_rear

        # evaluate slip angles
        alpha_f, alpha_r = mf.evaluate_slip_angles(vx, vy, w, mf.lf_self, mf.lr_self, steering_angle)

        # lateral forces
        Fy_wheel_f = mf.lateral_tire_force(alpha_f, mf.d_t_f_self, mf.c_t_f_self, mf.b_t_f_self, mf.m_front_wheel_self)
        Fy_wheel_r = mf.lateral_tire_force(alpha_r, mf.d_t_r_self, mf.c_t_r_self, mf.b_t_r_self, mf.m_rear_wheel_self)

        acc_x, acc_y, acc_w = mf.solve_rigid_body_dynamics(vx, vy, w, steering_angle, Fx_front, Fx_rear, Fy_wheel_f,
                                                           Fy_wheel_r, mf.lf_self, mf.lr_self, mf.m_self, mf.Jz_self)


        xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
        xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
        xdot3 = w
        xdot4 = acc_x
        xdot5 = acc_y
        xdot6 = acc_w

        # 6) Euler integration to get next state
        x = x + xdot1 * self._dt
        y = y + xdot2 * self._dt
        yaw = yaw + xdot3 * self._dt
        vx = vx + xdot4 * self._dt
        vy = vy + xdot5 * self._dt
        w = w + xdot6 * self._dt

        new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)

        return new_states,  actions

        return xdot