import torch
from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import numpy as np
import sys
import importlib.resources

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

# class Kinematic_Bicycle:
#     def __init__(self, dt=0.05, device="cpu") -> None:
#         self._dt = dt
#         self._device = device
#
#     def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
#         x, y, yaw, vx, vy, w = states.unbind(dim=1)
#
#         throttle, steer = actions.unbind(dim=1)
#         # evaluate steering angle
#         steering_angle = mf.steering_2_steering_angle(steer, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)
#
#         # evaluate longitudinal forces
#         Fx = + mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self) \
#              + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)
#
#         acc_x = Fx / mf.m_self  # acceleration in the longitudinal direction
#
#         w = vx * torch.tan(steering_angle) / (mf.lr_self + mf.lf_self)  # angular velocity
#         vy = mf.l_COM_self * w
#
#         xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
#         xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
#         xdot3 = w
#         xdot4 = acc_x
#
#         # 6) Euler integration to get next state
#         x = x + xdot1 * self._dt
#         y = y + xdot2 * self._dt
#         yaw = yaw + xdot3 * self._dt
#         vx = vx + xdot4 * self._dt
#         vy = vy #s Taken from the ROS simulator
#         w = w
#
#         new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)
#
#         return new_states,  actions

class Kinematic_Bicycle:
    def __init__(self, dt=0.05, device="cpu") -> None:
        self._dt = dt
        self._device = device

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        # decide substepping based on outer dt
        dt_total = self._dt
        if dt_total > 0.08:
            n_sub = 2
        else:
            n_sub = 1
        dt_sub = dt_total / n_sub

        # unpack once; we will update these inside the loop
        x, y, yaw, vx, vy, w = states.unbind(dim=1)
        throttle, steer = actions.unbind(dim=1)

        for _ in range(n_sub):
            # evaluate steering angle
            #
            steering_angle = mf.steering_2_steering_angle(
                steer,
                mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
            )

            # steering_angle = mf.steering_2_steering_angle(
            #         steer,
            #         mf.a_s_self, mf.b_s_self, 0.0, mf.d_s_self, mf.e_s_self
            #     )

            # steering_angle = mf.steering_2_steering_angle(
            #     -torch.abs(steer),
            #     mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
            # ) * -torch.sign(steer)
            # steering_angle = mf.steering_2_steering_angle(
            #     -torch.abs(steer),
            #     mf.a_s_self, mf.b_s_self, 0.0, mf.d_s_self, mf.e_s_self
            # ) * -torch.sign(steer)

            # evaluate longitudinal forces
            Fx = (
                mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self)
                + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)
            )

            # th_abs = torch.abs(throttle)
            # th_sign = torch.sign(throttle)  # -1, 0, +1
            #
            # Fx_motor_mag = torch.abs(mf.motor_force(th_abs, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self))
            # Fx_motor = th_sign * Fx_motor_mag
            #
            # Fx = Fx_motor + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)
            # acc_x = Fx / mf.m_self


            acc_x = Fx / mf.m_self  # longitudinal acceleration
            w = vx * torch.tan(steering_angle) / (mf.lr_self + mf.lf_self)  # angular velocity
            vy = mf.l_COM_self * w

            xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
            xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
            xdot3 = w
            xdot4 = acc_x

            # Euler integration with substep dt_sub
            x   = x   + xdot1 * dt_sub
            y   = y   + xdot2 * dt_sub
            yaw = yaw + xdot3 * dt_sub
            vx  = vx  + xdot4 * dt_sub
            # vy and w are directly set above (no integration needed here)

        new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)
        return new_states, actions


# class Dynamic_Bicycle:
#     def __init__(self, dt=0.05, device="cpu") -> None:
#         self._dt = dt
#         self._device = device
#
#     def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
#         # extract states
#         x, y, yaw, vx, vy, w = states.unbind(dim=1)
#
#         throttle, steer = actions.unbind(dim=1)
#
#         # evaluate steering angle
#         steering_angle = mf.steering_2_steering_angle(steer, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
#                                                       mf.e_s_self)
#
#         # # evaluate longitudinal forces
#         Fx_wheels = + mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self) \
#                     + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self) \
#                     + mf.F_friction_due_to_steering(steering_angle, vx, mf.a_stfr_self, mf.b_stfr_self, mf.d_stfr_self,
#                                                     mf.e_stfr_self)
#
#         c_front = (mf.m_front_wheel_self) / mf.m_self
#         c_rear = (mf.m_rear_wheel_self) / mf.m_self
#
#         # redistribute Fx to front and rear wheels according to normal load
#         Fx_front = Fx_wheels * c_front
#         Fx_rear = Fx_wheels * c_rear
#
#         # evaluate slip angles
#         alpha_f, alpha_r = mf.evaluate_slip_angles(vx, vy, w, mf.lf_self, mf.lr_self, steering_angle)
#
#         # lateral forces
#         Fy_wheel_f = mf.lateral_tire_force(alpha_f, mf.d_t_f_self, mf.c_t_f_self, mf.b_t_f_self, mf.m_front_wheel_self)
#         Fy_wheel_r = mf.lateral_tire_force(alpha_r, mf.d_t_r_self, mf.c_t_r_self, mf.b_t_r_self, mf.m_rear_wheel_self)
#
#         acc_x, acc_y, acc_w = mf.solve_rigid_body_dynamics(vx, vy, w, steering_angle, Fx_front, Fx_rear, Fy_wheel_f,
#                                                            Fy_wheel_r, mf.lf_self, mf.lr_self, mf.m_self, mf.Jz_self)
#
#
#         xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
#         xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
#         xdot3 = w
#         xdot4 = acc_x
#         xdot5 = acc_y
#         xdot6 = acc_w
#
#         # 6) Euler integration to get next state
#         x = x + xdot1 * self._dt
#         y = y + xdot2 * self._dt
#         yaw = yaw + xdot3 * self._dt
#         vx = vx + xdot4 * self._dt
#         vy = vy + xdot5 * self._dt
#         w = w + xdot6 * self._dt
#
#         new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)
#
#         return new_states, actions
def _check_finite(name, tensor, vx):
    bad = ~torch.isfinite(tensor)
    if bad.any():
        print(f"[NaN/Inf in {name}]")
        print(" indices:", torch.where(bad))
        print(" values:", tensor[bad])
        print(" vx values at bad indices:", vx[bad])
        raise RuntimeError(f"Non-finite values detected in {name}")


class Dynamic_Bicycle:
    def __init__(self, dt=0.05, device="cpu") -> None:
        self._dt = dt
        self._device = device

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        # decide substepping based on outer dt
        dt_total = self._dt
        if dt_total > 0.08:
            n_sub = 2
        else:
            n_sub = 1
        dt_sub = dt_total / n_sub

        # unpack once; we will update these inside the loop
        x, y, yaw, vx, vy, w = states.unbind(dim=1)
        throttle, steer = actions.unbind(dim=1)

        for _ in range(n_sub):
            # steering angle
            steering_angle = mf.steering_2_steering_angle(
                steer,
                mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
            )
            # _check_finite(f"steering_angle", steering_angle)
            # longitudinal forces
            Fx_wheels = (
                mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self)
                + mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self)
                + mf.F_friction_due_to_steering(
                    steering_angle, vx,
                    mf.a_stfr_self, mf.b_stfr_self, mf.d_stfr_self, mf.e_stfr_self
                )
            )


            _check_finite(f"vx", vx, vx)

            _check_finite(f"motor force", mf.motor_force(throttle, vx, mf.a_m_self, mf.b_m_self, mf.c_m_self), vx)
            _check_finite(f"rolling friction", mf.rolling_friction(vx, mf.a_f_self, mf.b_f_self, mf.c_f_self, mf.d_f_self), vx)
            _check_finite(f"steering friction", mf.F_friction_due_to_steering(
                    steering_angle, vx,
                    mf.a_stfr_self, mf.b_stfr_self, mf.d_stfr_self, mf.e_stfr_self
                ), vx)

            _check_finite(f"Fx_wheels", Fx_wheels, vx)


            c_front = mf.m_front_wheel_self / mf.m_self
            c_rear  = mf.m_rear_wheel_self  / mf.m_self

            Fx_front = Fx_wheels * c_front
            Fx_rear  = Fx_wheels * c_rear

            # _check_finite(f"Fx_front", Fx_front)
            # _check_finite(f"Fx_rear", Fx_rear)

            # slip angles
            alpha_f, alpha_r = mf.evaluate_slip_angles(
                vx, vy, w, mf.lf_self, mf.lr_self, steering_angle
            )
            # _check_finite(f"alpha_f", alpha_f)
            # _check_finite(f"alpha_r", alpha_r)

            # lateral forces
            Fy_wheel_f = mf.lateral_tire_force(
                alpha_f,
                mf.d_t_f_self, mf.c_t_f_self, mf.b_t_f_self, mf.m_front_wheel_self
            )
            Fy_wheel_r = mf.lateral_tire_force(
                alpha_r,
                mf.d_t_r_self, mf.c_t_r_self, mf.b_t_r_self, mf.m_rear_wheel_self
            )

            # _check_finite(f"Fy_wheel_f", Fy_wheel_f)
            # _check_finite(f"Fy_wheel_r", Fy_wheel_r)

            # accelerations
            acc_x, acc_y, acc_w = mf.solve_rigid_body_dynamics(
                vx, vy, w, steering_angle,
                Fx_front, Fx_rear,
                Fy_wheel_f, Fy_wheel_r,
                mf.lf_self, mf.lr_self,
                mf.m_self, mf.Jz_self
            )

            AX_MAX = 30.0  # m/s^2
            AY_MAX = 30.0
            AW_MAX = 50.0  # rad/s^2, tune

            acc_x = torch.clamp(acc_x, -AX_MAX, AX_MAX)
            acc_y = torch.clamp(acc_y, -AY_MAX, AY_MAX)
            acc_w = torch.clamp(acc_w, -AW_MAX, AW_MAX)

            _check_finite(f"acc_x", acc_x, vx)
            # _check_finite(f"acc_y", acc_y)
            # _check_finite(f"acc_w", acc_w)

            xdot1 = vx * torch.cos(yaw) - vy * torch.sin(yaw)
            xdot2 = vx * torch.sin(yaw) + vy * torch.cos(yaw)
            xdot3 = w
            xdot4 = acc_x
            xdot5 = acc_y
            xdot6 = acc_w

            # _check_finite(f"xdot1", xdot1)
            # _check_finite(f"xdot2", xdot2)
            # _check_finite(f"xdot3", xdot3)
            # _check_finite(f"xdot4", xdot4)
            # _check_finite(f"xdot5", xdot5)
            # _check_finite(f"xdot6", xdot6)

            # Euler integration with substep dt_sub
            x   = x   + xdot1 * dt_sub
            y   = y   + xdot2 * dt_sub
            yaw = yaw + xdot3 * dt_sub
            vx  = vx  + xdot4 * dt_sub
            vy  = vy  + xdot5 * dt_sub
            w   = w   + xdot6 * dt_sub

        new_states = torch.stack([x, y, yaw, vx, vy, w], dim=1)
        return new_states, actions


class RateAugmentedDynamics:
    """
    Wrap a base dynamics model so that MPPI samples [dthrottle, dsteer]
    but the underlying model still uses [throttle, steer].

    State for MPPI: [x, y, yaw, vx, vy, w, throttle, steer]  (nx = 8)
    Action for MPPI: [dthrottle, dsteer]
    """
    def __init__(self, base_dyn, dt=0.05,
                 th_min=0.0, th_max=1.0,
                 steer_min=-1.0, steer_max=1.0,
                 device="cpu") -> None:
        self.base_dyn = base_dyn          # Kinematic_Bicycle or Dynamic_Bicycle
        self._dt = dt
        self._device = device
        self.th_min = th_min
        self.th_max = th_max
        self.steer_min = steer_min
        self.steer_max = steer_max

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int):
        """
        states: [K, 8] = [x, y, yaw, vx, vy, w, throttle, steer]
        actions: [K, 2] = [dthrottle, dsteer]  (what MPPI samples)
        returns:
            new_states: [K, 8]
            actions_out: [K, 2]  (we return the *rates* again for costs)
        """
        # split state
        x6   = states[:, :6]             # vehicle states
        th   = states[:, 6]              # current throttle
        steer = states[:, 7]             # current steer

        # split actions as *rates*
        dth, dsteer = actions.unbind(dim=1)

        # integrate to get actual commands, with saturation
        # th_new = torch.clamp(th + dth * self._dt, self.th_min, self.th_max)
        th_new = dth


        # steer_new = torch.clamp(steer + dsteer * self._dt, self.steer_min, self.steer_max)
        steer_new = dsteer

        # underlying dynamics still expect [throttle, steer]
        u_abs = torch.stack([th_new, steer_new], dim=1)  # [K, 2]

        # step the base model with the absolute commands
        x6_new, _ = self.base_dyn.step(x6, u_abs, t)

        new_states = torch.cat([x6_new, th_new.unsqueeze(1), steer_new.unsqueeze(1)], dim=1)

        actions_out = actions

        return new_states, actions_out







class SVGP:  # RK4 wants a function that takes as input time and state
    def __init__(self, dt=0.05, device="cpu") -> None:
        self._dt = dt
        self._device = device
        self.dynamic = Dynamic_Bicycle(dt=self._dt, device=self._device)
        with importlib.resources.path('DART_dynamic_models', 'SVGP_saved_parameters') as data_path:
            folder_path = str(data_path)

        evaluate_covariance_tag = False

        self.model_vx, self.model_vy, self.model_w = load_SVGPModel_actuator_dynamics_analytic(folder_path, evaluate_covariance_tag)

        self.torch_vx = TorchSVGP(self.model_vx.outputscale, self.model_vx.lengthscale,
                                  self.model_vx.inducing_locations, self.model_vx.right_vec,
                                  device=self._device, dtype=torch.float32)
        self.torch_vy = TorchSVGP(self.model_vy.outputscale, self.model_vy.lengthscale,
                                  self.model_vy.inducing_locations, self.model_vy.right_vec,
                                  device=self._device, dtype=torch.float32)
        self.torch_w = TorchSVGP(self.model_w.outputscale, self.model_w.lengthscale,
                                 self.model_w.inducing_locations, self.model_w.right_vec,
                                 device=self._device, dtype=torch.float32)

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:

        # extract states
        x, y, yaw, vx, vy, w = states.unbind(dim=1)

        throttle, steer = actions.unbind(dim=1)

        # ['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
        state_action_base_model = np.array([vx, vy, w, throttle, steer])

        # states, actions are [K, ...]
        # Pack features as [K,5] in the order the model was trained on:
        X_star = torch.stack([vx, vy, w, throttle, steer], dim=1).to(self._device).to(torch.float32)  # [K,5]

        acc_x = self.torch_vx.forward_mean(X_star)  # [K]
        acc_y = self.torch_vy.forward_mean(X_star)  # [K]
        acc_w = self.torch_w.forward_mean(X_star)  # [K]

        # to avoid strange jittering close to 0 velocity, blend
        act_coeff = self.activation_coeff(vx)

        # add nominal model
        xdot_nom, _ = self.dynamic.step(states, actions, t)

        acc_x = act_coeff * acc_x + xdot_nom[:,3]
        acc_y = act_coeff * acc_y + xdot_nom[:,4]
        acc_w = act_coeff * acc_w + xdot_nom[:,5]
        # assemble derivatives [throttle, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
        # xdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6])

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

        return new_states, actions

    def activation_coeff(self, vx):
        c = 0.5
        sharpness = 40
        out = torch.tanh(sharpness * (vx - c)) * 0.5 + 0.5
        # print with 2 decimal points
        # print('activation coefficient: ' + "{:.2f}".format(out))
        return out


class TorchSVGP:
    def __init__(self, outputscale, lengthscale, inducing_locations, right_vec, device="cpu", dtype=torch.float32):
        # Convert saved numpy params once to torch
        self.device = device
        self.dtype = dtype
        self.outputscale = torch.as_tensor(outputscale, device=device, dtype=dtype).reshape(())
        self.lengthscale = torch.as_tensor(lengthscale, device=device, dtype=dtype)   # shape [D]
        Z = torch.as_tensor(inducing_locations, device=device, dtype=dtype)           # [M, D]
        self.Z = Z
        # right_vec should be [M] or [M,1]
        rv = torch.as_tensor(right_vec, device=device, dtype=dtype)
        self.right_vec = rv.view(-1)  # [M]

    @torch.no_grad()
    def forward_mean(self, X):
        """
        X: [K, D] torch tensor on same device
        returns: mean [K] torch tensor
        """
        # RBF kernel K(X,Z) with ARD lengthscales, outputscale included
        # pairwise squared Mahalanobis distances
        Xs = X / self.lengthscale          # [K, D]
        Zs = self.Z / self.lengthscale     # [M, D]
        # ||Xs||^2 + ||Zs||^2 - 2 Xs·Zs^T
        x2 = (Xs**2).sum(dim=1, keepdim=True)          # [K,1]
        z2 = (Zs**2).sum(dim=1).unsqueeze(0)           # [1,M]
        cross = Xs @ Zs.t()                             # [K,M]
        d2 = x2 + z2 - 2.0 * cross                      # [K,M]
        Kxz = self.outputscale * torch.exp(-0.5 * d2)   # [K,M]
        mean = Kxz @ self.right_vec                     # [K]
        return mean
