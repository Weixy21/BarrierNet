from re import X
from typing import Optional, List, Union
import numpy as np
import scipy.integrate as ode_solve

from ...utils import logging, transform


class State:
    """ Vehicle state without steering angle and velocity. """
    def __init__(self,
                 x: Optional[float] = 0.,
                 y: Optional[float] = 0.,
                 yaw: Optional[float] = 0.) -> None:
        self.update(x, y, yaw)

    def update(self, x: float, y: float, yaw: float) -> None:
        """ Set values. """
        self._x = x
        self._y = y
        self._yaw = yaw

    def reset(self) -> None:
        """ Reset to zeros. """
        self.update(0., 0., 0.)

    def numpy(self) -> np.ndarray:
        """ Convert to numpy. """
        return np.array([self._x, self._y, self._yaw])

    @property
    def x(self) -> float:
        """ Position of the car in x-axis. """
        return self._x

    @property
    def y(self) -> float:
        """ Position of the car in y-axis. """
        return self._y

    @property
    def yaw(self) -> float:
        """ Heading/yaw of the car. """
        return self._yaw

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' + \
               f'[{self.x}, {self.y}, {self.yaw}]>'


class BicycleDynamics:
    """ Simple continuous kinematic (bicycle) model of a rear-wheel driven vehicle.
    Check out Eq.3 in https://www.autonomousrobots.nl/docs/17-schwartig-autonomy-icra.pdf

    Args:
        x (float): Position of the car in x-axis.
        y (float): Position of the car in y-axis.
        yaw (float): Heading/yaw of the car.
        steering (float): Steering angle of tires instead of steering wheel.
        speed (float): Forward speed.
        steering_bound (list): Upper and lower bound of steering angle.
        speed_bound (list): Upper and lower bound of speed.
        wheel_base(float): Wheel base.

    """
    def __init__(self,
                 x: Optional[float] = 0.,
                 y: Optional[float] = 0.,
                 yaw: Optional[float] = 0.,
                 steering: Optional[float] = 0.,
                 speed: Optional[float] = 0.,
                 steering_bound: Optional[List[float]] = [-0.75, 0.75],
                 omega_bound: Optional[List[float]] = [-1.0, 1.0],
                 a_bound: Optional[List[float]] = [-7.0, 7.0],
                 speed_bound: Optional[List[float]] = [0., 15.],
                 wheel_base: Optional[float] = 2.8) -> None:
        self._x = 0.
        self._y = 0.
        self._yaw = 0.
        self._steering = 0.
        self._speed = 0.

        self._s = 0.

        self.update(x, y, yaw, steering, speed)
        self._steering_bound = steering_bound
        self._speed_bound = speed_bound
        self._omega_bound = omega_bound
        self._a_bound = a_bound
        self._wheel_base = wheel_base

    def step(self,
             steering_velocity: float,
             acceleration: float,
             dt: float,
             max_steps: Optional[int] = 100) -> np.ndarray:
        """ Step the vehicle state by solving ODE of the continuous kinematic (bicycle) model.

        Args:
            steering_velocity (float): Steering velocity (of tire angle).
            acceleration (float): Acceleration.
            dt (float): Elapsed time.
            max_steps (int): Maximal step to run the ODE solver; default to 100.

        Returns:
            np.ndarray: The updated state (x, y, yaw, tire angle, speed).

        """

        # Define dynamics
        def _ode_func(t, z):
            _x, _y, _phi, _delta, _v = z
            u_delta = steering_velocity
            u_a = acceleration
            new_z = np.array([
                -_v * np.sin(_phi), _v * np.cos(_phi),
                _v / self._wheel_base * np.tan(_delta), u_delta, u_a
            ])
            self._s += np.linalg.norm([-_v * np.sin(_phi), _v * np.cos(_phi)])
            return new_z

        # Solve ODE; NOTE: x and y axes are swapped
        z_0 = np.array(
            [self._x, self._y, self._yaw, self._steering, self._speed])
        solver = ode_solve.RK45(_ode_func, 0., z_0, dt)
        steps = 0
        while solver.status == 'running' and steps <= max_steps:
            solver.step()
            steps += 1
        if (dt - solver.t) < 0:
            logging.error('Reach max steps {} without reaching t_bound ({} < {})'.format( \
                max_steps, solver.t, solver.t_bound))

        self._x, self._y, self._yaw, self._steering, self._speed = solver.y

        # Clip by value bounds
        self._steering = np.clip(self._steering, *self._steering_bound)
        self._speed = np.clip(self._speed, *self._speed_bound)

        return self.numpy()

    def numpy(self) -> np.ndarray:
        """ Return a numpy array of vehicle state.

        Returns:
            np.ndarray: Vehicle state.

        """
        return np.array(
            [self._x, self._y, self._yaw, self._steering, self._speed])

    def copy(self):
        """ Create a copy. """
        return BicycleDynamics(x=self._x,
                               y=self._y,
                               yaw=self._yaw,
                               steering=self._steering,
                               speed=self._speed)

    def update(self, x: float, y: float, yaw: float, steering: float,
               speed: float) -> None:
        """ Set state.
        Args:
            x (float): Position of the car in x-axis.
            y (float): Position of the car in y-axis.
            yaw (float): Heading/yaw of the car.
            steering (float): Steering angle of tires instead of steering wheel.
            speed (float): Forward speed.

        """
        self._x = x
        self._y = y
        self._yaw = yaw
        self._steering = steering
        self._speed = speed

    def reset(self) -> None:
        """ Reset all state to zeros. """
        self.update(0., 0., 0., 0., 0.)
        self._s = 0.

    @property
    def s(self) -> float:
        """ Progress of the car. """
        return self._s

    @property
    def x(self) -> float:
        """ Position of the car in x-axis. """
        return self._x

    @property
    def y(self) -> float:
        """ Position of the car in y-axis. """
        return self._y

    @property
    def yaw(self) -> float:
        """ Heading/yaw of the car. """
        return self._yaw

    @property
    def steering(self) -> float:
        """ Steering (tire) angle. """
        return self._steering

    @property
    def steering_bound(self) -> List[float]:
        """ Lower and upper bound of steering (tire) angle. """
        return self._steering_bound

    @property
    def speed(self) -> float:
        """ Speed. """
        return self._speed

    @property
    def speed_bound(self) -> List[float]:
        """ Lower and upper bound of speed. """
        return self._speed_bound

    @property
    def acceleration_bound(self) -> List[float]:
        return self._a_bound

    @property
    def steering_rate_bound(self) -> List[float]:
        return self._omega_bound

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' + \
            f'[{self.x}, {self.y}, {self.yaw}, {self.steering}, {self.speed}]>'


# TODO: only to make previous pickled object still working
StateDynamicsBase = BicycleDynamics
StateDynamics = StateDynamicsBase


class CurvilinearDynamics:
    def __init__(self,
                 s: Optional[float] = 0.,
                 d: Optional[float] = 0.,
                 mu: Optional[float] = 0.,
                 delta: Optional[float] = 0.,
                 v: Optional[float] = 0.,
                 delta_bound: Optional[List[float]] = [-0.75, 0.75],
                 v_bound: Optional[List[float]] = [0., 15.],
                 omega_bound: Optional[List[float]] = [-1.0, 1.0],
                 a_bound: Optional[List[float]] = [-7.0, 7.0],
                 lr: Optional[float] = 2.,
                 lf: Optional[float] = 2.,
                 ref: Optional[np.ndarray] = None) -> None:
        self._s = 0.  # progress in curvilinear frame
        self._d = 0.  # lateral displacement in curvilinear frame
        self._mu = 0.  # local heading error
        self._delta = 0.  # steering angle
        self._v = 0.  # speed

        self.update(s, d, mu, delta, v, set_xyphi=False)

        self._lr = lr
        self._lf = lf

        self._delta_bound = delta_bound
        self._v_bound = v_bound
        self._omega_bound = omega_bound
        self._a_bound = a_bound

        if ref is not None:
            assert ref.shape[
                1] == 4, 'reference path should include x, y, heading, and road curvature'
            self.set_reference(ref)

    def set_reference(self, ref):
        # x, y, phi, kappa, s
        dx = ref[1:, 0] - ref[:-1, 0]
        dy = ref[1:, 1] - ref[:-1, 1]
        s = np.cumsum(np.sqrt(dx**2 + dy**2))
        s = np.insert(s, 0, 0.)
        self._ref = np.concatenate([ref, s[:, None]], axis=1)
        self._ref_idx = 0

    def step(self,
             steering_velocity: float,
             acceleration: float,
             dt: float,
             max_steps: Optional[int] = 100) -> np.ndarray:
        # Define dynamics
        def _ode_func(t, z):
            _s, _d, _mu, _v, _delta = z
            _v = np.clip(_v, *self._v_bound)
            _delta = np.clip(_delta, *self._delta_bound)
            _beta = np.arctan(
                (self._lr / (self._lr + self._lf) * np.tan(_delta)))
            _sin_mu_beta = np.sin(_mu + _beta)
            _cos_mu_beta = np.cos(_mu + _beta)
            self._ref_idx = np.argmin(np.abs(_s - self._ref[:, 4]))
            self._kappa = self._ref[self._ref_idx, 3]
            z_dot = np.array([(_v * _cos_mu_beta) / (1 - _d * self._kappa),
                              _v * _sin_mu_beta,
                              _v / self._lr * np.sin(_beta) - self._kappa *
                              (_v * _cos_mu_beta) / (1 - _d * self._kappa),
                              acceleration, steering_velocity])
            assert _s >= self._s  # NOTE: sanity check
            return z_dot

        z_0 = self.numpy(return_xyphi=False)
        solver = ode_solve.RK45(_ode_func, 0., z_0, dt)
        steps = 0
        while solver.status == 'running' and steps <= max_steps:
            solver.step()
            steps += 1
        if (dt - solver.t) < 0:
            print('Reach max steps {} without reaching t_bound ({} < {})'.format( \
                max_steps, solver.t, solver.t_bound))

        self._s, self._d, self._mu, self._v, self._delta = solver.y
        self._x, self._y, self._phi = self.get_xyphi()

        # Clip by value bounds
        self._delta = np.clip(self._delta, *self._delta_bound)
        self._v = np.clip(self._v, *self._v_bound)

        return self.numpy()

    def get_xyphi(self):
        ref_pose = self._ref[self._ref_idx]
        x = ref_pose[0] - self._d * np.cos(ref_pose[2])
        y = ref_pose[1] + self._d * np.sin(ref_pose[2])
        phi = ref_pose[2] + self._mu
        return np.array([x, y, phi])

    def numpy(self, return_xyphi=True):
        if return_xyphi:
            x, y, phi = self.get_xyphi()
            return np.array([x, y, phi, self._v, self._delta])
        else:
            return np.array([self._s, self._d, self._mu, self._v, self._delta])

    def copy(self):
        return CurvilinearDynamics(s=self._s,
                                   d=self._d,
                                   mu=self._mu,
                                   delta=self._delta,
                                   v=self._v,
                                   delta_bound=self._delta_bound,
                                   v_bound=self._v_bound,
                                   lr=self._lr,
                                   lf=self._lf,
                                   ref=self._ref.copy())

    def update(self,
               s_or_x: float,
               d_or_y: float,
               mu_or_phi: float,
               delta: float,
               v: float,
               set_xyphi=True) -> None:
        if set_xyphi:
            self._x = s_or_x
            self._y = d_or_y
            self._phi = mu_or_phi
            self._ref_idx = np.argmin(
                np.linalg.norm(np.array([[self._x, self._y]]) -
                               self._ref[:, :2],
                               axis=1))
            ref_pose = self._ref[self._ref_idx]
            # NOTE: there exists some error by assigning s to that of the closest reference point
            self._s = ref_pose[4]
            if True:
                dx, dy, dyaw = transform.compute_relative_latlongyaw(
                    np.array([self._x, self._y, self._phi]), ref_pose[:3])
                self._d = -dx
                self._mu = dyaw
            else:
                self._mu = self._phi - ref_pose[2]
                d1 = -(self._x - ref_pose[0]) / (np.cos(self._mu) + 1e-8)
                d2 = (self._y - ref_pose[1]) / (np.sin(self._mu) + 1e-8)
                self._d = (d1 + d2) / 2.
                assert np.abs(d1 - d2) <= 1e-2  # NOTE: sanity check
            self._kappa = ref_pose[3]
        else:
            self._s = s_or_x
            self._d = d_or_y
            self._mu = mu_or_phi
            self._delta = delta
            self._v = v
            # NOTE: no update for ref_idx and kappa

    def reset(self) -> None:
        self.update(0., 0., 0., 0., 0.)
        self.update(0., 0., 0., 0., 0., set_xyphi=False)

    @property
    def s(self) -> float:
        return self._s

    @property
    def d(self) -> float:
        return self._d

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def yaw(self) -> float:
        return self._phi

    @property
    def steering(self) -> float:
        return self._delta

    @property
    def steering_bound(self) -> List[float]:
        return self._delta_bound

    @property
    def speed(self) -> float:
        return self._v

    @property
    def speed_bound(self) -> List[float]:
        return self._v_bound

    @property
    def acceleration_bound(self) -> List[float]:
        return self._a_bound

    @property
    def steering_rate_bound(self) -> List[float]:
        return self._omega_bound

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' + \
            f'[{self.s}, {self.d}, {self.mu}, {self.steering}, {self.speed}]>'


def curvature2tireangle(curvature: float, wheel_base: float) -> float:
    """ Convert curvature to tire angle.

    Args:
        curvature (float): Curvature.
        wheel_base (float): Wheel base.

    Returns:
        float: Tire angle.

    """
    return np.arctan(wheel_base * curvature)


def tireangle2curvature(tire_angle: float, wheel_base: float) -> float:
    """ Convert tire angel to curvature.

    Args:
        tire_angle (float): Tire angle.
        wheel_base (float): Wheel base.

    Returns:
        float: Curvature.

    """
    return np.tan(tire_angle) / wheel_base


def curvature2steering(curvature: float, wheel_base: float,
                       steering_ratio: float) -> float:
    """ Convert curvature to steering angle.

    Args:
        curvature (float): Curvature.
        wheel_base (float): Wheel base.
        steering_ratio (float): Steering ratio.

    Returns:
        float: Steering (wheel) angle.

    """
    tire_angle = curvature2tireangle(curvature, wheel_base)
    steering = tire_angle * steering_ratio * 180. / np.pi

    return steering


def steering2curvature(steering: float, wheel_base: float,
                       steering_ratio: float) -> float:
    """ Convert steering angle to curvature.

    Args:
        steering (float): Steering (wheel) angle.
        wheel_base (float): Wheel base.
        steering_ratio (float): Steering ratio.

    Returns:
        float: Curvature.

    """
    tire_angle = steering * (np.pi / 180.) / steering_ratio
    curvature = tireangle2curvature(tire_angle, wheel_base)

    return curvature


def update_with_perfect_controller(
        desired_state: List[float], dt: float,
        dynamics: Union[BicycleDynamics, CurvilinearDynamics]) -> None:
    """ Update vehicle state assuming a perfect low-level controller. This
    basically simulate that the low-level controller can "instantaneously"
    achieve the desired state (e.g., steering (tire) angle and speed) with control
    command (steering velocity and accleration) in vehicle dynamics.

    Args:
        desired_state (List[float]): A list of desired vehicle state.
        dt (float): Elapsed time.
        dynamics (Union[BicycleDynamics, CurvilinearDynamics]): Vehicle state.

    """
    # simulate condition when the desired state can be instantaneously achieved
    new_dyn = dynamics.numpy()
    new_dyn[-2:] = desired_state
    dynamics.update(*new_dyn)
    dynamics.step(0., 0., dt)
