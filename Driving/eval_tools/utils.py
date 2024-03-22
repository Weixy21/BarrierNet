import os
import numpy as np
import cv2
import torch

from data_modules.utils import transform_rgb
from vista.entities.agents.Dynamics import tireangle2curvature
from vista.tasks.multi_agent_base import MultiAgentBase
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes


def get_env(args):
    trace_config = dict(
        road_width=args.road_width,
        reset_mode=args.reset_mode,
        master_sensor='camera_front',
    )

    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
        use_curvilinear_dynamics=args.use_curvilinear_dynamics,
        road_buffer_size=int(1e5),
        lookahead_road=True,
        control_mode=args.control_mode,
    )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    rig_path = os.path.join(current_dir, 'RIG.xml')
    camera_config = dict(
        type='camera',
        name='camera_front',
        rig_path=rig_path,
        size=(200, 320),
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )

    task_config = dict(
        n_agents=args.n_agents,
        mesh_dir=args.mesh_dir,
        init_dist_range=args.init_dist_range,
        init_lat_noise_range=args.init_lat_noise_range,
    )

    env = MultiAgentBase(
        trace_paths=args.trace_paths,
        trace_config=trace_config,
        car_configs=[car_config] * task_config['n_agents'],
        sensors_configs=[[camera_config]] + [[]] * (task_config['n_agents'] - 1), # no sensor
        task_config=task_config,
        logging_level='WARNING')

    for agent in env.world.agents:
        agent._ego_dynamics._v = 0.
        agent._ego_dynamics._steering = 0.

    return env


def preprocess_obs(env, agent, model, observations):
    sensor_name = 'camera_front'
    obs = observations[agent.id][sensor_name]
    camera = [_s for _s in agent.sensors if _s.name == sensor_name][0]
    roi = camera.camera_param.get_roi()
    imgs = obs[:, :, ::-1].copy() # bgr to rgb
    imgs = imgs[roi[0]:roi[2], roi[1]:roi[3]] # crop roi
    imgs, _ = transform_rgb(imgs, train=False,
        use_color_jitter=model.hparams.use_color_jitter,
        use_fixed_standardize=model.hparams.use_fixed_standardize) # transform used during training
    imgs = imgs[None, None, ...] # add batch and time dimension

    if agent.config['use_curvilinear_dynamics']:
        state_data = np.concatenate([agent.ego_dynamics.numpy(return_xyphi=False),
                                     [agent.ego_dynamics.kappa]], axis=0)
        state_data = torch.from_numpy(state_data[None, None, ...]).float()

        other_agents = [_a for _a in env.world.agents if _a.id != agent.id]
        if len(other_agents) > 0:
            other_s = np.array([_a.ego_dynamics.s for _a in other_agents])
            closest_idx = np.argmin(np.abs(other_s - agent.ego_dynamics.s))
            offset_d = np.sign(other_agents[closest_idx].ego_dynamics.d) * 5.
            obs_data = torch.Tensor([other_agents[closest_idx].ego_dynamics.s,
                                    other_agents[closest_idx].ego_dynamics.d + offset_d])
            obs_data = obs_data[None, None, ...].float()
        else:
            obs_data = torch.stack([state_data[:, :, 0] + model.hparams.non_obstacle_ds,
                                    torch.ones_like(state_data[:, :, 1]) * 8], dim=2)
    else:
        dx, dy, dphi = agent.relative_state.numpy()
        x, y, phi, delta, v = agent.ego_dynamics.numpy()
        kappa = tireangle2curvature(agent.human_dynamics.steering, agent.wheel_base)
        state_data = np.concatenate([np.array([agent.ego_dynamics.s, -dx, dphi, v, delta]),
                                     [kappa]], axis=0)
        state_data = torch.from_numpy(state_data[None, None, ...]).float()

        other_agents = [_a for _a in env.world.agents if _a.id != agent.id]
        if len(other_agents) > 0:
            raise NotImplementedError
        else:
            obs_data = torch.stack([state_data[:, :, 0] + model.hparams.non_obstacle_ds,
                                    torch.ones_like(state_data[:, :, 1]) * 8], dim=2)
    
    ctrl_data = torch.Tensor([agent.ego_dynamics.speed,
                              agent.ego_dynamics.steering,
                              0., 0.]) # NOTE: set dummy a and omega
    ctrl_data = ctrl_data[None, None, ...].float()

    return [imgs, state_data, obs_data, ctrl_data]


def construct_actions(env, agent, model, pred):
    # actions of all agents initialized as zeros
    actions = {_a.id: np.zeros((2,)) for _a in env.world.agents}

    # extract action for the specified agent
    action = []
    for ctrl_name in agent.control_mode.split('-'):
        ctrl = pred[model.hparams.output_mode.index(ctrl_name)]
        if ctrl_name == 'v':
            ctrl = np.clip(ctrl, *agent.ego_dynamics.speed_bound)
        elif ctrl_name == 'delta':
            ctrl = np.clip(ctrl, *agent.ego_dynamics.steering_bound)
        elif ctrl_name == 'a':
            ctrl = np.clip(ctrl, *agent.ego_dynamics.acceleration_bound)
        elif ctrl_name == 'omega':
            ctrl = np.clip(ctrl, *agent.ego_dynamics.steering_rate_bound)
        action.append(ctrl)
    action = np.array(action)

    # set action for the agent
    actions[agent.id] = action

    return actions


class ImageTextFormatter:
    def __init__(self, img, start_xy, step_xy, font_size, font_color=(255, 0, 0)):
        self._img = img
        self._start_xy = np.array(start_xy)
        self._current_xy = self._start_xy.copy()
        self._step_xy = np.array(step_xy)
        self._font_size = font_size
        self._font_color = font_color

    def add_text(self, text):
        self._img = cv2.putText(self._img.copy(),
                                text,
                                tuple(self._current_xy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self._font_size,
                                self._font_color)
        self._current_xy += self._step_xy

        if self._current_xy[1] > 0.95 * self._img.shape[0]:
            self._current_xy[0] += 170 # next column
            self._current_xy[1] = self._start_xy[1]

        return self._img


def add_descriptions(env, agent, model, pred, img):
    texts = [f'model: {model.hparams.model_module}']
    if agent.config['use_curvilinear_dynamics']:
        texts.extend([
            f's: {agent.ego_dynamics.s:.4f}',
            f'd: {agent.ego_dynamics.d:.4f}',
            f'mu: {agent.ego_dynamics.mu:.4f}',
            f'kappa: {agent.ego_dynamics.kappa:.4f}',
        ])
    else:
        texts.extend([
            f'x: {agent.ego_dynamics.x:.4f}',
            f'y: {agent.ego_dynamics.y:.4f}',
            f'yaw: {agent.ego_dynamics.yaw:.4f}',
        ])
    texts.extend([
        f'v: {agent.ego_dynamics.speed:.4f}',
        f'delta: {agent.ego_dynamics.steering:.4f}',
    ])
    for i, ctrl_name in enumerate(model.hparams.output_mode):
        texts.append(f'[pred]{ctrl_name}: {pred[i]:.4f}')
    if hasattr(model, 'intermediate_data'):
        for name, val in model.intermediate_data.items():
            if isinstance(val, (bool, str)):
                texts.append(f'[model]{name}: {val}')
            else:
                texts.append(f'[model]{name}: {val:.4f}')

    aug_img = np.concatenate([np.zeros((img.shape[0], 200, 3)),
                              img], axis=1)
    formatter = ImageTextFormatter(aug_img, [0, 10], [0, 16], 0.45)

    for text in texts:
        img = formatter.add_text(text)

    return img


def extract_logs(env, agent, model=None, pred=None):
    logs = dict()

    logs['x'] = agent.ego_dynamics.x
    logs['y'] = agent.ego_dynamics.y
    logs['yaw'] = agent.ego_dynamics.yaw
    logs['v'] = agent.ego_dynamics.speed
    logs['delta'] = agent.ego_dynamics.steering
    logs['dx'], logs['dy'], logs['dyaw'] = agent.relative_state.numpy()
    if agent.config['use_curvilinear_dynamics']:
        logs['s'] = agent.ego_dynamics.s
        logs['d'] = agent.ego_dynamics.d
        logs['mu'] = agent.ego_dynamics.mu
        logs['kappa'] = agent.ego_dynamics.kappa
    logs['trace_path'] = agent.trace.trace_path
    logs['trace_index'] = agent.trace_index
    logs['segment_index'] = agent.segment_index
    logs['frame_index'] = agent.frame_index

    if model is not None and pred is not None:
        for i, ctrl_name in enumerate(model.hparams.output_mode):
            logs[f'control/{ctrl_name}'] = pred[i]

        if hasattr(model, 'intermediate_data'):
            for name, val in model.intermediate_data.items():
                logs[f'model/{name}'] = val

    return logs
