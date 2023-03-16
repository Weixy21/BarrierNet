import os
import numpy as np
import pickle
from shapely.geometry import box as Box
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib import cm

from vista.entities.agents.Dynamics import curvature2tireangle
from vista.tasks.multi_agent_base import MultiAgentBase
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import transform


def get_default_camera_config():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    rig_path = os.path.join(current_dir, 'RIG.xml')
    return {
        'type': 'camera',
        'name': 'camera_front',
        'rig_path': rig_path,
        'size': (200, 320),
        'depth_mode': DepthModes.FIXED_PLANE,
        'use_lighting': False,
    }


def setup_default_env(args, use_camera=False):
    trace_config = dict(
        road_width=4,
        reset_mode='segment_start',
        master_sensor='camera_front',
    )

    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
        road_buffer_size=20000,
        lookahead_road=True,
        control_mode='delta-v',
    )

    if use_camera:
        sensors_config = [get_default_camera_config()]
    else:
        sensors_config = []

    task_config = dict(
        n_agents=args.n_agents,
        mesh_dir=args.mesh_dir,
        init_dist_range=None, # randomly distributed over the entire trace
        init_lat_noise_range=[1., 1.5],
    )

    env = MultiAgentBase(
        trace_paths=args.trace_paths,
        trace_config=trace_config,
        car_configs=[car_config] * task_config['n_agents'],
        sensors_configs=[sensors_config] + [[]] * (task_config['n_agents'] - 1), # no sensor
        task_config=task_config,
        logging_level='DEBUG')

    return env


def state2poly(state, car_dim):
    poly = Box(state[0] - car_dim[0] / 2., state[1] - car_dim[1] / 2.,
               state[0] + car_dim[0] / 2., state[1] + car_dim[1] / 2.)
    poly = affinity.rotate(poly, np.degrees(state[2]))

    return poly


def generate_human_actions(world):
    actions = dict()
    for agent in world.agents:
        if agent.control_mode == 'kappa-v':
            action = np.array([
                agent.trace.f_curvature(agent.timestamp),
                agent.trace.f_speed(agent.timestamp)
            ])
        elif agent.control_mode == 'delta-v':
            kappa = agent.trace.f_curvature(agent.timestamp)
            delta = curvature2tireangle(kappa, agent.wheel_base)
            action = np.array([
                delta,
                agent.trace.f_speed(agent.timestamp)
            ])
        else:
            raise NotImplementedError(f'Unsupported control mode {agent.control_mode}')
        actions[agent.id] = action

    return actions


def fetch_privileged_info(world, agent):
    ref_pose = agent.ego_dynamics.numpy()[:3].copy()

    # get ado cars state w.r.t. agent
    other_agents = [_a for _a in world.agents if _a.id != agent.id]
    other_states = []
    for other_agent in other_agents:
        other_latlongyaw = transform.compute_relative_latlongyaw(
            other_agent.ego_dynamics.numpy()[:3], ref_pose)
        other_latlongyawlw = np.concatenate(
            [other_latlongyaw, [other_agent.length, other_agent.width]])
        other_states.append(other_latlongyawlw)

    # get road w.r.t. the agent
    road = np.array(agent.road)[:, :3].copy()
    road_in_agent = np.array(
        [transform.compute_relative_latlongyaw(_v, ref_pose) for _v in road])

    return road_in_agent, other_states

def check_traces(env, trace_paths):
    def _get_name(v):
        v = v.split('/')
        v = v[-2] if v[-1] == '' else v[-1]
        return v

    for i, trace in enumerate(env.world.traces):
        trace_name_in_frozen_sim = _get_name(trace.trace_path)
        trace_name_in_paths = _get_name(trace_paths[i])
        if trace_name_in_frozen_sim != trace_name_in_paths:
            proceed = input(f'>> Different trace name {trace_name_in_frozen_sim}' + 
                f'vs {trace_name_in_paths}. Proceed? [yes/no]: ') == 'yes'
            if not proceed:
                continue
        env.world.traces[i]._trace_path = os.path.expanduser(trace_paths[i])


def parse_control_data(data):
    return {
        's': data[:, 0], # progress
        'd': data[:, 1], # lateral displacement
        'mu': data[:, 2], # local heading error
        'v': data[:, 3], # speed
        'delta': data[:, 4], # steering angle
        'kappa': data[:, 5], # road curvature
        'a': data[:, 6], # acceleration
        'omega': data[:, 7], # steering rate
        'x': data[:, 8], # x position
        'y': data[:, 9], # y position
        'phi': data[:, 10], # heading
        'jerk': data[:, 11], # jerk
        'steering_acc': data[:, 12], # steering acceleration
    }


class PrivilegedInfoVisualizer:
    def __init__(self, env, vis_agent_ids=None):
        fig, axes = plt.subplots(1, len(vis_agent_ids))
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        for ai, agent in enumerate(env.world.agents):
            if agent.id not in vis_agent_ids:
                continue
            axes[ai].set_title(f'Agent ({agent.id})')
        artists = dict()

        fig.tight_layout()
        fig.show()

        self._env = env
        self._vis_agent_ids = [env.world.agents[0]] if vis_agent_ids is None else vis_agent_ids

        self._fig = fig
        self._axes = axes
        self._artists = artists

    def update(self, privileged_info):
        for ai, (aid, pinfo) in enumerate(privileged_info.items()):
            if aid not in self._vis_agent_ids:
                continue
            agent = [_a for _a in self._env.world.agents if _a.id == aid][0]

            self._update_road_vis(pinfo[0], self._axes[ai], self._artists, f'{aid}:road')

            other_car_dims = [(_a.width, _a.length) for _a in self._env.world.agents 
                              if _a.id != agent.id]
            ego_car_dim = (agent.width, agent.length)
            self._update_car_vis(pinfo[1], other_car_dims, ego_car_dim,
                                 self._axes[ai], self._artists, f'{aid}:ado_car')

        self._fig.tight_layout()
        self._fig.canvas.draw()

    def save_snapshot(self, path_prefix):
        img_path = path_prefix + '.png'
        pkl_path = path_prefix + '.figax'
        self._fig.savefig(img_path, dpi=1200)
        with open(pkl_path, 'wb') as __f:
            pickle.dump([self._fig, self._axes], __f)

    def _update_car_vis(self, other_states, other_car_dims, ego_car_dim, 
                        ax, artists, name_prefix):
        # clear car visualization at previous timestamp
        for existing_name in artists.keys():
            if name_prefix in existing_name:
                artists[existing_name].remove()

        # initialize some helper object
        colors = list(cm.get_cmap('Set1').colors) + list(
            cm.get_cmap('Set2').colors) + list(cm.get_cmap('Set3').colors)
        poly_i = 0

        # plot ego car (reference pose; always at the center)
        ego_poly = state2poly([0., 0., 0.], ego_car_dim)
        artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
            ego_poly.exterior.coords.xy[0],
            ego_poly.exterior.coords.xy[1],
            c=colors[poly_i % len(colors)],
        )
        poly_i += 1

        # plot ado cars
        for other_state, other_car_dim in zip(other_states, other_car_dims):
            other_poly = state2poly(other_state, other_car_dim)
            artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
                other_poly.exterior.coords.xy[0],
                other_poly.exterior.coords.xy[1],
                c=colors[poly_i % len(colors)],
            )
            poly_i += 1

    def _update_road_vis(self, road, ax, artists, name, plot_arrow=True, arrow_steps=50):
        if name in artists.keys():
            artists[name].remove()
        artists[name], = ax.plot(road[:, 0], road[:, 1], c='k', linewidth=2, linestyle='dashed')
        if plot_arrow:
            theta = -road[:, 2]
            dxy = np.stack([np.sin(theta), np.cos(theta)], axis=1)
            for i in range(road.shape[0]):
                if i % arrow_steps != 0:
                    continue
                arrow_name = name + f'_arrow_{i}'
                if arrow_name in artists.keys():
                    artists[arrow_name].remove()
                artists[arrow_name] = ax.arrow(road[i, 0], road[i, 1], dx=dxy[i, 0], 
                    dy=dxy[i, 1], head_width=5, head_length=10, shape='right')
                ax.text(road[i, 0] + dxy[i, 0] * 10, road[i, 1] + dxy[i, 1] * 10,
                        f'{theta[i]:.2f}', c='b')
