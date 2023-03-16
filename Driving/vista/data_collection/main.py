import os
import argparse
import numpy as np
import pickle
from scipy.io import savemat, loadmat
import cv2
from skvideo.io import FFmpegWriter

import vista
from vista.entities.sensors.MeshLib import MeshLib
from . import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run VISTA with multiple cars for data collection')
    # general vista arguments
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--mesh-dir',
                        type=str,
                        default=None,
                        help='Directory of meshes for virtual agents')
    parser.add_argument('--n-agents',
                        type=int,
                        default=40,
                        help='Number of agents')
    parser.add_argument('--use-display',
                        action='store_true',
                        default=False,
                        help='Use VISTA default display')
    # data collection and visualization
    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        choices=['collect_init', 'collect_imgs', 'rollout_delta_v',
                                 'rollout_omega_a'],
                        help='Data collection mode')
    parser.add_argument('--visualize-privileged-info',
                        action='store_true',
                        default=False,
                        help='Visualize privileged information')
    parser.add_argument('--video-path',
                        type=str,
                        default=None,
                        help='Path to store video')
    # mode `collect_init`
    parser.add_argument('--init-mat-path',
                        type=str,
                        default=None,
                        help='Path to store mat file in `collect_init` mode')
    parser.add_argument('--init-frozen-sim-path',
                        type=str,
                        default=None,
                        help='Path to store frozen sim in `collect_init` mode')
    # mode `collect_imgs`, `rollout_delta_v`, `rollout_omega_a`
    parser.add_argument('--load-frozen-sim',
                        type=str,
                        default=None,
                        help='Load frozen VISTA; default as None to not load')
    parser.add_argument('--load-control',
                        type=str,
                        default=None,
                        help='Load control data; default as None to not load')
    parser.add_argument('--imgs-dir',
                        type=str,
                        default=None,
                        help='Directory to save images')
    args = parser.parse_args()

    return args


def main():
    # setup environment
    args = parse_args()
    if args.load_frozen_sim is None:
        dt = 1 / 30. # NOTE: we use 30hz in Devens data collection
        env = utils.setup_default_env(args)
        env.reset()
        ego_agent = env.world.agents[0]
    else:
        dt = 1 / 10. # NOTE: we use 10hz in MPC
        control_data = loadmat(args.load_control)['data']
        control_data = utils.parse_control_data(control_data)

        with open(args.load_frozen_sim, 'rb') as f:
            env = pickle.load(f)
        utils.check_traces(env, args.trace_paths)
        ego_agent = env.world.agents[0]

        if not hasattr(env, '_meshlib'):
            env._meshlib = MeshLib(env.config['mesh_dir'])
            env._reset_meshlib()
        camera_config = utils.get_default_camera_config()
        camera = ego_agent.spawn_camera(camera_config)
        ego_agent.trace.multi_sensor.set_main_sensor('camera', camera.name)
        camera.reset()

    # setup visualizer
    if args.use_display:
        display_config = dict(road_buffer_size=1000, )
        display = vista.Display(env.world, display_config=display_config)
        display.reset()  # reset should be called after env reset

    if args.visualize_privileged_info:
        pi_visualizer = utils.PrivilegedInfoVisualizer(env)

    if args.video_path:
        assert args.use_display, 'Set `use_display` to save video'
        video_path = os.path.abspath(os.path.expanduser(args.video_path))
        rate = f'{(1 / dt)}'
        video_writer = FFmpegWriter(video_path,
                                    inputdict={'-r': rate},
                                    outputdict={'-vcodec': 'libx264',
                                                '-pix_fmt': 'yuv420p',
                                                '-r': rate,})

    # collect data
    if args.mode == 'collect_init':
        assert np.all([_a.control_mode == 'delta-v' for _a in env.world.agents]), \
            'Use control mode `delta-v` for data collection mode `collect_init`'
        assert args.init_mat_path is not None
        assert args.init_frozen_path is not None

        actions = {_a.id: np.zeros((2, )) for _a in env.world.agents}
        _, _, _, _ = env.step(actions, dt)

        privileged_info = dict()
        for agent in env.world.agents:
            privileged_info[agent.id] = utils.fetch_privileged_info(env.world, agent)

        data = {'car_1': privileged_info[ego_agent.id]}
        savemat(args.init_mat_path, data)
        with open(args.init_frozen_sim_path, 'wb') as f:
            pickle.dump(env, f)

        if args.visualize_privileged_info:
            pi_visualizer.update(privileged_info)
            path_prefix = os.path.splitext(args.init_frozen_sim_path)[0]
            pi_visualizer.save_snapshot(path_prefix)
    elif args.mode in ['collect_imgs', 'rollout_delta_v', 'rollout_omega_a']:
        assert args.load_frozen_sim is not None
        assert args.load_control is not None

        for agent in env.world.agents:
            if args.mode in ['collect_imgs', 'rollout_delta_v']:
                agent._ego_dynamics._v = 0.
                agent._ego_dynamics._steering = 0.
                agent._control_mode = 'delta-v'
            elif args.mode == 'rollout_omega_a':
                if agent.id == ego_agent.id:
                    agent._ego_dynamics._v = 6.
                else:
                    agent._ego_dynamics._v = 0.
                agent._ego_dynamics._steering = 0.
                agent._control_mode = 'omega-a'

        step = 0
        while True:
            try:
                actions = {_a.id: np.zeros((2, )) for _a in env.world.agents}
                if args.mode == 'collect_imgs':
                    ego_agent.ego_dynamics._x = control_data['x'][step]
                    ego_agent.ego_dynamics._y = control_data['y'][step]
                    ego_agent.ego_dynamics._yaw = control_data['phi'][step] - np.pi / 2.
                    ego_agent.ego_dynamics._steering = control_data['delta'][step]
                    ego_agent.ego_dynamics._v = control_data['v'][step]
                    actions[ego_agent.id] = np.zeros((2, ))
                elif args.mode == 'rollout_delta_v':
                    actions[ego_agent.id] = np.array([
                        control_data['delta'][step],
                        control_data['v'][step]
                    ])
                elif args.mode == 'rollout_omega_a':
                    actions[ego_agent.id] = np.array([
                        control_data['omega'][step],
                        control_data['a'][step]
                    ])

                observations, _, _, _ = env.step(actions, dt)

                if args.imgs_dir is not None:
                    if not os.path.isdir(args.imgs_dir):
                        os.makedirs(args.imgs_dir)
                    img = observations[ego_agent.id]['camera_front']
                    roi = camera.camera_param.get_roi()
                    img = img[roi[0]:roi[2], roi[1]:roi[3]]
                    cv2.imwrite(f'{args.imgs_dir}/{step:04d}.png', img)

                if args.video_path:
                    img = display.render()
                    video_writer.writeFrame(img)

                step += 1
                if step >= control_data['x'].shape[0]:
                    break
            except KeyboardInterrupt:
                if args.video_path:
                    video_writer.close()
                break
    else:
        raise NotImplementedError(f'Unrecognized mode {args.mode}')


if __name__ == '__main__':
    main()
