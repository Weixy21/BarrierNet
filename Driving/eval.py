import os
import argparse
import importlib
import json
import pickle
import numpy as np
from skvideo.io import FFmpegWriter
import torch

from eval_tools import utils
import vista


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run VISTA for evaluation')
    # model
    parser.add_argument('--model-module',
                        type=str,
                        required=True,
                        help='Model class; should be consistent with the checkpoint')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--state-net-model-module',
                        type=str,
                        default=None,
                        help='Model class for state net; should be consistent with the checkpoint')
    parser.add_argument('--state-net-ckpt',
                        type=str,
                        default=None,
                        help='Path to checkpoint for state net model')
    parser.add_argument('--ego-init-v',
                        type=float,
                        default=6.,
                        help='Initial velocity used for acceleration control')
    parser.add_argument('--set-obs-d-lower-bound',
                        type=float,
                        default=None,
                        help='Lower bound for predicted obs_d')
    parser.add_argument('--use-reference-control',
                        action='store_true',
                        default=False,
                        help='Use reference control')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Not use cuda')
    parser.add_argument('--lf-cbf-threshold',
                        type=float,
                        default=2.,
                        help='Threshold for lane following CBF')
    parser.add_argument('--use-lf-cbf-only',
                        action='store_true',
                        default=False,
                        help='Use lane following CBF only')
    # general vista arguments
    parser.add_argument('--loadmat',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Load .pkl for env and .mat file for control')
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
                        default=2,
                        help='Number of agents')
    parser.add_argument('--use-display',
                        action='store_true',
                        default=False,
                        help='Use VISTA default display')
    parser.add_argument('--road-width',
                        type=float,
                        default=4.,
                        help='Road width in VISTA')
    parser.add_argument('--reset-mode',
                        type=str,
                        default='default',
                        choices=['default', 'segment_start', 'uniform'],
                        help='Trace reset mode in VISTA')
    parser.add_argument('--use-curvilinear-dynamics',
                        action='store_true',
                        default=False,
                        help='Use curvilinear dynamics for vehicle dynamics')
    parser.add_argument('--max-step',
                        type=int,
                        default=None,
                        help='Maximal step to be ran')
    parser.add_argument('--init-dist-range',
                        type=float,
                        nargs='+',
                        default=[15., 25.],
                        help='Initial distance range of obstacle')
    parser.add_argument('--init-lat-noise-range',
                        type=float,
                        nargs='+',
                        default=[1., 1.5],
                        help='Initial lateral displacement of obstacle')
    # evaluation and logging
    parser.add_argument('--n-episodes',
                        type=int,
                        default=1,
                        help='Number of episodes')
    parser.add_argument('--out-dir',
                        type=str,
                        default=None,
                        help='Directory to save output')
    parser.add_argument('--save-video',
                        action='store_true',
                        default=False,
                        help='Save video for every episodes')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    LitModel = importlib.import_module(f'.{args.model_module}', 'models').LitModel
    model = LitModel.load_from_checkpoint(args.ckpt)
    if set(model.hparams.output_mode) == set(['delta', 'v']):
        args.control_mode = 'delta-v'
    elif set(model.hparams.output_mode) == set(['omega', 'a']):
        args.control_mode = 'omega-a'
    elif set(model.hparams.output_mode[:4]) == set(['delta', 'v', 'omega', 'a']):
        args.control_mode = 'omega-a'
    else:
        raise NotImplementedError(f'No corresponding control mode for {model.hparams.output_mode}')
    if args.model_module == 'barrier_net':
        model_kwargs = {'solver': 'cvxpy',
                        'store_intermediate_data': True}
        model.hparams.not_use_gt = True
        if args.use_reference_control:
            model.hparams.model_type = 'deri_ref'
    elif args.model_module == 'old_barrier_net':
        model_kwargs = {'solver': 'cvxpy',
                        'store_intermediate_data': True}
        model.hparams.not_use_gt = True
        model.hparams.use_color_jitter = False
        model.hparams.use_fixed_standardize = True
    else:
        model_kwargs = dict()
    if not args.no_cuda:
        model.cuda()
    model.eval()

    if args.state_net_model_module:
        LitModelStateNet = importlib.import_module(f'.{args.state_net_model_module}', 'models').LitModel
        model_state_net = LitModelStateNet.load_from_checkpoint(args.state_net_ckpt)
        if not args.no_cuda:
            model_state_net.cuda()
        model_state_net.eval()
        model.hparams.use_indep_state_net = True
        model.hparams.indep_state_net_output = model_state_net.hparams.output_mode
    
    model.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model.hparams.use_lf_cbf_only = args.use_lf_cbf_only

    if args.loadmat:
        from scipy.io import loadmat, savemat
        use_camera = True

        with open(args.loadmat[0], 'rb') as f:
            env = pickle.load(f)

        for i in range(len(env.world.traces)):
            trace_name_in_frozen_env = env.world.traces[i].trace_path.split(
                '/')
            trace_name_in_frozen_env = trace_name_in_frozen_env[
                -2] if trace_name_in_frozen_env[
                    -1] == '' else trace_name_in_frozen_env[-1]
            trace_name_in_args = args.trace_paths[i].split('/')
            trace_name_in_args = trace_name_in_args[-2] if trace_name_in_args[
                -1] == '' else trace_name_in_args[-1]
            if trace_name_in_frozen_env != trace_name_in_args:
                assert trace_name_in_frozen_env == trace_name_in_args
                old_trace_path = env.world.traces[i]._trace_path
                env.world.traces[i]._trace_path = args.trace_paths[i]
                different_trace = True
            else:
                different_trace = False
            env.world.traces[i]._trace_path = os.path.expanduser(
                args.trace_paths[i])

        display_config = dict(road_buffer_size=1000, )
        task_config = env.config

        dt = 1. / 10

        if len(args.loadmat) == 2:
            mat_key = ['state_control_pos', 'data'][1]
            state_control = loadmat(args.loadmat[1])[mat_key]

            if different_trace:  # NOTE: doens't work
                import utm
                assert len(env.world.traces) == 1
                gps_old = np.genfromtxt(os.path.join(old_trace_path,
                                                     'gps.csv'),
                                        delimiter=',')[1:]
                gps_xy_old = np.array(
                    [utm.from_latlon(v[0], v[1])[:2] for v in gps_old[:, 1:3]])
                gps_origin_old = gps_xy_old[0]

                gps_new = np.genfromtxt(os.path.join(args.trace_paths[0],
                                                     'gps.csv'),
                                        delimiter=',')[1:]
                gps_xy_new = np.array(
                    [utm.from_latlon(v[0], v[1])[:2] for v in gps_new[:, 1:3]])
                gps_origin_new = gps_xy_new[0]

                gps_origin_diff = gps_origin_new - gps_origin_old
                # state_control[:, 8:10] += gps_origin_diff

        ego_agent = env.world.agents[0]
        ego_agent_id = ego_agent.id
        ego_control = {'acc': 0., 'vel': 6., 's_ang_vel': 0., 's': 0, 'd': 0}

        # HACK
        for ag in env.world.agents:
            ag._control_mode = 'omega-a'
            ag.human_dynamics._s = 0.
            ag.ego_dynamics._s = 0.
            ag.relative_state._s = 0.
            ag._road_dynamics._s = 0.
            ag.ego_dynamics._omega_bound = [-1.0, 1.0]
            ag.ego_dynamics._a_bound = [-7.0, 7.0]

        if False:  # filter out non-relevant agents
            ego_xy = state_control[i, 8:10].copy()
            obs_dist_to_ego = [
                np.linalg.norm(_a.ego_dynamics.numpy()[:2] -
                               ego_xy) if _a.id != ego_agent_id else 0.
                for _a in env.world.agents
            ]
            relevant_agents_idcs = np.argsort(
                obs_dist_to_ego)[:3]  # ego + 2 closest
            env.world._agents = [
                env.world.agents[_idx] for _idx in relevant_agents_idcs
            ]

        for _a in env.world.agents:
            _a._config['use_curvilinear_dynamics'] = False
        if args.use_curvilinear_dynamics:  # to test curvilinear dynamics
            from vista.entities.agents.Dynamics import CurvilinearDynamics
            ego_agent._config['use_curvilinear_dynamics'] = True
            ego_agent._config['lookahead_road'] = True
            ego_agent._config['road_buffer_size'] = 1e5

            if not isinstance(ego_agent.ego_dynamics, CurvilinearDynamics):
                # old_ego_dynamics = ego_agent.ego_dynamics.copy()
                ego_agent._ego_dynamics = CurvilinearDynamics()
                if True:
                    ego_agent.reset(*env.world.sample_new_location())
                else:
                    env._world._agents = env.world.agents[0:4]
                    env.reset()
            ego_agent._ego_dynamics._v = 6  # NOTE: start with speed 6

        # if use_camera:
        #     from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes

        #     if not hasattr(env, '_meshlib'):
        #         from vista.entities.sensors.MeshLib import MeshLib
        #         env._meshlib = MeshLib(env.config['mesh_dir'])
        #         env._reset_meshlib()

        #     examples_path = os.path.dirname(os.path.realpath(__file__))
        #     camera_config = dict(
        #         type='camera',
        #         # camera params
        #         name='camera_front',
        #         rig_path=os.path.join(examples_path, "RIG.xml"),
        #         size=(200, 320),
        #         # rendering params
        #         depth_mode=DepthModes.FIXED_PLANE,
        #         use_lighting=False,
        #     )
        #     camera = ego_agent.spawn_camera(camera_config)
        #     ego_agent.trace.multi_sensor.set_main_sensor('camera', camera.name)
        #     camera.reset()

        loaded_env = env

        env = utils.get_env(args)
        ego_agent = env.world.agents[0]

        args.ego_init_v = 6

        closest_dist_in_loaded_env = 9999.
        for ag_i, ag in enumerate(loaded_env.world.agents):
            if ag_i == 0:
                continue
            dist = np.linalg.norm(ag.ego_dynamics.numpy()[:2] - ego_agent.ego_dynamics.numpy()[:2])
            if dist < closest_dist_in_loaded_env:
                closest_dist_in_loaded_env = dist
                closest_agent_in_loaded_env = ag

        env.world.agents[1]._ego_dynamics = closest_agent_in_loaded_env.ego_dynamics.copy()
    else:
        env = utils.get_env(args)
        ego_agent = env.world.agents[0]
        ego_agent._ego_dynamics._v = args.ego_init_v
    if args.use_display:
        display_config = dict(road_buffer_size=1000, )
        display = vista.Display(env.world, display_config=display_config)
        display.reset()  # reset should be called after env reset
    dt = 1 / 10.

    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

        config_path = os.path.join(args.out_dir, 'eval_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        all_results = []

    stop_eval = False
    for episode_i in range(args.n_episodes):
        print(f'Episode {episode_i:04d}')
        if not args.loadmat:
            observations = env.reset()
        else:
            observations = env.reset()
            # env.world.agents[1]._ego_dynamics = closest_agent_in_loaded_env.ego_dynamics.copy()
            # observations, rewards, dones, infos = env.step({ego_agent.id: np.zeros((2,))}, dt)
        if args.use_display:
            display.reset()  # reset should be called after env reset
        ego_agent._ego_dynamics._v = args.ego_init_v

        if args.out_dir:
            episode_results = []

        if args.save_video:
            assert args.out_dir is not None
            video_dir = os.path.join(args.out_dir, 'video')
            if not os.path.isdir(video_dir):
                os.makedirs(video_dir)
            video_path = os.path.join(video_dir, f'episode_{episode_i:04d}.mp4')
            rate = f'{(1. / dt)}'
            video_writer = FFmpegWriter(video_path,
                                        inputdict={'-r': rate},
                                        outputdict={'-vcodec': 'libx264',
                                                    '-pix_fmt': 'yuv420p',
                                                    '-r': rate,})

        done = False
        if hasattr(model, 'get_initial_state'):
            rnn_state = model.get_initial_state(batch_size=1)
            if not args.no_cuda:
                rnn_state = [_v.cuda() for _v in rnn_state]
        else:
            rnn_state = None
        if args.state_net_model_module:
            if hasattr(model_state_net, 'get_initial_state'):
                rnn_state_state_net = model_state_net.get_initial_state(batch_size=1)
                if not args.no_cuda:
                    if isinstance(rnn_state_state_net, list):
                        rnn_state_state_net = [_v.cuda() for _v in rnn_state_state_net]
                    elif isinstance(rnn_state_state_net, dict):
                        for k, v in rnn_state_state_net.items():
                            assert isinstance(v, list)
                            rnn_state_state_net[k] = [vv.cuda() for vv in v]
                    else:
                        raise NotImplementedError
            else:
                rnn_state_state_net = None
        step_i = 0
        while not done:
            try:
                model_inputs = utils.preprocess_obs(env, ego_agent, model, observations)
                if not args.no_cuda:
                    model_inputs = [_v.cuda() for _v in model_inputs]
                with torch.no_grad():
                    if args.state_net_model_module:
                        pred_state_net, rnn_state_state_net = model_state_net(model_inputs, rnn_state_state_net, **model_kwargs)
                        if args.set_obs_d_lower_bound is not None:
                            obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                            if torch.sign(pred_state_net[:, :, obs_d_idx]) > 0:
                                pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                    pred_state_net[:, :, obs_d_idx], min=args.set_obs_d_lower_bound)
                            else:
                                pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                    pred_state_net[:, :, obs_d_idx], max=-args.set_obs_d_lower_bound)
                        if model_state_net.hparams.drop_obs_d_offset:
                            if 'obs_d' in model_state_net.hparams.output_mode:
                                obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                                pred_state_net[:, :, obs_d_idx] += \
                                    torch.sign(pred_state_net[:, :, obs_d_idx]) * 5
                            if 'dd' in model_state_net.hparams.output_mode:
                                dd_idx = model_state_net.hparams.output_mode.index('dd')
                                pred_state_net[:, :, dd_idx] += \
                                    torch.sign(pred_state_net[:, :, dd_idx]) * 5
                        model_inputs.append(pred_state_net)
                    pred, rnn_state = model(model_inputs, rnn_state, **model_kwargs)
                pred = pred.cpu().numpy()[0, 0] # drop batch and time dimension
                actions = utils.construct_actions(env, ego_agent, model, pred)

                observations, rewards, dones, infos = env.step(actions, dt)
                done = dones[ego_agent.id]

                if args.out_dir:
                    logs = dict()
                    for _a in env.world.agents:
                        if _a.id == ego_agent.id:
                            logs[_a.id] = utils.extract_logs(env, _a, model, pred)
                        else:
                            logs[_a.id] = utils.extract_logs(env, _a)
                    step_results = {
                        'ego_agent_id': ego_agent.id,
                        'logs': logs,
                        'actions': actions,
                        'infos': infos,
                    }
                    episode_results.append(step_results)

                if args.use_display:
                    img = display.render()
                    img = utils.add_descriptions(env, ego_agent, model, pred, img)

                    if args.save_video:
                        video_writer.writeFrame(img)

                step_i += 1

                if args.max_step:
                    done = False # don't use terminal condition defined in env
                    if step_i >= args.max_step:
                        has_crashed = np.any([v['infos'][ego_agent.id]['crashed'] for v in episode_results])
                        print('n_passed: {}'.format(step_results['infos'][ego_agent.id]['n_passed'])
                            + ' has_crashed: {}'.format(has_crashed))
                        break
            except KeyboardInterrupt:
                if args.save_video:
                    video_writer.close()
                stop_eval = True
                break
            # except:
            #     break

        print(step_results['infos'][ego_agent.id], step_i)

        if args.out_dir:
            all_results.append(episode_results)

        if args.save_video:
            video_writer.close()
        if stop_eval:
            break

    if args.out_dir:
        results_path = os.path.join(args.out_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
