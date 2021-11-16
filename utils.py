import torch
import typing
import logging
import traceback
import pickle
import zlib
import time
from env import Env
from env_utils import Environment

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('hexzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)


def create_env(flags):
    return Env(flags.chess_num)


def get_batch(position, batch_num):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    while True:
        with open("batch/{}_batch_num.log".format(position), "r") as f:
            total_batch = int(f.read())
        print(position, total_batch, batch_num)
        if total_batch >= batch_num:
            break
        time.sleep(0.5)
    try:
        with open("batch/{}_batch_{}".format(position, batch_num), "rb") as f:
            batch = f.read()
        batch = zlib.decompress(batch)
        batch = pickle.loads(batch)
    except:
        batch = get_batch(position, batch_num + 1)
    return batch


def create_buffers(flags, device_iterator):
    """
    为不同的位置以及不同的设备(如GPU)创建缓冲区。也就是说，每个设备对于两个位置都有3倍长度作为特征值
    """
    T = flags.unroll_length  # 展开长度
    positions = ['white', 'black']
    buffers = {}
    for device in device_iterator:  # 每个设备
        buffers[device] = {}
        for position in positions:  # 每个位置
            x_dim = flags.chess_num * 3
            specs = dict(
                target=dict(size=(T,), dtype=torch.float32),
                obs_x=dict(size=(T, x_dim), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):  # num_buffers:50
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['white', 'black']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        target_buf = {p: [] for p in positions}
        obs_x_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
                with torch.no_grad():
                    agent_output = model.forward(position, obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                env_output['obs_x_no_action'][action * 3] = 0
                if position == "white":
                    env_output['obs_x_no_action'][action * 3 + 1] = 1
                else:
                    env_output['obs_x_no_action'][action * 3 + 2] = 1
                obs_x_buf[position].append(env_output['obs_x_no_action'])
                size[position] += 1
                position, obs, env_output = env.step(action)

                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            episode_return = env_output['episode_return'] if p == 'white' else -env_output[
                                'episode_return']
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                while size[p] > T:
                    # free_queue出队
                    index = free_queue[p].get()
                    print("index", index)
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x'][index][t, ...] = obs_x_buf[p][t]
                    # full_queue入队
                    full_queue[p].put(index)
                    target_buf[p] = target_buf[p][T:]
                    obs_x_buf[p] = obs_x_buf[p][T:]
                    size[p] -= T
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def create_optimizers(flags, learner_model):
    """
    为两个位置创建RMSprop优化器
    """
    positions = ['white', 'black']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers
