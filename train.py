import threading
import torch
from torch import nn
from arguments import parser
from models import Model
from collections import deque
import os
from utils import create_optimizers, get_batch, log
import time
import requests
import numpy as np
import pprint
import timeit


mean_episode_return_buf = {p: deque(maxlen=100) for p in ['white', 'black']}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def learn(position,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x = torch.flatten(batch['obs_x'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    with lock:
        learner_outputs = model(obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        return stats


def train(flags):
    """
    这个函数主要用来训练。它首先初始化所有内容，如缓冲区、优化器等。然后它将启动子进程来模拟。
    然后，它将调用多线程学习功能。
    """
    if flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    print(flags.training_device)
    # Learner model for training
    learner_model = Model(device=flags.training_device, board_size=flags.board_size)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_white',
        'loss_white',
        'mean_episode_return_black',
        'loss_black',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'white': 0, 'black': 0}
    # 导入模型
    print(flags.load_model, os.path.exists(checkpointpath))
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['white', 'black']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
    batch_num = {"white": 1, "black": 1}

    def batch_and_learn(i, position, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats, batch_num
        while frames < flags.total_frames:
            batch = get_batch(position, batch_num[position])
            batch_num[position] += 1
            _stats = learn(position, learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                frames += T * B
                position_frames[position] += T * B

    threads = []
    position_locks = {'white': threading.Lock(), 'black': threading.Lock()}

    for i in range(flags.num_threads):
        for position in ['white', 'black']:
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i,
                args=(i, position, position_locks[position]))
            thread.start()
            threads.append(thread)

    def checkpoint(frames):
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['white', 'black']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '/var/ftp/pub/%s' % (position + '_weights' + '.ckpt')))
            print(model_weights_dir)
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)
        with open("/var/ftp/pub/model_version.txt", "w+") as f:
            f.write(str(frames))

    fps_log = []
    timer = timeit.default_timer
    try:
        last_send_time = timer()
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)
            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info('After %i (W:%i B:%i) frames: @ %.1f fps (avg@ %.1f fps) (W:%.1f B:%.1f) Stats:\n%s',
                     frames,
                     position_frames['white'],
                     position_frames['black'],
                     fps,
                     fps_avg,
                     position_fps['white'],
                     position_fps['black'],
                     pprint.pformat(stats))
            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            if timer() - last_send_time > 30 * 60:
                message = 'After {} (W:{} B:{}) frames: @ {} fps (avg@ {} fps) (W:{} B:{}) Stats:\n{}'.format(
                    frames,
                    position_frames['white'],
                    position_frames['black'],
                    fps,
                    fps_avg,
                    position_fps['white'],
                    position_fps['black'],
                    pprint.pformat(stats))
                requests.get(url='http://82.156.187.209:5700/send_private_msg?user_id=329309742&message={}'.format(message))
                last_send_time = timer()

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
    checkpoint(frames)


if __name__ == "__main__":
    flags = parser.parse_args(['--actor_device_cpu', '--load_model'])
    train(flags)








