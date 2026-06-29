import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(opt):
    env_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    opt.distributed = bool(getattr(opt, 'distributed', False) or env_distributed)

    if opt.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed training requires CUDA/NCCL, but CUDA is not available')
        opt.rank = int(os.environ.get('RANK', 0))
        opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
        default_local_rank = getattr(opt, 'local_rank', -1)
        if default_local_rank < 0:
            default_local_rank = 0
        opt.local_rank = int(os.environ.get('LOCAL_RANK', default_local_rank))
        torch.cuda.set_device(opt.local_rank)
        if not dist.is_available():
            raise RuntimeError('torch.distributed is not available')
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        opt.device = torch.device('cuda', opt.local_rank)
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.device = torch.device('cuda' if torch.cuda.is_available() and not getattr(opt, 'cpu', False) else 'cpu')

    return opt


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def get_amp_dtype(opt):
    return torch.float16 if getattr(opt, 'amp_dtype', 'fp16') == 'fp16' else torch.bfloat16


def _fsdp_mixed_precision(opt):
    if not getattr(opt, 'fsdp_mixed_precision', False):
        return None
    from torch.distributed.fsdp import MixedPrecision

    dtype = get_amp_dtype(opt)
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _fsdp_cpu_offload(opt):
    if not getattr(opt, 'fsdp_cpu_offload', False):
        return None
    from torch.distributed.fsdp import CPUOffload

    return CPUOffload(offload_params=True)


def wrap_model_for_training(model, opt):
    if not getattr(opt, 'distributed', False):
        return model

    if getattr(opt, 'dist_backend', 'fsdp') == 'ddp':
        find_unused_parameters = bool(
            getattr(opt, 'ddp_find_unused_parameters', False)
            or (
                getattr(opt, 'fcb_reparam', False)
                and getattr(opt, 'fcb_reparam_freeze_global_before', False)
                and getattr(opt, 'fcb_reparam_epoch', 0) > 0
            )
            or not getattr(opt, 'fcb_use_sim_mask', True)
        )
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    if getattr(opt, 'dist_backend', 'fsdp') == 'fsdp':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(
            model,
            device_id=opt.local_rank,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
            mixed_precision=_fsdp_mixed_precision(opt),
            cpu_offload=_fsdp_cpu_offload(opt),
        )

    raise ValueError(f"Unsupported dist_backend: {getattr(opt, 'dist_backend', None)}")


def get_model_state_dict_for_save(model, opt):
    if getattr(opt, 'distributed', False) and getattr(opt, 'dist_backend', None) == 'fsdp':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()

    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()
