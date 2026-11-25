"""SuperMimic + AMP 起身训练入口；默认算法配置见 agents/x2_supermimic_cfg.py。"""
import sys

sys.dont_write_bytecode = True

from getup_config import handle_meta_arguments


def main():
    """先处理 --print-config / --help，再交给运行时启动训练。"""
    if handle_meta_arguments(sys.argv[1:]):
        return

    from getup_runtime import run_training

    run_training()


if __name__ == '__main__':
    main()
