import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from templr.config import get, load_default_config, parse_config_path


def has_option(argv, option):
    return any(arg == option or arg.startswith(option + '=') for arg in argv)


def apply_run_defaults(command, command_argv):
    config_path = parse_config_path(command_argv)
    config = load_default_config(config_path)

    if command in {'train', 'test'} and not has_option(command_argv, '--dataset'):
        dataset = get(config, 'run', 'dataset', None)
        if dataset is not None and str(dataset).strip().upper() not in {'', 'NAN', 'NONE', 'NULL'}:
            command_argv = ['--dataset', str(dataset)] + command_argv

    return command_argv


def main():
    commands = {
        'train': ('templr.training', 'main'),
        'test': ('templr.testing', 'main'),
        'submit': ('templr.submission', 'main'),
        'check-data': ('templr.core', 'check_data_main'),
        'check-submission': ('templr.core', 'check_submission_main'),
        'local-mrr': ('templr.core', 'local_mrr_main'),
    }

    if len(sys.argv) >= 2 and sys.argv[1] in {'-h', '--help'}:
        print('Usage: python scripts/run.py [{train,test,submit,check-data,check-submission,local-mrr}] [options]')
        print('If command is omitted, run.default_command from config is used.')
        return

    argv = sys.argv[1:]
    if argv and argv[0] in commands:
        command = argv[0]
        command_argv = argv[1:]
    else:
        config_path = parse_config_path(argv)
        config = load_default_config(config_path)
        command = get(config, 'run', 'default_command', 'submit')
        command_argv = argv

    if command not in commands:
        choices = ', '.join(commands)
        raise SystemExit(f'Unknown command "{command}". Choices: {choices}')

    command_argv = apply_run_defaults(command, command_argv)

    module_name, function_name = commands[command]
    module = __import__(module_name, fromlist=[function_name])
    getattr(module, function_name)(command_argv)


if __name__ == '__main__':
    main()
