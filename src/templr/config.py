from __future__ import annotations

import argparse
import os
from typing import Any

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')
MODEL_CHOICES = {'baseline', 'mynet'}


def load_default_config(path: str | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Default config must be a mapping: {config_path}')
    return data


def parse_config_path(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args(argv)
    return args.config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'Config section "{name}" must be a mapping')
    return value


def get(config: dict[str, Any], section_name: str, key: str, default: Any) -> Any:
    return section(config, section_name).get(key, default)


def get_run_model_name(config: dict[str, Any]) -> str:
    model_name = get(config, 'run', 'model', None)
    if model_name is None:
        raise ValueError('Missing required config value: run.model')
    model_name = str(model_name).strip().lower()
    if model_name not in MODEL_CHOICES:
        choices = ', '.join(sorted(MODEL_CHOICES))
        raise ValueError(f'Unknown run.model "{model_name}". Choices: {choices}')
    return model_name


def get_model_config(config: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    selected = (model_name or get_run_model_name(config)).strip().lower()
    if selected not in config:
        raise ValueError(f'Missing required config section: {selected}')
    return section(config, selected)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)
