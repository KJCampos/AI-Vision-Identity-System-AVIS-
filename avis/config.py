"""
config.yaml -> argparse defaults. Command-line flags still override the file.
Sections in the YAML are only for readability; keys inside them are flag names
(e.g. match_thresh under 'recognition:' == --match-thresh).
"""
import logging
import os
import re

try:
    import yaml
except ImportError:  # pyyaml ships with ultralytics; this is just a safety net
    yaml = None


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        logging.warning("[CONFIG] pyyaml not installed; ignoring %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    flat = {}
    for key, val in data.items():
        if isinstance(val, dict):
            for k, v in val.items():
                flat[str(k).replace("-", "_")] = v
        else:
            flat[str(key).replace("-", "_")] = val
    return flat


def apply_config(parser, path):
    """Set parser defaults from the config file; warn about keys that aren't real options."""
    cfg = load_config(path)
    if not cfg:
        return {}
    dests = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg if k not in dests)
    if unknown:
        logging.warning("[CONFIG] unknown keys in %s ignored: %s", os.path.basename(path), unknown)
    good = {k: v for k, v in cfg.items() if k in dests}
    parser.set_defaults(**good)
    return good


def set_config_value(path, key, value):
    """Change one 'key: value' line in config.yaml in place, keeping comments and layout."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(#.*)?)$", re.M)
    if not pat.search(text):
        raise KeyError(f"{key} not found in {path}")
    text = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
