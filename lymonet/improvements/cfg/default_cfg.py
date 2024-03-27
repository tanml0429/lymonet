
import copy
from lymonet.apis.yolov8_api import (
    IterableSimpleNamespace,
    DEFAULT_CFG_DICT,
)

LYMO_DEFAULT_CFG_DICT = copy.deepcopy(DEFAULT_CFG_DICT)

LYMO_DEFAULT_CFG_DICT['cls_loss_gain'] = 1.0
LYMO_DEFAULT_CFG_DICT['echo_loss_gain'] = 1.0
LYMO_DEFAULT_CFG_DICT['merge_type'] = None
LYMO_DEFAULT_CFG_DICT['content_loss_gain'] = 0.1
LYMO_DEFAULT_CFG_DICT['texture_loss_gain'] = 0.1
LYMO_DEFAULT_CFG_DICT['load_correspondence'] = False
LYMO_DEFAULT_CFG_DICT['fine_cls'] = False  # 是否使用精细分类模型


LYMO_DEFAULT_CFG = IterableSimpleNamespace(**LYMO_DEFAULT_CFG_DICT)

from ...ultralytics.ultralytics.cfg import *

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = LYMO_DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                )
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                )
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                )

    # Return instance
    return IterableSimpleNamespace(**cfg)