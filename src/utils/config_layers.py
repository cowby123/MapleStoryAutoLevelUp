'''
Config layers.

  config/config_default.yaml
  -> config/config_wingpu.yaml or config/config_macOS.yaml (platform)
  -> config/config_<--cfg>.yaml
  -> user override (wingpu: /work/msbot_config.yaml, written by the web UI)

The shipped files are never written; the web UI only writes the user override.
'''
# Standard import
import os

# Local import
from src.utils.common import (load_yaml, override_cfg, set_platform, is_wingpu, is_mac)
from src.utils.logger import logger


def default_user_override():
    '''
    /work/msbot_config.yaml in the wingpu script container, else ./msbot_config.yaml
    '''
    env = os.environ.get("MSBOT_USER_CONFIG")
    if env:
        return env
    base = "/work" if os.path.isdir("/work") and os.access("/work", os.W_OK) else "."
    return os.path.join(base, "msbot_config.yaml")


def build_cfg(cfg_name, user_override="default"):
    '''
    Merged config dict. user_override: path, "default" (default_user_override()),
    or None (skip the user layer).
    '''
    cfg = load_yaml("config/config_default.yaml")
    cfg_user = load_yaml(f"config/config_{cfg_name}.yaml")
    if user_override == "default":
        user_override = default_user_override()
    over = {}
    if user_override and os.path.exists(user_override):
        over = load_yaml(user_override)
    # Platform: env WINGPU > override / config system.platform > auto
    plat = (over.get("system", {}) or {}).get("platform") or \
        (cfg_user.get("system", {}) or {}).get("platform") or \
        cfg["system"].get("platform", "auto")
    set_platform(plat)
    if is_wingpu() and user_override:
        # first start with profiles: the user file is split (profiles.migrate)
        from src.utils.profiles import Profiles
        if Profiles(user_override).migrate() and os.path.exists(user_override):
            over = load_yaml(user_override) or {}
    if is_wingpu():
        cfg = override_cfg(cfg, load_yaml("config/config_wingpu.yaml"))
    elif is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    cfg = override_cfg(cfg, cfg_user)
    if over:
        # "live" in the user file only counts with remember_live (the web UI's
        # "remember" box); otherwise every start is a dry run
        w = over.get("wingpu", {}) or {}
        if w.get("dry_run") is False and not w.get("remember_live", False):
            w = dict(w)
            w["dry_run"] = True
            over = dict(over)
            over["wingpu"] = w
        cfg = override_cfg(cfg, over)
        logger.info(f"[config] user override {user_override} applied")
    # per-character profile (wingpu: /work/profiles/<active>.yaml), after the
    # shared file; its global keys (if any) are ignored
    if user_override and is_wingpu():
        from src.utils.profiles import Profiles, split
        prof = Profiles(user_override)
        pid = prof.active()
        if pid:
            _, part = split(prof.settings(pid))
            if part:
                cfg = override_cfg(cfg, part)
            logger.info(f"[config] profile {pid} ({prof.read(pid)[0].get('name')}) applied")
    return cfg
