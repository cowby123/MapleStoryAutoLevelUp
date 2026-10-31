'''
WingpuVM
Shared connection to a wingpu VM (the game runs inside a QEMU/KVM Windows guest).

Inside the VM's script container the wingpu SDK (`import wingpu`) reads the guest
screen from /wingpu/frame (shared memory) and sends input through /wingpu/input.sock.
Outside the container it can use the manager's network API instead
(env WINGPU_URL / WINGPU_VM / WINGPU_TOKEN, or cfg wingpu.url / vm / token).

Only one connection per process: the capturor (frames) and the keyboard (input)
share it.
'''
# Standard import
import os
import sys
import threading

# Local import
from src.utils.logger import logger

_lock = threading.Lock()
_vm = None


def _import_sdk():
    '''
    Import the wingpu SDK. It is pre-installed in the script container image;
    elsewhere point WINGPU_SDK at the directory holding wingpu.py
    (wingpu repo: tools/containers/sdk).
    '''
    sdk_dir = os.environ.get("WINGPU_SDK")
    if sdk_dir and sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    try:
        import wingpu  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise RuntimeError(
            "[WingpuVM] Cannot import the wingpu SDK. Run inside the VM's script "
            "container, or set WINGPU_SDK=<dir containing wingpu.py>") from e
    return wingpu


def get_vm(cfg=None):
    '''
    Return the process-wide wingpu VM handle (created on first call).
    '''
    global _vm
    with _lock:
        if _vm is not None:
            return _vm
        wcfg = (cfg or {}).get("wingpu", {}) or {}
        fps = 10
        if cfg is not None:
            fps = cfg["system"]["fps_limit_window_capturor"]
        sdk = _import_sdk()
        url = wcfg.get("url") or None
        kw = {}
        if url:
            kw = {"url": url, "vm": wcfg.get("vm") or None, "token": wcfg.get("token") or None}
        _vm = sdk.connect(fps=fps, **kw)
        logger.info(f"[WingpuVM] Connected ({type(_vm).__name__}, fps wish {fps})")
        return _vm


def close_vm():
    '''
    Close the shared connection (if any).
    '''
    global _vm
    with _lock:
        if _vm is not None:
            try:
                _vm.close()
            except Exception:
                pass
        _vm = None
