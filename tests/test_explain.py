'''
「目前狀態」: plain-language explanation of decision records, the history of
state changes, and its cost.

  python tests/test_explain.py
'''
import json
import logging
import os
import sys
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import wingpu_detect_test as T  # noqa: E402
from src.engine.explain import explain, route_text  # noqa: E402

T.logger.set_level(logging.ERROR)


def bot_():
    T.OVERRIDES.clear()
    T.OVERRIDES.update({"directional_attack.shape": "wedge", "buff_skill.keys": ["a"],
                        "buff_skill.names": ["加速"], "buff_skill.cooldown": [60]})
    b = T.make_bot("wingpu_maple")
    b.kb.is_enable = True
    return b


def rec(why, keys=(), player=(700, 450)):
    return {"t": time.time(), "player": list(player), "why": list(why), "keys": list(keys)}


ROUTE_R = {"kind": "route", "cmd": "right none none", "colour": [0, 0, 255]}


def test_route_text():
    assert route_text("right none none", [0, 0, 255]) == "照路線往右走（藍色）"
    assert route_text("right none jump", [0, 255, 255]) == "路線：往右跳上平台（青色）"
    assert route_text("none up jump") == "路線：跳上繩子（按住上＋跳）"
    assert route_text("none up none", [127, 127, 127]) == "路線：往上爬（灰色）"
    assert route_text("none down jump") == "路線：往下跳"


def test_cases():
    b = bot_()
    hm = b.health_monitor
    hm.hp_percent = 42.0
    cases = [
        ([ROUTE_R, {"kind": "attack_target", "dir": "right", "name": "pig", "target": [900, 430, 60, 45]}], [],
         "攻擊右邊的肥肥（距離 230 px，扇形內）"),
        ([ROUTE_R, {"kind": "stand_target", "dir": "left", "waiting": "cooldown"}], [], "站著等攻擊冷卻（目標在左邊）"),
        ([ROUTE_R], [], "照路線往右走（藍色）"),
        ([dict(ROUTE_R, cmd="right none jump", colour=[0, 255, 255])], [], "路線：往右跳上平台（青色）"),
        ([ROUTE_R, {"kind": "jump_runup", "phase": "approach"}], [], "跳不上去 → 退後助跑（第 2 步：跑回牆邊）"),
        ([ROUTE_R, {"kind": "stuck_random"}], [], "卡住 10 秒沒移動 → 隨機動作"),
        ([ROUTE_R, {"kind": "edge"}], [], "平台邊緣，跳回"),
        ([ROUTE_R, {"kind": "attack_cooldown", "dir": "left"}], [], "照路線往右走（藍色）（攻擊冷卻中）"),
    ]
    for why, keys, want in cases:
        k, main, det = explain(b, rec(why, keys), time.time())
        print(f"  {main}")
        assert main == want, (main, want)
    b.t_target_seen = time.time() - 0.1
    _, main, _ = explain(b, rec([ROUTE_R, {"kind": "stand_target", "dir": "left", "waiting": "hold"}]), time.time())
    assert main == "目標剛消失，再站 0.4 秒", main
    print(f"  {main}")
    b.t_player_seen = time.time() - 2.1
    _, main, _ = explain(b, rec([{"kind": "lost_pause", "reason": ["player"]}]), time.time())
    assert main.startswith("偵測不到角色 2.1 秒，已停手"), main
    print(f"  {main}")
    # potion / buff appear in the details
    _, main, det = explain(b, rec([ROUTE_R], ["press Delete ('delete') 0.05s", "press KeyA ('a') 0.05s"]), time.time())
    assert "喝紅水（HP 42% < 20%）" in det[1] or "喝紅水（HP 42%" in " ".join(det), det
    assert "放 Buff：加速" in det, det
    print(f"  details: {det}")
    b.kb.is_enable = False
    assert explain(b, rec([ROUTE_R]), time.time())[1].startswith("暫停中")
    # rest (hunt cycle)
    from src.engine.HuntCycle import HuntCycle
    b.kb.is_enable = True
    b.hunt_cycle = HuntCycle({"enable": True, "hunt_minutes": [1, 1], "rest_minutes": [3.4, 3.4]})
    b.hunt_cycle._enter("rest", "test")
    _, main, _ = explain(b, rec([{"kind": "cycle", "phase": "rest"}]), time.time())
    assert main.startswith("休息中，剩 3:2"), main
    print(f"  {main}")


def test_history_and_cost():
    b = bot_()
    t0 = time.time()
    seq = []
    for i in range(60):   # 3 s: attack left 1 frame, walk right 2 frames (the live oscillation)
        if i % 3 == 0:
            why = [ROUTE_R, {"kind": "attack_target", "dir": "left", "name": "pig", "target": [400, 430, 60, 45]}]
        else:
            why = [ROUTE_R]
        r = rec(why)
        r["t"] = t0 + i * 0.05
        b.update_now_state(r)
        seq.append(b.now_state["main"])
    h = list(b.state_history)
    print(f"  history ({len(h)} kept): " + " → ".join(x['main'][:8] for x in h[-6:]))
    assert len(h) == 20 and h[-1]["dur"] in (0.05, 0.1)
    # cost per frame
    r = rec([ROUTE_R, {"kind": "attack_target", "dir": "left", "name": "pig", "target": [400, 430, 60, 45]}],
            ["press Delete ('delete') 0.05s"])
    n = 5000
    c = time.perf_counter()
    for _ in range(n):
        b.update_now_state(r)
    us = 1e6 * (time.perf_counter() - c) / n
    print(f"  cost: {us:.1f} us per frame -> {us * 20 / 1e6 * 100:.3f}% of a core at 20 frames/s")
    assert us < 500


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
