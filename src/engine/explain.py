'''
Plain-language explanation of what the bot is doing and why (web UI 「目前狀態」).

explain(bot, rec, now) turns one decision-trace record (MapleStoryAutoBot.
record_decision: the why entries of the frame + the keyboard events) into
  key      a short id of the state, for the history of state changes
  main     one line in Traditional Chinese
  details  more lines (route underneath, potions, buffs ...)
Pure string work, a few microseconds per frame.
'''
# Standard import
import re

DIR = {"left": "左", "right": "右"}
COLOUR_NAME = {
    (255, 0, 0): "紅色", (0, 0, 255): "藍色", (255, 127, 0): "橘色", (0, 255, 255): "青色",
    (127, 255, 0): "黃綠色", (255, 0, 255): "洋紅色", (0, 255, 127): "淺綠色", (255, 255, 0): "黃色",
    (127, 127, 127): "灰色", (255, 255, 127): "淺黃色", (0, 127, 255): "湛藍色",
}
_KEY_RE = re.compile(r"^(press|key_down|key_up) (\w+)")


def route_text(cmd, colour=None):
    '''"right none jump" -> 「路線：往右跳」 etc.'''
    if not cmd:
        return "沒有路線指令"
    x, y, a = (cmd.split() + ["none", "none", "none"])[:3]
    col = f"（{COLOUR_NAME.get(tuple(colour), '')}）" if colour and tuple(colour) in COLOUR_NAME else ""
    if a == "goal":
        return f"到達路線終點，換下一條路線{col}"
    if x == "stop":
        return f"路線：停{col}"
    if y == "up" and a == "jump":
        return f"路線：跳上繩子（按住上＋跳）{col}"
    if y == "up":
        return f"路線：往上爬{col}"
    if y == "down" and a == "jump":
        return f"路線：往下跳{col}"
    if y == "down":
        return f"路線：往下爬{col}"
    if a == "jump" and x in DIR:
        return f"路線：往{DIR[x]}跳上平台{col}"
    if a == "jump":
        return f"路線：原地跳{col}"
    if x in DIR:
        return f"照路線往{DIR[x]}走{col}"
    return f"路線：不動{col}"


def _mob_name(bot, name):
    names = (getattr(bot, "data", None) or {}).get("eng_to_cn", {})
    return names.get(name, name or "怪")


def _fmt_s(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}" if s >= 60 else f"{s} 秒"


def key_events_text(bot, keys):
    '''potions / buffs among the keyboard events of the frame'''
    cfg = bot.cfg
    from src.input.WingpuKeyboard import to_key_code
    hp_code = to_key_code(cfg["key"].get("add_hp") or "")
    mp_code = to_key_code(cfg["key"].get("add_mp") or "")
    b = cfg.get("buff_skill") or {}
    buffs = {to_key_code(k): (n or k) for k, n in
             zip(b.get("keys") or [], list(b.get("names") or []) + [""] * 20) if k}
    hm = bot.health_monitor
    out = []
    for txt in keys or []:
        m = _KEY_RE.match(txt)
        if not m or m.group(1) != "press":
            continue
        code = m.group(2)
        if hp_code and code == hp_code:
            out.append(f"喝紅水（HP {hm.hp_percent:.0f}% < {cfg['health_monitor']['add_hp_percent']:g}%）" if hm else "喝紅水")
        elif mp_code and code == mp_code:
            out.append(f"喝藍水（MP {hm.mp_percent:.0f}% < {cfg['health_monitor']['add_mp_percent']:g}%）" if hm else "喝藍水")
        elif code in buffs:
            out.append(f"放 Buff：{buffs[code]}")
    return out


def explain(bot, rec, now):
    '''
    (key, main, details) for a decision record
    '''
    kb = bot.kb
    why = rec.get("why") or []
    kinds = {w["kind"]: w for w in why}
    route = kinds.get("route")
    details = []
    if route:
        details.append(route_text(route.get("cmd"), route.get("colour")) +
                       ("（附近沒有路線顏色，沿用上一個動作）" if route.get("kept_previous") else ""))
    details += key_events_text(bot, rec.get("keys"))

    if kb is not None and not kb.is_enable:
        return "paused", "暫停中（不送任何按鍵）", details
    if "curse_seen" in kinds:
        details.insert(0, "畫面上有詛咒（經驗值、掉寶 0）" +
                       ("，休息時間到了，照常練功" if kinds["curse_seen"].get("skip") else "，「解符文」沒開"))
    if "lost_pause" in kinds:
        what = "、".join({"minimap": "小地圖", "player": "角色"}.get(r, r) for r in kinds["lost_pause"].get("reason", []))
        t_seen = bot.t_player_seen if "player" in kinds["lost_pause"].get("reason", []) else bot.t_minimap_ok
        age = f" {now - t_seen:.1f} 秒" if t_seen else ""
        return "lost", f"偵測不到{what}{age}，已停手、放開所有鍵", details
    mg = getattr(bot, "minigame", None)
    if "rune_game" in kinds:
        w = kinds["rune_game"]
        seq = " ".join({"up": "↑", "right": "→", "down": "↓", "left": "←"}[d] for d in w.get("seq", []))
        n = (w.get("total") or 0) - len(w.get("seq", [])) + 1
        return "rune_game", f"符文小遊戲：{seq}（第 {n} 個）" + (f"，重試 {w['tries']} 次" if w.get("tries") else ""), details
    if mg:
        from src.engine.RuneGame import sequence_text
        n = (getattr(bot, "mg_total", 0) or len(mg)) - len(mg) + 1
        details.insert(0, f"畫面上有符文小遊戲：{sequence_text(mg)}（第 {n} 個；bot 沒有在解）")
    if "rune_sweep" in kinds:
        w = kinds["rune_sweep"]
        details.insert(0, f"接近原因：{w.get('why', '')}；已找 {w.get('elapsed', 0):.0f} 秒、按上 {w.get('presses', 0)} 次")
        return "rune_sweep", f"符文：接近中 → 小步左右找（±{w.get('radius', 0)} px）＋按上", details
    if "rune_near" in kinds:
        return "rune_near", f"符文：就在附近（{kinds['rune_near'].get('why', '')}）→ 先停下", details
    if "rune_seek" in kinds:
        w = kinds["rune_seek"]
        how = w.get("how")
        where = {"left": "往左", "right": "往右", "down": "往下（下跳）", "up": "往上（按上）",
                 "gone": "箭頭不見了，應該就在符文旁"}.get(how, "")
        ang = f"，箭頭 {w['angle']:.0f}°" if w.get("angle") is not None else ""
        main = f"詛咒中：跟著黃色箭頭找符文，{where}{ang}（{w.get('elapsed', 0):.0f} / {w.get('timeout', 120):.0f} 秒）"
        if "rune_interact" in kinds:
            details.insert(0, f"按「{kinds['rune_interact'].get('key')}」試著解放符文（第 {kinds['rune_interact'].get('n')} 次）")
        return f"rune_seek_{how}", main, details
    if "rune_to_safe" in kinds:
        return "rune_to_safe", "詛咒中：前往安全點，到了就停手", details
    if "rune_rest" in kinds:
        w = kinds["rune_rest"]
        left = w.get("left")
        if left is not None:
            return "rune_rest", f"詛咒中，休息 {_fmt_s(left)} 後再練功", details
        since = w.get("since") or 0
        why = w.get("why") or ""
        return "rune_rest", f"詛咒中：停止練功，等詛咒解除（已 {_fmt_s(since)}）" + (f"　{why}" if why else ""), details
    if "minimap_grace" in kinds:
        w = kinds["minimap_grace"]
        return "minimap_grace", (f"小地圖被擋住 {w.get('age', 0):.1f} 秒：只原地攻擊"
                                 f"（{w.get('limit', 10):g} 秒後停手）"), details
    if "cycle" in kinds:
        hc = bot.hunt_cycle.status() if bot.hunt_cycle is not None else {}
        ph = kinds["cycle"].get("phase")
        left = hc.get("remaining_s")
        if ph == "rest":
            return "rest", f"休息中，剩 {_fmt_s(left or 0)}", details
        if ph == "to_safe":
            plan = kinds["cycle"].get("plan") or {}
            how = {"route_safe": "照安全路線", "same_level": "同一層直接走"}.get(plan.get("how"), "")
            return "to_safe", f"前往安全點{('（' + how + '）') if how else ''}", details
        if ph == "wait_window":
            return "wait_window", "不在每日時段內，停手中", details
        if ph == "ended":
            return "ended", f"已結束：{hc.get('ended_why', '')}", details
    if "human_pause" in kinds:
        w = kinds["human_pause"]
        hp = getattr(bot, "human_pause", None)
        details.insert(0, f"這次停 {w.get('dur', 0):.1f} 秒" + ("（偶爾停久一點）" if w.get("long") else "")
                       + "；結束後照常練功，下一次在幾分鐘後（隨機）")
        return "human_pause", f"類人保護：暫停操作 {w.get('left', 0):.1f} 秒（只補血）", details
    if "human_pause_deferred" in kinds:
        why = {"ladder": "在爬繩子 / 梯子", "runup": "在助跑跳", "rune": "在處理符文",
               "minimap_grace": "小地圖被擋住"}.get(kinds["human_pause_deferred"].get("reason"), "")
        details.append(f"類人保護：時間到了，等{why}結束再停")
    if "jump_runup" in kinds:
        step = {"start": 1, "back": 1, "approach": 2, "jump": 3}.get(kinds["jump_runup"].get("phase"), 1)
        word = {1: "往回走", 2: "跑回牆邊", 3: "起跳"}[step]
        return f"runup{step}", f"跳不上去 → 退後助跑（第 {step} 步：{word}）", details
    if "stuck_random" in kinds:
        t = bot.cfg["watchdog"]["timeout"]
        return "stuck", f"卡住 {t:g} 秒沒移動 → 隨機動作", details
    if "aoe_attack" in kinds:
        w = kinds["aoe_attack"]
        key = str(w.get("key") or "").upper()
        side = f"，面向{DIR.get(w.get('side'), '')}邊" if w.get("side") else ""
        if w.get("mode") == "aoe":
            return "aoe_cast", f"群體技能：範圍內 {w.get('n')} 隻（{key}）{side}", details
        return "aoe_cast", f"群體技能：範圍內 {w.get('n')} 隻 ≥ {w.get('need')}（{key}）{side}", details
    aoe = kinds.get("aoe_count")
    aoe_pre = ""
    if aoe:
        r = aoe.get("reason")
        if r == "count":
            aoe_pre = f"單體攻擊：範圍內 {aoe.get('n')} 隻 < {aoe.get('need')}"
        elif r == "cooldown":
            if aoe.get("mode") == "aoe":
                return "aoe_wait", f"群體技能冷卻中，再 {aoe.get('left', 0):.1f} 秒（範圍內 {aoe.get('n')} 隻）", details
            aoe_pre = f"群體技能冷卻中（範圍內 {aoe.get('n')} 隻），先單體攻擊"
        elif r == "no_key":
            aoe_pre = "沒設群體技能鍵，改單體攻擊"
        if aoe_pre and "attack_target" not in kinds:
            details.insert(0, aoe_pre)
    if "attack_target" in kinds:
        w = kinds["attack_target"]
        d = w.get("dir")
        tgt = w.get("target")
        dist = ""
        if tgt:
            dist = f"距離 {abs(tgt[0] + tgt[2] / 2 - rec['player'][0]):.0f} px"
            if bot.wedge_params() is not None:
                dist += "，扇形內"
        name = _mob_name(bot, w.get("name"))
        main = f"攻擊{DIR.get(d, '')}邊的{name}" + (f"（{dist}）" if dist else "")
        if aoe_pre:
            main = f"{aoe_pre}：{main}"
        return f"attack_{d}", main, details
    if "stand_target" in kinds:
        w = kinds["stand_target"]
        if w.get("waiting") == "cooldown":
            return "stand_cd", f"站著等攻擊冷卻（目標在{DIR.get(w.get('dir'), '')}邊）", details
        hold = float(bot.cfg["directional_attack"].get("target_hold_s", 0.5))
        rest = max(0.0, hold - (now - getattr(bot, "t_target_seen", now)))
        return "stand_hold", f"目標剛消失，再站 {rest:.1f} 秒", details
    if "dead_zone_back_off" in kinds:
        return "back_off", f"怪太近（在盲區），往{DIR.get(kinds['dead_zone_back_off'].get('to'), '')}退開", details
    if "edge" in kinds:
        return "edge", "平台邊緣，跳回", details
    if route:
        main = details.pop(0)
        if "attack_cooldown" in kinds:
            main += "（攻擊冷卻中）"
        cmd = (route.get("cmd") or "none none none").split()
        return f"route_{cmd[0]}_{cmd[1]}_{cmd[2]}", main, details
    return "idle", "待命", details
