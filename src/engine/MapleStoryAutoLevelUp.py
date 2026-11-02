'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import glob
import sys
import logging
import os
import datetime
import threading

# Library import
import numpy as np
import cv2
import yaml

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import (find_pattern_sqdiff, draw_rectangle, screenshot, nms,
    load_image, get_mask, get_minimap_loc_size, get_player_location_on_minimap,
    is_mac, override_cfg, load_yaml, get_all_other_player_locations_on_minimap,
    click_in_game_window, mask_route_colors, to_opencv_hsv, debug_minimap_colors,
    activate_game_window, is_img_16_to_9, normalize_pixel_coordinate, resize_window,
    is_windows, is_wingpu, set_platform
)
from src.utils.resolution import (compute_scale, apply_scale, snapshot_base,
    scale_template, sc, MinimapScale, GREEN)
from src.utils.nametag import NametagTracker, default_dir as nametag_default_dir
from src.input.KeyBoardController import KeyBoardController, press_key
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
elif is_windows() and not is_wingpu(): # WINGPU=1: offline tools on a Windows box
    from src.input.GameWindowCapturor import GameWindowCapturor
else:
    GameWindowCapturor = None # Linux: only the wingpu platform
from src.engine.HealthMonitor import HealthMonitor
from src.engine.Profiler import Profiler
from src.engine.RuneSolver import RuneSolver
from src.engine.FiniteStateMachine import FiniteStateMachine
from src.states.hunting import HuntingState
from src.states.finding_rune import FindingRuneState
from src.states.near_rune import NearRuneState
from src.states.solving_rune import SolvingRuneState
from src.states.auxiliary import AuxiliaryState
from src.states.patrol import PatrolState

class MapleStoryAutoBot:
    '''
    MapleStoryAutoBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryAutoBot
        '''
        self.args = args # User args
        self.cfg = None # Configuration
        self.idx_routes = 0 # Index of route map
        self.monsters_info = {} # monster information
        self.monsters = [] # monster detected in current frame
        self.fps = 0 # Frame per second
        self.red_dot_center_prev = None # previous other player location in minimap
        self.video_writer = None # For video recording feature
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
        self.thread_auto_bot = None # thread for running autobot
        self.cmd_move_x = "none" # "left" "right"
        self.cmd_move_y = "none" # "up" "down"
        self.cmd_action = "none" # "jump" "attack" ....
        # Signals (for UI)
        self.image_debug_signal = None
        self.route_map_viz_signal = None
        # Flags
        self.is_first_frame = True # first frame flag
        self.is_terminated = False # Close all object and thread if True
        self.is_on_ladder = False # Character is on ladder or not
        self.is_show_debug_window = not args.disable_viz #
        self.is_need_show_debug_window = not args.disable_viz #
        self.is_disable_control = args.disable_control
        self.is_ui = args.is_ui # Whether is using UI framework to invoke engine
        self.is_frame_done = False #
        self.frame_count = 0 # frames processed by the main loop
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_party_red_bar = (0, 0) # party red bar location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()
        # Images
        self.img_map = None
        self.img_routes = []
        self.img_nametag = None
        self.img_nametag_gray = None
        self.img_create_party_enable = None
        self.img_create_party_disable = None
        self.img_login_button = None
        # Resolution awareness (src/utils/resolution.py)
        self.scale = 1.0 # UI/world scale of the game frame vs the template reference frame
        self.frame_hw = None # (h, w) of the game frame the scale was computed for
        self.cfg_base = None # unscaled copy of the scaled config sections
        self.tpl_src = {} # templates as loaded from disk (reference scale)
        self.minimap_scale = None # MinimapScale: live minimap vs minimaps/<map>/map.png
        self.t_minimap_scale_try = 0.0
        self.img_minimap_route = np.zeros((10, 10, 3), dtype=np.uint8) # minimap at route-map scale
        self.minimap_rect = None # (x, y, w, h) white minimap frame, for ROI search
        self.monster_min_area = 0
        self.monster_min_h = 0
        self.monster_contour = {} # (name, i) -> blurred black-pixel mask of the template
        # Detection-lost safety pause
        self.t_minimap_ok = 0.0 # last frame with minimap + player dot found
        self.t_player_seen = 0.0 # last frame with the player found on screen
        self.is_lost_paused = False
        self.lost_reason = []
        # Player detection (player_detect.mode: party_bar | nametag | auto)
        self.nametag = None # NametagTracker (template captured from the game, /work/nametag/)
        self.player_source = None # "party_bar" | "nametag" | None (this frame)
        self.t_nametag_check = 0.0
        self.mob_search_box = None # (x0, y0, x1, y1) of the last monster search
        self.det_counts = {"frames": 0, "minimap": 0, "player_dot": 0, "player": 0,
                           "player_party_bar": 0, "player_nametag": 0,
                           "monsters": 0, "lost_paused": 0}

        # Database
        self.data = load_yaml("config/config_data.yaml")
        # Threads & Objects
        self.kb = None # Keyboard controller
        self.capture = None # Game window capturor
        self.health_monitor = None # Health monitor
        self.profiler = None # Profiler, for performance issue debugging
        self.rune_solver = None # Rune solver

        # Finite State Machine
        self.fsm = FiniteStateMachine()
        self.fsm.add_state(HuntingState    ("hunting"     , self))
        self.fsm.add_state(FindingRuneState("finding_rune", self))
        self.fsm.add_state(NearRuneState   ("near_rune"   , self))
        self.fsm.add_state(SolvingRuneState("solving_rune", self))
        self.fsm.add_state(AuxiliaryState  ("aux"         , self))
        self.fsm.add_state(PatrolState     ("patrol"      , self))
        self.fsm.add_transition("hunting", "finding_rune") # When saw a "Rune has created" messgae
        self.fsm.add_transition("finding_rune", "hunting") # After finding rune timeout
        self.fsm.add_transition("finding_rune", "near_rune") # When detect a nearby rune
        self.fsm.add_transition("finding_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("near_rune", "finding_rune") # After rune solving timeout
        self.fsm.add_transition("near_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("solving_rune", "hunting") # After rune solving
        self.fsm.set_init_state("hunting")

    def update_signals(self, image_debug_signal, route_map_viz_signal):
        '''
        Update signal from UI framework.
        For debug window viz
        '''
        self.image_debug_signal = image_debug_signal
        self.route_map_viz_signal = route_map_viz_signal

    def load_config(self, cfg):
        '''
        load_config
        '''
        # Parse color code in config
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        self.color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }

        if cfg["bot"]["mode"] == "normal":
            map_name = cfg['bot']['map']
            # Check if the map is supported in config_data.yaml
            if map_name not in self.data["map_mobs_mapping"]:
                text = f"Invalid map name: {map_name}. "\
                        "Not supported in config/config_data.yaml."
                logger.error(text)
                return -1
                # raise RuntimeError(text)

            # Load map.png from minimaps/
            self.img_map = load_image(f"minimaps/{map_name}/map.png",
                                      cv2.IMREAD_COLOR)
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            # route.files: only these routes, in this order (web UI); [] = all
            chosen = list(cfg["route"].get("files") or [])
            if chosen:
                by_name = {os.path.basename(p): p for p in route_files}
                missing = [r for r in chosen if r not in by_name]
                if missing:
                    logger.warning(f"[load_config] route.files not in minimaps/{map_name}: {missing}")
                route_files = [by_name[r] for r in chosen if r in by_name] or route_files
            self.img_routes = []
            for route_file in route_files:
                img = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
                # Remove pixel in map that is color code
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code"])
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code_up_down"])
                self.img_routes.append(img)

            # Minimap scale (live minimap vs map.png), measured on the first frames
            self.minimap_scale = MinimapScale(self.img_map)

            # Load monsters images from monster/<monster_name>
            self.tpl_src["monsters"] = {}
            for monster_name in self.data["map_mobs_mapping"][map_name]:
                imgs = [load_image(file) for file in
                        sorted(glob.glob(f"monster/{monster_name}/{monster_name}*.png"))]
                if imgs:
                    self.tpl_src["monsters"][monster_name] = imgs
                else:
                    logger.error(f"No images found in monster/{monster_name}/{monster_name}*")
                    return -1
                    # raise RuntimeError(f"No images found in monster/{monster_name}/{monster_name}*")
            logger.info(f"Loaded monsters: {list(self.tpl_src['monsters'].keys())}")

        # Load player's name tag
        if cfg["nametag"]["enable"]:
            self.tpl_src["nametag"] = load_image(f"nametag/{cfg['nametag']['name']}.png")

        # Load misc image
        lang = cfg["system"]["language"]
        self.tpl_src["party_enable"]  = load_image(f"misc/party_button_create_enable_{lang}.png")
        self.tpl_src["party_disable"] = load_image(f"misc/party_button_create_disable_{lang}.png")
        self.tpl_src["login_button"] = load_image(f"misc/login_button_{lang}.png")

        # Normalized pixel coordinate configuration
        cfg['rune_warning_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_solver']['arrow_box_coord'] = normalize_pixel_coordinate(
            cfg['rune_solver']['arrow_box_coord'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_top_left'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_top_left'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_bottom_right'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_bottom_right'], cfg['game_window']['size'])

        # Print mode on log
        logger.info(f"[load_config] Config AutoBot as {cfg['bot']['mode']} mode")

        # Update cfg
        self.cfg = cfg
        # Unscaled values of everything on_resolution() rescales
        self.cfg_base = snapshot_base(cfg)
        self.build_templates(1.0)

        return 0 # load successfully

    def build_templates(self, s):
        '''
        (Re)build every template at scale s (1.0 = as on disk) and the values
        derived from them.
        '''
        self.monsters_info = {}
        self.monster_contour = {}
        self.monster_color = {}
        self.monster_color_pre = {}
        blur = self.cfg["monster_detect"]["contour_blur"]
        for name, imgs in self.tpl_src.get("monsters", {}).items():
            pairs = []
            for img in imgs:
                img = scale_template(img, s)
                for im in (img, cv2.flip(img, 1)):
                    pairs.append((im, get_mask(im, GREEN)))
            self.monsters_info[name] = pairs
            ds = self.monster_match_downscale()
            for i, (im, _) in enumerate(pairs):
                black = np.all(im == [0, 0, 0], axis=2).astype(np.uint8) * 255
                tb = cv2.GaussianBlur(black, (blur, blur), 0)
                if ds > 1:
                    tb = cv2.resize(tb, (max(1, tb.shape[1] // ds), max(1, tb.shape[0] // ds)),
                                    interpolation=cv2.INTER_AREA)
                self.monster_contour[(name, i)] = tb
                # colour template + mask at 1/ds for "color" mode
                im_ds, mk_ds = im, pairs[i][1]
                if ds > 1:
                    size = (max(1, im.shape[1] // ds), max(1, im.shape[0] // ds))
                    im_ds = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
                    mk_ds = cv2.resize(mk_ds, size, interpolation=cv2.INTER_NEAREST)
                self.monster_color[(name, i)] = (im_ds, mk_ds)
                if self.cfg["monster_detect"].get("color_prefilter", False):
                    size = (max(1, im_ds.shape[1] // 2), max(1, im_ds.shape[0] // 2))
                    self.monster_color_pre[(name, i)] = (
                        cv2.resize(im_ds, size, interpolation=cv2.INTER_AREA),
                        cv2.resize(mk_ds, size, interpolation=cv2.INTER_NEAREST))
        sizes = [im.shape[:2] for pairs in self.monsters_info.values() for im, _ in pairs]
        self.monster_min_area = min((h * w for h, w in sizes), default=0)
        self.monster_min_h = min((h for h, _ in sizes), default=sc(40, s))
        if "nametag" in self.tpl_src:
            self.img_nametag = scale_template(self.tpl_src["nametag"], s)
            self.img_nametag_gray = cv2.cvtColor(self.img_nametag, cv2.COLOR_BGR2GRAY)
        self.img_create_party_enable = scale_template(self.tpl_src["party_enable"], s, key=None)
        self.img_create_party_disable = scale_template(self.tpl_src["party_disable"], s, key=None)
        self.img_login_button = scale_template(self.tpl_src["login_button"], s, key=None)
        if self.rune_solver is not None:
            self.rune_solver.set_scale(s)

    def color_matches(self, img_ds, img_pre, name, i_tpl):
        '''
        "color" mode: (x, y, score) in img_ds pixels where the masked template
        matches with SQDIFF_NORMED <= diff_thres. With the prefilter, the
        template is first matched at half that size (loose threshold) and only
        the windows around those candidates are matched at full size
        (fixtures: same frames with hits, half the CPU).
        '''
        thr = self.cfg["monster_detect"]["diff_thres"]
        tpl, mask = self.monster_color[(name, i_tpl)]
        th, tw = tpl.shape[:2]
        if th > img_ds.shape[0] or tw > img_ds.shape[1]:
            return []
        windows = [(0, 0, img_ds.shape[1], img_ds.shape[0])]
        pre = self.monster_color_pre.get((name, i_tpl)) if img_pre is not None else None
        if pre is not None and pre[0].shape[0] <= img_pre.shape[0] and pre[0].shape[1] <= img_pre.shape[1]:
            res = cv2.matchTemplate(img_pre, pre[0], cv2.TM_SQDIFF_NORMED, mask=pre[1])
            res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
            cand = (res <= float(self.cfg["monster_detect"].get("color_prefilter_thres", 0.16))).astype(np.uint8)
            if not cand.any():
                return []
            n, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
            windows = []
            for i in range(1, n):
                x, y, w, h = stats[i][:4]
                wx0, wy0 = max(0, 2 * x - 3), max(0, 2 * y - 3)
                wx1 = min(img_ds.shape[1], 2 * (x + w) + 3 + tw)
                wy1 = min(img_ds.shape[0], 2 * (y + h) + 3 + th)
                windows.append((wx0, wy0, wx1, wy1))
        out = []
        for (wx0, wy0, wx1, wy1) in windows:
            win = img_ds[wy0:wy1, wx0:wx1]
            if win.shape[0] < th or win.shape[1] < tw:
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_SQDIFF_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
            ys, xs = np.where(res <= thr)
            out += [(int(x) + wx0, int(y) + wy0, float(res[y, x])) for y, x in zip(ys, xs)]
        return out

    def monster_match_downscale(self):
        '''
        contour_only matching runs on the blurred outline masks shrunk by this
        factor (1 = upstream). matchTemplate was ~80% of the bot's CPU.
        '''
        return max(1, int(self.cfg["monster_detect"].get("match_downscale", 1)))

    def on_resolution(self, h, w):
        '''
        The game frame is h x w (first frame, or the resolution changed):
        rescale config lengths, UI positions and templates; forget positions.
        '''
        wcfg = self.cfg.get("wingpu", {}) or {}
        s = compute_scale(h, w, wcfg.get("scale", "auto"))
        self.scale = s
        self.frame_hw = (h, w)
        apply_scale(self.cfg, self.cfg_base, s, (h, w))
        self.build_templates(s)
        if self.health_monitor is not None:
            self.health_monitor.set_scale(s)
        if self.nametag is not None:
            self.nametag.set_scale(s)
        if self.minimap_scale is not None:
            self.minimap_scale.m = None
            self.t_minimap_scale_try = 0.0
        self.minimap_rect = None
        self.loc_player = (0, 0)
        self.loc_party_red_bar = (0, 0)
        self.t_minimap_ok = self.t_player_seen = 0.0
        logger.info(f"[on_resolution] game frame {w}x{h}, scale {s:.4f}")

    def init_nametag(self):
        '''
        Nametag tracker from player_detect.nametag_template (default
        /work/nametag/nametag.png, made with the web UI's 截取名牌)
        '''
        pd = self.cfg.get("player_detect", {}) or {}
        path = pd.get("nametag_template") or f"{nametag_default_dir()}/nametag.png"
        self.nametag = NametagTracker(path, thres=float(pd.get("nametag_thres", 0.3)),
                                      local_timeout=float(pd.get("nametag_local_timeout", 1.0)))
        self.nametag.set_scale(self.scale)

    def player_detect_mode(self):
        '''
        party_bar | nametag | auto (legacy: nametag.enable -> upstream nametag code)
        '''
        return (self.cfg.get("player_detect", {}) or {}).get("mode", "party_bar")

    def title_bar_height(self):
        '''
        Rows above the client in click coordinates (wingpu: none, clicks are client pixels)
        '''
        return 0 if is_wingpu() else self.cfg["game_window"]["title_bar_height"]

    def is_wingpu_dry_run(self):
        '''
        True while wingpu input is not being sent to the guest (dry run)
        '''
        from src.input.WingpuKeyboard import get_keyboard
        return get_keyboard().dry_run

    def start(self):
        '''
        Start all threads
        '''
        # wingpu: input goes to the VM; dry run unless explicitly turned off
        if is_wingpu():
            from src.input.WingpuKeyboard import get_keyboard
            wcfg = self.cfg.get("wingpu", {}) or {}
            live = bool(getattr(self.args, "wingpu_live", False)) or                    wcfg.get("dry_run", True) is False
            get_keyboard().configure(self.cfg, live=live)

        # Start keyboard controller thread
        self.kb = KeyBoardController(self.cfg)
        if self.is_disable_control:
            self.kb.disable() # Disable keyboard controller for debugging

        # Start game window capturing thread
        if is_wingpu():
            from src.input.GameWindowCapturorForWingpu import WingpuCapturor
            Capturor = WingpuCapturor
        else:
            Capturor = GameWindowCapturor
        if self.args.test_image == '':
            self.capture = Capturor(self.cfg)
        else:
            self.capture = Capturor(self.cfg, self.args.test_image)

        # Start health monitoring thread
        self.health_monitor = HealthMonitor(self.cfg, self.kb)
        if self.cfg["health_monitor"]["enable"] and \
            not self.is_disable_control:
            self.health_monitor.start()

        # Init profiler
        self.profiler = Profiler(self.cfg)

        # Init rune solver
        self.rune_solver = RuneSolver(self.cfg)

        # Nametag tracker (player_detect.mode nametag / auto)
        self.init_nametag()

        # Reset all timers
        self.t_last_frame = time.time()
        self.t_watch_dog = time.time()
        self.t_last_teleport = time.time()
        self.t_last_attack = time.time()
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()

        # Set init state
        if self.args.init_state != "":
            self.fsm.set_init_state(self.args.init_state) # For debugging
        elif self.cfg["bot"]["mode"] == "aux":
            self.fsm.set_init_state("aux")
        elif self.cfg["bot"]["mode"] == "patrol":
            self.fsm.set_init_state("patrol")
        else:
            self.fsm.set_init_state("hunting")

        # Start Auto Bot main thread
        self.thread_auto_bot = threading.Thread(target=self.loop)
        self.thread_auto_bot.start()
        self.is_first_frame = True

        logger.info("[MapleStoryAutoBot] Started")

    def pause(self):
        '''
        Terminate thread except main thread
        '''
        self.terminate_threads()

    def enable_viz(self):
        '''
        Enable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = True
        logger.debug("[enable_viz] is_show_debug_window = True")

    def disable_viz(self):
        '''
        Disable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = False
        logger.debug("[disable_viz] is_show_debug_window = False")

    def start_record(self):
        '''
        Start record
        '''
        # Prepare video writer if need to record
        if not self.is_show_debug_window:
            self.enable_viz()

        # Make sure video/ exist
        os.makedirs("video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("video", f"{timestamp}.mp4")

        # Get video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
        self.video_writer = cv2.VideoWriter(path, fourcc, 10, WINDOW_WORKING_SIZE)

        logger.info(f"[start_record] Record video to {path}")

    def stop_record(self):
        '''
        Stop Record
        '''
        self.video_writer = None
        logger.info("[stop_record] Stop recording")

    def get_player_location_by_nametag(self):
        '''
        Detects the player's location based on the nametag position in the game window.

        This function works by:
        - Extracting a vertical region of interest (ROI) where the nametag is expected.
        - Padding the ROI to avoid template matching edge issues.
        - Using template matching to locate the nametag, split into left and right halves
        to improve robustness against partial occlusion.
        - Selecting the best match (left or right) based on score and cache status.
        - Computing the player's center position by applying a fixed offset to the nametag.

        Returns:
            loc_player (tuple): The (x, y) coordinates of the player's estimated location.
        '''
        # Get camera region in the game window
        img_camera = self.img_frame_gray[
            :self.cfg["ui_coords"]["ui_y_start"], :]

        # Get nametag image and search image
        if self.cfg["nametag"]["mode"] == "white_mask":
            # Apply Gaussian blur for smoother white detection
            img_camera = cv2.GaussianBlur(img_camera, (3, 3), 0)
            img_nametag = cv2.GaussianBlur(self.img_nametag_gray, (3, 3), 0)
            lower_white, upper_white = (150, 255)
            img_roi = cv2.inRange(img_camera, lower_white, upper_white)
            img_nametag  = cv2.inRange(img_nametag, lower_white, upper_white)
        elif self.cfg["nametag"]["mode"] == "grayscale":
            img_roi = img_camera
            img_nametag = self.img_nametag_gray
        elif self.cfg["nametag"]["mode"] == "histogram_eq":
            # Apply histogram equalization
            img_nametag_eq = cv2.equalizeHist(self.img_nametag_gray)
            img_camera_eq = cv2.equalizeHist(img_camera)

            # Apply global (fixed) threshold
            _, img_nametag = cv2.threshold(img_nametag_eq, 150, 255, cv2.THRESH_BINARY)
            _, img_roi = cv2.threshold(img_camera_eq, 150, 255, cv2.THRESH_BINARY)
        else:
            logger.error(f"Unsupported nametag detection mode: {self.cfg['nametag']['mode']}")
            return
        # cv2.imshow("img_roi", img_roi)
        # cv2.imshow("img_nametag", img_nametag)

        # Pad search region to deal with fail detection when player is at map edge
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Get last frame name tag location
        if self.is_first_frame:
            last_result = None
        else:
            last_result = (
                self.loc_nametag[0] + pad_x,
                self.loc_nametag[1] + pad_y
            )

        # Get number of splits
        h, w = img_nametag.shape
        num_splits = max(1, w // self.cfg["nametag"]["split_width"])
        w_split = w // num_splits

        # Get nametag's background mask
        mask = get_mask(self.img_nametag, (0, 255, 0))

        # Vertically split the nametag image
        nametag_splits = {}
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else w
            nametag_splits[f"{i+1}/{num_splits}"] = {
                "img": img_nametag[:, x_s:x_e],
                "mask": mask[:, x_s:x_e],
                "last_result": (
                    (last_result[0] + x_s, last_result[1]) if last_result else None
                ),
                "score_penalty": 0.0,
                "offset_x": x_s
            }

        # Match tempalte
        matches = []
        for tag_type, split in nametag_splits.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi,
                split["img"],
                last_result=split["last_result"],
                mask=split["mask"],
                global_threshold=self.cfg["nametag"]["global_diff_thres"]
            )
            w_match = split["img"].shape[1]
            h_match = split["img"].shape[0]
            score += split["score_penalty"]
            matches.append((tag_type, loc, score, w_match, h_match, is_cached, split["offset_x"]))

        # Select best match and fix offset:
        matches.sort(key=lambda x: (not x[5], x[2]))  # prefer cached, then low score
        tag_type, loc_nametag, score, w_match, h_match, is_cached, offset_x = matches[0]

        # Adjust match location back to full nametag coordinates
        loc_nametag = (loc_nametag[0] - offset_x, loc_nametag[1])
        loc_nametag = (
            loc_nametag[0] - pad_x,
            loc_nametag[1] - pad_y
        )

        # Only update nametag location when score is good enough
        self.nametag_found = score < self.cfg["nametag"]["diff_thres"]
        if self.nametag_found:
            self.loc_nametag = loc_nametag

        loc_player = (
            self.loc_nametag[0] + w // 2,
            self.loc_nametag[1] - self.cfg["nametag"]["offset"][1]
        )

        # Draw name tag detection box for debugging
        draw_rectangle(self.img_frame_debug, self.loc_nametag,
                       self.img_nametag.shape, (0, 255, 0), "")
        text = f"NameTag,{round(score, 2)}," + \
                f"{'cached' if is_cached else 'missed'}," + \
                f"{tag_type}"
        cv2.putText(self.img_frame_debug, text,
                    (self.loc_nametag[0],
                     self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return loc_player

    def find_party_red_bars(self, x0, y0, x1, y1):
        '''
        Party red bar candidates in img_frame[y0:y1, x0:x1] (frame coordinates).
        Upstream geometry (5-7 px high, <= 50 px wide, area >= 10) times the scale.
        '''
        s = self.scale
        img = self.img_frame[y0:y1, x0:x1]
        if img.size == 0:
            return []
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = to_opencv_hsv(self.cfg["party_red_bar"]["lower_red"])
        upper_red = to_opencv_hsv(self.cfg["party_red_bar"]["upper_red"])
        mask_red = cv2.inRange(img_hsv, lower_red, upper_red)

        # Zero out the minimap area (its colours are not the party bar)
        mx, my = self.loc_minimap
        mh, mw = self.img_minimap.shape[:2]
        ax0, ay0 = max(mx - x0, 0), max(my - y0, 0)
        ax1, ay1 = min(mx + mw - x0, x1 - x0), min(my + mh - y0, y1 - y0)
        if ax1 > ax0 and ay1 > ay0:
            mask_red[ay0:ay1, ax0:ax1] = 0

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        h_min, h_max = int(5 * s), int(np.ceil(7 * s))
        w_max, area_min = int(np.ceil(50 * s)), 10 * s * s
        boxs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not (h_min <= h <= h_max and 1 <= w <= w_max):
                continue
            # fill = red pixels in the box (contourArea of a 5-6 px high bar
            # undercounts it: the bar with its light top row read 0.49, missed)
            area = cv2.countNonZero(mask_red[y:y+h, x:x+w])
            fill_rate = float(area) / (h*w)
            if area_min <= area and fill_rate >= 0.6 and self.has_dark_frame(img, x, y, w, h):
                boxs.append((x + x0, y + y0, w, h))
        return boxs

    @staticmethod
    def has_dark_frame(img, x, y, w, h):
        '''
        The party HP bar sits in a black frame: the rows just above and below
        the red part are dark (red flowers / damage numbers / potions are not)
        '''
        # (a light highlight row can sit between the red and the frame: look
        #  up to 3 rows away; measured on maple: frame 1-2 rows above/below)
        if y < 3 or y + h + 3 > img.shape[0] or w < 3:
            return False

        def dark_row(yy):
            return float(np.mean(img[yy, x:x + w].max(axis=1) < 90)) >= 0.8

        return any(dark_row(y - d) for d in (1, 2, 3)) and \
            any(dark_row(y + h - 1 + d) for d in (1, 2, 3))

    def get_player_location_by_party_red_bar(self):
        '''
        get_player_location_by_party_red_bar
        Searches near the last bar first (cheap), then the whole camera area.
        '''
        H, W = self.img_frame.shape[:2]
        ui_y = min(H, self.cfg["ui_coords"]["ui_y_start"])
        regions = []
        if time.time() - self.t_player_seen < 1.0:
            bx, by = self.loc_party_red_bar
            dx, dy = sc(200, self.scale), sc(120, self.scale)
            regions.append((max(0, bx - dx), max(0, by - dy), min(W, bx + dx), min(ui_y, by + dy)))
        regions.append((0, 0, W, ui_y))
        boxs = []
        for region in regions:
            boxs = self.find_party_red_bars(*region)
            if boxs:
                break

        if not boxs:
            return None, None  # red bar not found

        # Sort box by area
        boxs.sort(key=lambda box: box[2] * box[3], reverse=True)

        # Consider the biggest area as party red bar
        x, y, w, h = (int(v) for v in boxs[0])

        # Offset coordinate
        loc_party_red_bar = (x, y)
        loc_player = (x + self.cfg["party_red_bar"]["offset"][0],
                      y + self.cfg["party_red_bar"]["offset"][1])

        # visualize for debug
        draw_rectangle(self.img_frame_debug, loc_party_red_bar,
                    (h, w), (0, 255, 0), "party red bar", thickness=1, text_height=0.4)

        return loc_player, loc_party_red_bar

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap_route)
        self.minimap_global_score = score

        x_offset, y_offset = self.cfg["minimap"]["offset"]
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0] + x_offset,
            self.loc_minimap_global[1] + self.loc_player_minimap[1] + y_offset
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap_route.shape[1],
            self.loc_minimap_global[1] + self.img_minimap_route.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg["route"]["search_range"])
        x_max = min(w, x0 + self.cfg["route"]["search_range"])
        y_min = max(0, y0 - self.cfg["route"]["search_range"])
        y_max = min(h, y0 + self.cfg["route"]["search_range"])

        nearest = None
        nearest_up_down = None
        min_dist = float('inf')
        min_dist_up_down = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                dist = abs(x - x0) + abs(y - y0)
                # Get nearest color
                if pixel in self.color_code and dist < min_dist:
                    nearest = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code[pixel],
                        "distance": dist
                    }
                    min_dist = dist
                # Get nearest color (up, dowm)
                if pixel in self.color_code_up_down and dist < min_dist_up_down:
                    nearest_up_down = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code_up_down[pixel],
                        "distance": dist
                    }
                    min_dist_up_down = dist

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg["route"]["search_range"]*2,
             self.cfg["route"]["search_range"]*2),
            (0, 0, 255), "", text_height=0.4, thickness=1,
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            cv2.line(
                self.img_route_debug,
                self.loc_player_global, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )
            # Print color code on debug image
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest['command']}",
                (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.putText(
                self.img_frame_debug, f"Route Index: {self.idx_routes}",
                (650, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        if nearest_up_down is not None:
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest_up_down['command']}",
                (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.line(
                self.img_route_debug,
                self.loc_player_global,  # start point
                nearest_up_down["pixel"],# end point
                (0, 0, 255),             # green line
                1                        # thickness
            )

        return nearest, nearest_up_down  # if not found return none

    def get_attack_range(self, is_left=True):
        '''
        get_attack_range
        '''
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        elif self.cfg["bot"]["attack"] == "directional":
            if is_left:
                x0 = self.loc_player[0] - self.cfg["directional_attack"]["range_x"]
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + self.cfg["directional_attack"]["range_x"]
            y0 = self.loc_player[1] - self.cfg["directional_attack"]["range_y"] // 2
            y1 = y0 + self.cfg["directional_attack"]["range_y"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")

        return (x0, y0, x1, y1)

    def get_nearest_monster(self, is_left=True):
        '''
        Finds the nearest monster within the player's attack range.

        This function:
        - Defines an attack box relative to the player position,
            depending on the facing direction (`is_left`).
        - Iterates through all detected monsters and checks which ones overlap
          with the attack box.
        - Returns the closest valid monster that meets the overlap criteria.

        Args:
            is_left (bool): If True, assume the player is facing left;
                            adjusts attack box accordingly.
        Returns:
            dict or None: The nearest monster's info dict, or None if no valid match.
        '''

        x0, y0, x1, y1 = self.get_attack_range(is_left=is_left)

        nearest_monster = None
        min_distance = float('inf')
        for monster in self.monsters:
            mx1, my1 = monster["position"]
            mw, mh = monster["size"]
            mx2 = mx1 + mw
            my2 = my1 + mh

            # Calculate intersection
            ix1 = max(x0, mx1)
            iy1 = max(y0, my1)
            ix2 = min(x1, mx2)
            iy2 = min(y1, my2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih

            inter_area_thres = min(self.monster_min_area, self.cfg['monster_detect']['max_mob_area_trigger'])
            if inter_area >= inter_area_thres:
                # Compute distance to player center
                monster_center = (mx1 + mw // 2, my1 + mh // 2)
                dx = monster_center[0] - self.loc_player[0]
                dy = monster_center[1] - self.loc_player[1]
                distance = abs(dx) + abs(dy)  # Manhattan distance

                if distance < min_distance:
                    min_distance = distance
                    nearest_monster = monster

        return nearest_monster

    def get_monsters_in_range(self, top_left, bottom_right):
        '''
        get_monsters_in_range
        '''
        x0, y0 = top_left
        x1, y1 = bottom_right

        img_roi = self.img_frame[y0:y1, x0:x1]

        # Shift player's location into ROI coordinate system
        px, py = self.loc_player
        px_in_roi = px - x0
        py_in_roi = py - y0

        # Define rectangle range around player (in ROI coordinate)
        char_x_min = max(0, px_in_roi - self.cfg["character"]["width"] // 2)
        char_x_max = min(img_roi.shape[1], px_in_roi + self.cfg["character"]["width"] // 2)
        char_y_min = max(0, py_in_roi - self.cfg["character"]["height"] // 2)
        char_y_max = min(img_roi.shape[0], py_in_roi + self.cfg["character"]["height"] // 2)

        s = self.scale
        mode = self.cfg["monster_detect"]["mode"]
        img_roi_blur = None
        img_roi_gray = None
        if self.cfg["bot"]["mode"] != "patrol" and mode == "contour_only":
            # Black outline pixels of the ROI, once for all templates
            # (black_thres > 0: sprites the game draws scaled are filtered, their
            #  outline is near-black, not exactly 0,0,0)
            bt = int(self.cfg["monster_detect"].get("black_thres", 0))
            if bt > 0:
                mask_roi = cv2.inRange(img_roi, (0, 0, 0), (bt, bt, bt))
            else:
                mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
            # Zero out mask inside this region (ignore player's own character)
            mask_roi[char_y_min:char_y_max, char_x_min:char_x_max] = 0
            blur = self.cfg["monster_detect"]["contour_blur"]
            img_roi_blur = cv2.GaussianBlur(mask_roi, (blur, blur), 0)
            ds = self.monster_match_downscale()
            if ds > 1:
                img_roi_blur = cv2.resize(img_roi_blur, (max(1, img_roi_blur.shape[1] // ds),
                                                         max(1, img_roi_blur.shape[0] // ds)),
                                          interpolation=cv2.INTER_AREA)
        elif self.cfg["bot"]["mode"] != "patrol" and mode == "grayscale":
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_roi_ds = img_roi
        img_roi_pre = None
        ds = self.monster_match_downscale()
        if mode == "color" and ds > 1:
            img_roi_ds = cv2.resize(img_roi, (max(1, img_roi.shape[1] // ds), max(1, img_roi.shape[0] // ds)),
                                    interpolation=cv2.INTER_AREA)
        if mode == "color" and self.monster_color_pre:
            img_roi_pre = cv2.resize(img_roi_ds, (max(1, img_roi_ds.shape[1] // 2),
                                                  max(1, img_roi_ds.shape[0] // 2)),
                                     interpolation=cv2.INTER_AREA)

        monsters = []
        for monster_name, monster_imgs in self.monsters_info.items():
            for i_tpl, (img_monster, mask_monster) in enumerate(monster_imgs):
                if self.cfg["bot"]["mode"] == "patrol":
                    pass # Don't detect monster using template in patrol mode
                elif self.cfg["monster_detect"]["mode"] == "template_free":
                    # Generate mask where pixel is exactly (0,0,0)
                    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    # cv2.imshow("Black Pixel Mask", black_mask)

                    # Zero out mask inside this region (ignore player's own character)
                    black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sc(20, s), sc(20, s)))
                    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Black Mask", closed_mask)

                    # draw player character bounding box
                    draw_rectangle(
                        self.img_frame_debug, (char_x_min+x0, char_y_min+y0),
                        (self.cfg["character"]["height"], self.cfg["character"]["width"]),
                        (255, 0, 0), "Character Box"
                    )

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

                    monsters = []
                    min_area = 1000 * s * s
                    for i in range(1, num_labels):
                        x, y, w, h, area = stats[i]
                        if area > min_area:
                            monsters.append({
                                "name": "",
                                "position": (x0+x, y0+y),
                                "size": (h, w),
                                "score": 1.0,
                            })
                elif self.cfg["monster_detect"]["mode"] == "contour_only":
                    # Use only black lines contour to detect monsters
                    # (template masks are blurred once in build_templates)
                    img_monster_blur = self.monster_contour[(monster_name, i_tpl)]

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        continue  # template bigger than roi, skip this template

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])

                    h, w = img_monster.shape[:2]
                    ds = self.monster_match_downscale()
                    for pt in zip(*match_locations[::-1]):
                        monsters.append({
                            "name": monster_name,
                            "position": (int(pt[0]) * ds + x0, int(pt[1]) * ds + y0),
                            "size": (h, w),
                            "score": float(res[pt[1], pt[0]]),
                        })
                elif self.cfg["monster_detect"]["mode"] == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    if img_monster_gray.shape[0] > img_roi_gray.shape[0] or \
                       img_monster_gray.shape[1] > img_roi_gray.shape[1]:
                        continue
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monsters.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                elif self.cfg["monster_detect"]["mode"] == "color":
                    # masked colour matching (at 1/ds: monster_detect.match_downscale)
                    tpl_c, mask_c = self.monster_color[(monster_name, i_tpl)]
                    if tpl_c.shape[0] > img_roi_ds.shape[0] or \
                       tpl_c.shape[1] > img_roi_ds.shape[1]:
                        continue
                    matches = self.color_matches(img_roi_ds, img_roi_pre, monster_name, i_tpl)
                    h, w = img_monster.shape[:2]
                    th, tw = tpl_c.shape[:2]
                    min_std = float(self.cfg["monster_detect"].get("color_min_std", 0))
                    for (px, py, score) in matches:
                        pt = (px, py)
                        # flat, blurred background (sky, far mountains) scores
                        # like a monster with SQDIFF_NORMED; monsters have texture
                        if min_std > 0:
                            patch = img_roi_ds[pt[1]:pt[1] + th, pt[0]:pt[0] + tw]
                            _, std = cv2.meanStdDev(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), mask=mask_c)
                            if float(std[0][0]) < min_std:
                                continue
                        monsters.append({
                            "name": monster_name,
                            "position": (int(pt[0]) * ds + x0, int(pt[1]) * ds + y0),
                            "size": (h, w),
                            "score": score,
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg['monster_detect']['mode']}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        # (SQDIFF scores: keep the lowest of overlapping boxes; upstream kept the worst)
        monsters = nms(monsters, iou_threshold=0.4,
                       lower_is_better=self.cfg["monster_detect"]["mode"] != "template_free")

        # Detect monster via health bar
        if self.cfg["monster_detect"]["with_enemy_hp_bar"]:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]),
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]))

            # Find connected components (each cluster of green pixels)
            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):  # skip background (label 0)
                x, y, w, h, area = stats[i]
                if area < 3:  # small noise filter
                    continue

                # Guess a monster bounding box
                y += sc(10, s)
                x = max(0, x)
                y = max(0, y)
                w = sc(70, s)
                h = self.monster_min_h

                monsters.append({
                    "name": "Health Bar",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        # Debug
        # Draw attack detection range
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Mob Detection Box"
        )

        # Draw monsters bounding box
        for monster in monsters:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            draw_rectangle(
                self.img_frame_debug, monster["position"], monster["size"],
                color, str(round(monster['score'], 2))
            )

        return monsters

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # wingpu: the capturor gives the game client 1:1 (no title bar, no
        # resize); detectors are scaled to it (on_resolution)
        if is_wingpu():
            if self.frame.shape[:2] != self.frame_hw:
                self.on_resolution(*self.frame.shape[:2])
            return self.frame

        # Cut the title bar and resize raw frame to (1296, 759)
        frame_no_title = self.frame[self.cfg["game_window"]["title_bar_height"]:, :]

        # Make sure the window ratio is as expected
        if self.args.test_image != "":
            pass # Disable size check if using test image for debugging
        elif self.cfg["bot"]["mode"] == "aux":
            if not is_img_16_to_9(frame_no_title, self.cfg): # Aux mode allow 16:9 resolution
                text = f"Unexpeted window size: {frame_no_title.shape[:2]} (expect window ratio 16:9)\n"
                text += "Please use windowed mode & smallest resolution."
                logger.error(text)
                return
        else:
            # Other mode only allow specific resolution
            if self.cfg["game_window"]["size"] != frame_no_title.shape[:2]:
                text = f"Unexpeted window size: {frame_no_title.shape[:2]} "\
                       f"(expect {self.cfg['game_window']['size']})\n"
                text += "Please use windowed mode & smallest resolution."
                logger.error(text)
                return

        return cv2.resize(frame_no_title, WINDOW_WORKING_SIZE,
                   interpolation=cv2.INTER_NEAREST)

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving)
        based on their global position on map.

        This function:
        - Compares the player's current position with their last known position
          tracked by the watchdog.
        - If the player has moved beyond a threshold (`watch_dog_range`),
          it resets the watchdog timer.
        - If the player hasn't moved and the elapsed time exceeds (`watch_dog_timeout`),
          it flags the player as stuck and resets the watchdog.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg["watchdog"]["range"]:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {round(dt, 2)} seconds.")
            return True
        return False

    def screenshot_img_frame(self):
        '''
        Save self.img_frame
        '''
        if self.img_frame is None:
            logger.error("[screenshot_img_frame] Failed, game window is not available")
        else:
            screenshot(self.img_frame, "img_frame")

        if self.img_frame_debug is None:
            pass
        else:
            screenshot(self.img_frame_debug, "img_frame_debug")

        if self.frame is None:
            pass
        else:
            screenshot(self.frame, "frame")

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 460
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        h, w = self.frame.shape[:2]
        text_list = [
            f"FPS: {self.fps}",
            f"State: {self.fsm.state.name}",
            f"Resolution: {h}x{w}, Ratio: {round(w/h, 2)}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
             "Press 'F12' to quit"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw attack box on debug window
        if self.cfg["bot"]["attack"] == "aoe_skill":
            x0, y0, x1, y1 = self.get_attack_range()
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range"
            )
        elif self.cfg["bot"]["attack"] == "directional":
            x0, y0, x1, y1 = self.get_attack_range(is_left=True)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Left)"
            )
            x0, y0, x1, y1 = self.get_attack_range(is_left=False)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Right)"
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Don't draw minimap in patrol mode
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 10
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

        # Draw HP/MP/EXP bar on debug window
        percent_bars = [self.health_monitor.hp_percent,
                      self.health_monitor.mp_percent,
                      self.health_monitor.exp_percent]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            x_s, y_s = (250, 30)
            # Print bar ratio on debug window
            cv2.putText(self.img_frame_debug,
                        f"{bar_name}: {percent_bars[i]:.1f}%",
                        (x_s, y_s + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw bar on debug window
            x_s, y_s = (410, 13)
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s+30*i:y_s+h+30*i, x_s:x_s+w] = \
                self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :][y:y+h, x:x+w]

        # Print command on screen
        cv2.putText(self.img_frame_debug, f"Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
            self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def ensure_is_in_party(self):
        '''
        ensure_is_in_party
        '''
        # open party window
        press_key(self.cfg["key"]["party"])

        # Wait party window to show up
        time.sleep(0.5)

        # Update image frame
        self.img_frame = self.get_img_frame()

        # Find the 'create party' button
        loc_enable, score_enable, _ = find_pattern_sqdiff(
                        self.img_frame, self.img_create_party_enable)

        lang = self.cfg["system"]["language"]
        thres = self.cfg['party_red_bar'][f'create_party_button_{lang}_thres']
        if score_enable < thres:
            logger.info(f"[ensure_is_in_party] Find party enable button({round(score_enable, 2)})")
            h, w = self.img_create_party_enable.shape[:2]
            click_in_game_window(self.capture.window_title,
                (loc_enable[0] + w // 2,
                 loc_enable[1] + h // 2 + self.title_bar_height())
            )
        else:
            logger.info("[ensure_is_in_party] Cannot find create party button."
                        "Maybe player already in party.")

        # close party window
        press_key(self.cfg["key"]["party"])
    # [新增] 等待 login 按鈕的共用函式（含 timeout 與例外復原）
    def wait_for_login_button(self, window_title, timeout=60):
        start = time.time()
        loc_login_button = None
        if not is_mac():
            resize_window(window_title, width=1296, height=759)
        while (time.time() - start) < timeout:
            try:
                self.img_frame = self.get_img_frame()
                loc_login_button = self.get_login_button_location()
                if loc_login_button is not None:
                    return loc_login_button
                logger.info("Waiting for login button to show up...")
            except Exception as e:
                logger.warning(f"Exception while waiting for login button: {e}")
                if not is_mac():
                    resize_window(window_title, width=1296, height=759)
                logger.info("Retrying login button detection...")
            time.sleep(3)
        return None
    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        window_title = self.capture.window_title
        ui_coords = self.cfg["ui_coords"]
        click_in_game_window(window_title, ui_coords["menu"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel_confirm"])
        time.sleep(1)

        loc_login_button = None
        # [替換] 單一等待＋逾時保護，找不到就中止，不再往下點擊
        loc_login_button = self.wait_for_login_button(window_title, timeout=60)
        logger.info(f"login_button button found: {loc_login_button}")

        if loc_login_button is None:
            logger.error("Login button not found after timeout. Abort channel_change.")
            return
        time.sleep(2)
        # Click login button
        click_in_game_window(window_title, loc_login_button)
        time.sleep(2)

        # Click "Select Character"
        click_in_game_window(window_title, ui_coords["select_character"])
        time.sleep(5)

        self.kb.enable()
        self.kb.set_command("none none none")
        self.kb.release_all_key()

        self.ensure_is_in_party() # Make sure player is in party

        self.fsm.set_init_state("hunting")
        self.t_last_attack = time.time() # Update timer

    def terminate_threads(self):
        '''
        terminate all threads
        '''
        # Terminate keyboard controller
        if self.kb is not None:
            self.kb.is_terminated = True
        # Terminate game window capturor
        if self.capture is not None:
            self.capture.stop()
        # Terminate health monitor
        if self.health_monitor is not None:
            self.health_monitor.stop()
        self.is_terminated = True
        logger.info(f"[terminate_threads] Terminated all threads")

    def get_attack_direction(self, monster_left, monster_right):
        '''
        get_attack_direction
        '''
        # Compute distance for left
        distance_left = float('inf')
        if monster_left is not None:
            mx, my = monster_left["position"]
            mw, mh = monster_left["size"]
            center_left = (mx + mw // 2, my + mh // 2)
            distance_left = abs(center_left[0] - self.loc_player[0]) + \
                            abs(center_left[1] - self.loc_player[1])
        # Compute distance for right
        distance_right = float('inf')
        if monster_right is not None:
            mx, my = monster_right["position"]
            mw, mh = monster_right["size"]
            center_right = (mx + mw // 2, my + mh // 2)
            distance_right = abs(center_right[0] - self.loc_player[0]) + \
                            abs(center_right[1] - self.loc_player[1])
        # Choose attack direction and nearest monster
        attack_direction = None
        # nearest_monster = None

        # Additional validation: check if monster is actually on the correct side
        def is_monster_on_correct_side(monster, direction):
            if monster is None:
                return False
            mx, my = monster["position"]
            mw, mh = monster["size"]
            monster_center_x = mx + mw // 2
            player_x = self.loc_player[0]

            if direction == "left":
                return monster_center_x < player_x  # Monster should be left of player
            else:  # direction == "right"
                return monster_center_x > player_x  # Monster should be right of player

        # Only choose direction if there's a clear winner and monster is on correct side
        if monster_left is not None and monster_right is None and \
            is_monster_on_correct_side(monster_left, "left"):
            attack_direction = "left"
            # nearest_monster = monster_left
        elif monster_right is not None and monster_left is None and \
            is_monster_on_correct_side(monster_right, "right"):
            attack_direction = "right"
            # nearest_monster = monster_right
        elif monster_left is not None and monster_right is not None:
            # Both sides have monsters, check distance and side validation
            left_valid = is_monster_on_correct_side(monster_left, "left")
            right_valid = is_monster_on_correct_side(monster_right, "right")

            if left_valid and not right_valid:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif right_valid and not left_valid:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid and distance_left < distance_right - sc(50, self.scale):
                attack_direction = "left"
                # nearest_monster = monster_left
            elif left_valid and right_valid and distance_right < distance_left - sc(50, self.scale):
                attack_direction = "right"
                # nearest_monster = monster_right
            # If both valid but distances too close, don't attack to avoid confusion

        # Debug attack direction selection
        if monster_left is not None or monster_right is not None:
            left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
            right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
            debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
            cv2.putText(self.img_frame_debug, debug_text,
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return attack_direction

    def is_need_change_channel(self, loc_other_players):
        '''
        is_need_change_channel
        '''
        # Calculate center value
        xs = [x for (x, _) in loc_other_players]
        ys = [y for (_, y) in loc_other_players]
        if len(xs) == 0 or len(ys) == 0:
            return False
        center_x, center_y = (np.mean(xs), np.mean(ys))
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        center = (int(np.mean(xs)), int(np.mean(ys)))
        #logger.info(f"[is_need_change_channel] Center of mass = {center}")

        # Change channel
        mode = self.cfg["channel_change"]["mode"]
        if mode == "true":
            logger.warning("[is_need_change_channel] Player detected, immediately change channel.")
            return True
        elif mode == "pixel":
            if self.red_dot_center_prev is None:
                self.red_dot_center_prev = center
            else:
                dx = abs(center[0] - self.red_dot_center_prev[0])
                dy = abs(center[1] - self.red_dot_center_prev[1])
                total = dx + dy
                logger.debug(f"[is_need_change_channel] Movement dx={dx}, dy={dy}, total={total}")
                thres = self.cfg["channel_change"]["other_player_move_thres"]
                if total > thres:
                    logger.warning(f"Other player movement > {thres} pixel detected. "
                                "Trigger channel change.")
                    return True
        else:
            logger.error(f"[is_need_change_channel] Unsupported mode: {mode}")

        return False

    def is_time_to_change_channel(self):
        '''
        is_time_to_change_channel
        '''
        if not self.cfg["scheduled_channel_switching"]["enable"]:
            return False
        dt = time.time() - self.t_to_change_channel
        if dt > self.cfg["scheduled_channel_switching"]["interval_seconds"]:
            self.t_to_change_channel = time.time()
            return True
        return False

    def get_login_button_location(self):
        '''
        get_login_button_location
        '''
        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Draw rectange on debug image
        draw_rectangle(self.img_frame_debug, (x0, y0),
                       (y1-y0, x1-x0), (0, 255, 0), "login_button box")

        # Find the 'login' button
        loc, score, _ = find_pattern_sqdiff(
                        img_roi, self.img_login_button)
        if score < self.cfg["ui_coords"]["login_button_thres"]:
            h, w = self.img_login_button.shape[:2]
            logger.info(f"[get_login_button_location] Found login button with score({score})")
            return (x0 + loc[0] + w // 2,
                    y0 + loc[1] + h // 2 + self.title_bar_height())
        else:
            return None

    def update_cmd_by_route(self):
        # get color code from img_route
        color_code, color_code_up_down = self.get_nearest_color_code()
        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code_up_down:
            if color_code["distance"] < color_code_up_down["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
                _, cmd, _ = color_code_up_down["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd # only complement cmd_move_y when player is on ladder
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
                cmd, _, _ = color_code["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd # only complement cmd_move_x when player is on ladder
        elif color_code:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
        elif color_code_up_down:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()

        # teleport away from edge to avoid falling off cliff
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # Use teleport while walking
        if self.cfg['teleport']['is_use_teleport_to_walk'] and \
            time.time() - self.t_last_teleport > self.cfg['teleport']['cooldown']:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # replace teleport to jump if user doesn't set teleport key
        if self.cfg["key"]["teleport"] == "" and self.cmd_action == "teleport":
            self.cmd_action = "jump"

    def update_cmd_by_mob_detection(self):
        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
            cooldown = self.cfg["aoe_skill"]["cooldown"]
        elif self.cfg["bot"]["attack"] == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
            cooldown = self.cfg["directional_attack"]["cooldown"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")
        # (clipped to the camera area: the UI band below ui_y_start is not the world)
        md = self.cfg["monster_detect"]
        search = md.get("search", "attack")
        y_max = min(self.img_frame.shape[0], self.cfg["ui_coords"]["ui_y_start"])
        if search == "full":
            # whole camera area
            x0, y0, x1, y1 = 0, 0, self.img_frame.shape[1], y_max
        else:
            if search == "roi":
                # box around the player: monster_detect.roi_left/right/up/down (scaled)
                dl, dr = md.get("roi_left", dx), md.get("roi_right", dx)
                du, dd = md.get("roi_up", dy), md.get("roi_down", dy)
            else:
                # upstream: attack range + search_box_margin
                dl = dr = dx
                du = dd = dy
            x0 = max(0                      , self.loc_player[0] - dl)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dr)
            y0 = max(0                      , self.loc_player[1] - du)
            y1 = min(y_max                  , self.loc_player[1] + dd)
        self.mob_search_box = (int(x0), int(y0), int(x1), int(y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            self.monsters = []
            return

        # Get monsters in the search box (every interval_frames frames; in
        # between the last result is used: halves the bot's biggest CPU cost)
        every = max(1, int(self.cfg["monster_detect"].get("interval_frames", 1)))
        self.n_mob_frames = getattr(self, "n_mob_frames", 0) + 1
        if every == 1 or self.n_mob_frames % every == 1 or not hasattr(self, "_mob_box"):
            self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))
            self._mob_box = (x0, y0, x1, y1)
            if self.monsters:
                self.det_counts["monsters"] += 1

        # Check if no mob to attack
        if len(self.monsters) == 0:
            return

        # Update attack command
        if self.cfg["bot"]["attack"] == "aoe_skill":
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()

        elif self.cfg["bot"]["attack"] == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            # Attack Command
            if time.time() - self.t_last_attack > cooldown and attack_direction is not None:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()
                # Set up attack direction
                self.cmd_move_x = attack_direction

    def update_cmd_by_random(self):
        '''
        update_cmd_by_random - pick a random action except 'up' and teleport command
        '''
        self.cmd_move_x = random.choice(["left", "right", "none"])
        self.cmd_move_y = random.choice(["down", "none"])
        self.cmd_action = random.choice(["jump", "none"])
        logger.warning("[update_cmd_by_random]"\
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}")

    def check_reach_goal(self):
        if self.cmd_action == "goal":
            # Switch to next route map
            self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
            logger.debug(f"Change to new route:{self.idx_routes}")

    def find_minimap(self):
        '''
        Minimap frame (x, y, w, h) in img_frame, or None. Looks around the last
        position first (the minimap rarely moves), then the whole frame.
        '''
        min_size = sc(100, self.scale)
        if self.minimap_rect is not None:
            x, y, w, h = self.minimap_rect
            pad = 6
            H, W = self.img_frame.shape[:2]
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            r = get_minimap_loc_size(self.img_frame[y0:y1, x0:x1], min_size)
            if r is not None:
                r = (int(r[0]) + x0, int(r[1]) + y0, int(r[2]), int(r[3]))
                self.minimap_rect = r
                return r
        r = get_minimap_loc_size(self.img_frame, min_size)
        if r is not None:
            r = tuple(int(v) for v in r)
        self.minimap_rect = r
        return r

    def check_detection_lost(self):
        '''
        True while the bot must hold still because the minimap (normal mode) or
        the player (normal/patrol) has not been found for lost_timeout seconds.
        Entering the pause releases every key; the keyboard controller and the
        health monitor send nothing while kb.is_hold is set.
        '''
        lcfg = self.cfg.get("detection_lost", {}) or {}
        if not lcfg.get("enable", True):
            return False
        timeout = float(lcfg.get("timeout", 3.0))
        mode = self.cfg["bot"]["mode"]
        now = time.time()
        reason = []
        if mode == "normal" and now - self.t_minimap_ok > timeout:
            reason.append("minimap")
        if mode in ("normal", "patrol") and now - self.t_player_seen > timeout:
            reason.append("player")
        if reason:
            self.det_counts["lost_paused"] += 1
            if not self.is_lost_paused or reason != self.lost_reason:
                logger.warning(f"[detection_lost] {' and '.join(reason)} not found for "
                               f">{timeout}s: holding all keys, sending nothing")
            if not self.is_lost_paused:
                self.is_lost_paused = True
                self.kb.set_command("none none none")
                self.kb.hold(True)
                self.kb.release_all_key()
                if is_wingpu():
                    from src.input.WingpuKeyboard import get_keyboard
                    get_keyboard().release_all()
            self.lost_reason = reason
            self.cmd_move_x = self.cmd_move_y = self.cmd_action = "none"
            self.monsters = []
            return True
        if self.is_lost_paused:
            logger.info("[detection_lost] minimap and player found again: resuming")
            self.is_lost_paused = False
            self.lost_reason = []
            self.kb.hold(False)
            # don't let the watchdogs fire on time spent paused
            self.t_watch_dog = self.t_last_attack = now
            self.loc_watch_dog = self.loc_player_global
        return False

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Start profiler for performance debugging
        self.profiler.start()

        # Check if need viz window
        self.is_show_debug_window = self.is_need_show_debug_window
        if not self.is_show_debug_window:
            self.img_frame_debug = None
            self.img_route_debug = None

        ###########################
        ### Image Preprocessing ###
        ###########################
        # Get game window frame
        img_frame = self.get_img_frame()
        if img_frame is None:
            if not is_mac():
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame
        self.det_counts["frames"] += 1

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug viz
        if self.is_show_debug_window:
            self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route = self.img_routes[self.idx_routes]
            if self.is_show_debug_window:
                self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        self.profiler.mark("Image Preprocessing")

        ###################
        ### Get Minimap ###
        ###################
        # Get minimap coordinate and size on game window
        minimap_result = self.find_minimap()
        if minimap_result is None:
            if time.time() - self.t_last_minimap_update > 30 and \
                    (not is_wingpu() or self.cfg["channel_change"]["enable"]):
                # Unable to get minimap for 30 seconds -> assume it's login screen
                # (wingpu: only with channel_change enabled, it clicks blindly otherwise)
                loc_login_button = self.get_login_button_location()
                if loc_login_button:
                    logger.info("Found login button on screen. Proceed to login.")
                    click_in_game_window(self.capture.window_title,
                                         loc_login_button)
                    time.sleep(3)
                    click_in_game_window(self.capture.window_title,
                                         self.cfg["ui_coords"]["select_character"])
                    time.sleep(2)
        else:
            x, y, w, h = minimap_result
            # Shrink minimap boardary by one pixel to avoid pixel leaking to minimap
            x += 1
            y += 1
            w -= 2
            h -= 2
            # update minimap image
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
            self.t_last_minimap_update = time.time()
            self.det_counts["minimap"] += 1
            # Minimap scale vs the route maps (once per resolution, retried every 2 s)
            if self.minimap_scale is not None and self.minimap_scale.m is None and \
                    time.time() - self.t_minimap_scale_try > 2.0:
                self.t_minimap_scale_try = time.time()
                m, score = self.minimap_scale.measure(self.img_minimap, guess=self.scale)
                if self.minimap_scale.m is not None:
                    logger.info(f"[minimap] minimap scale vs map.png = {m:.4f} (score {score:.3f}),"
                                f" ui scale {self.scale:.4f}")
                else:
                    logger.warning(f"[minimap] minimap does not match minimaps/"
                                   f"{self.cfg['bot']['map']}/map.png (best {m}, score {score}); "
                                   f"wrong map selected? using the ui scale {self.scale:.4f} meanwhile")
            if self.minimap_scale is not None:
                self.img_minimap_route = self.minimap_scale.to_route(self.img_minimap, self.scale)
            else:
                self.img_minimap_route = self.img_minimap

        self.profiler.mark("Get Minimap Location and Size")

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :])

        #################################
        ### Player Location Detection ###
        #################################
        # Get player location in game window
        player_found = False
        self.player_source = None
        if self.cfg["nametag"]["enable"]:
            # upstream nametag code (nametag/<name>.png at the reference scale)
            loc_player = self.get_player_location_by_nametag()
            player_found = getattr(self, "nametag_found", False)
            self.player_source = "nametag" if player_found else None
        else:
            mode = self.player_detect_mode()
            loc_player = None
            if mode in ("party_bar", "auto"):
                loc_player, loc_party_red_bar = self.get_player_location_by_party_red_bar()
                if loc_party_red_bar is not None:
                    self.loc_party_red_bar = loc_party_red_bar
                    self.player_source = "party_bar"
            # template changed on disk (web UI) -> reload, checked every 2 s
            if self.nametag is not None and time.time() - self.t_nametag_check > 2.0:
                self.t_nametag_check = time.time()
                self.nametag.reload_if_changed()
            if self.player_source is None and mode in ("nametag", "auto") and self.nametag is not None:
                ui_y = min(self.img_frame.shape[0], self.cfg["ui_coords"]["ui_y_start"])
                p = self.nametag.find(self.img_frame[:ui_y])
                if p is not None:
                    loc_player = p
                    self.player_source = "nametag"
                    if self.is_show_debug_window and self.nametag.tpl is not None:
                        th, tw = self.nametag.tpl.shape[:2]
                        draw_rectangle(self.img_frame_debug, self.nametag.loc, (th, tw), (255, 0, 255),
                                       f"nametag {self.nametag.last_score:.2f} {self.nametag.last_mode}",
                                       thickness=1, text_height=0.4)
            player_found = self.player_source is not None
        if player_found:
            self.t_player_seen = time.time()
            self.det_counts["player"] += 1
            if self.player_source == "party_bar":
                self.det_counts["player_party_bar"] += 1
            elif self.player_source == "nametag":
                self.det_counts["player_nametag"] += 1

        # Update player location
        if loc_player is not None:
            # Check if character is on ladder
            dx = abs(loc_player[0] - self.loc_player[0])
            dy = abs(loc_player[1] - self.loc_player[1])
            ladder_dx = max(3, sc(3, self.scale))
            if self.is_on_ladder:
                if dx > ladder_dx: # Leave ladder if there is horizontal move
                    self.is_on_ladder = False
            else:
                if dx < ladder_dx and dy != 0:
                    self.is_on_ladder = True
            # Update player location
            self.loc_player = loc_player

        # Draw player center for debugging
        cv2.circle(self.img_frame_debug,
                self.loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        # Get player location on minimap (exact colour on the live minimap,
        # then converted to route-map pixels)
        loc_player_minimap = None
        if minimap_result is not None:
            loc_player_minimap = get_player_location_on_minimap(
                                    self.img_minimap,
                                    minimap_player_color=self.cfg["minimap"]["player_color"])
        if self.minimap_scale is None:
            mm = 1.0
        else:
            mm = self.minimap_scale.m or self.scale
        if loc_player_minimap:
            self.loc_player_minimap = (int(round(loc_player_minimap[0] / mm)),
                                       int(round(loc_player_minimap[1] / mm)))
            self.det_counts["player_dot"] += 1
            self.t_minimap_ok = time.time()

        # Get other player location on minimap
        loc_other_players = get_all_other_player_locations_on_minimap(
                                self.img_minimap,
                                self.cfg['minimap']['other_player_color'])
        loc_other_players = [(x / mm, y / mm) for (x, y) in loc_other_players]

        # Get player location on global map
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            self.loc_player_global = self.loc_player_minimap
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        self.profiler.mark("Player Location Detection")

        ##########################
        ### Detection-lost pause ###
        ##########################
        # Minimap or player not found for lost_timeout seconds: hold every key,
        # send nothing, resume when both are back (no blind watchdog actions).
        if self.check_detection_lost():
            self.is_first_frame = False
            if self.is_show_debug_window:
                cv2.putText(self.img_frame_debug,
                            f"PAUSED: {', '.join(self.lost_reason)} not found",
                            (10, sc(400, self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.update_info_on_img_frame_debug()
                self.img_frame_debug_done = self.img_frame_debug
            self.t_last_frame = time.time()
            return 0

        ######################
        ### Change Channel ###
        ######################
        if self.cfg['channel_change']['enable'] and \
            self.is_need_change_channel(loc_other_players):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if self.is_time_to_change_channel() and \
            (not is_wingpu() or (self.cfg["channel_change"]["enable"] and
                                 not self.is_wingpu_dry_run())):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            return 0

        self.profiler.mark("Change Channel")

        #######################
        ### Attack WatchDog ###
        ####################### Check if last attack is timeout
        dt = time.time() - self.t_last_attack
        if self.cfg['bot']['mode'] == 'normal' and \
            dt > self.cfg["watchdog"]["last_attack_timeout"]:
            logger.info(f"[Attack Timeout] Last attack timeout for {round(dt, 2)} seconds")
            cfg_action = self.cfg["watchdog"]["last_attack_timeout_action"]
            # Re-arm first: otherwise a failed action repeats on every frame
            self.t_last_attack = time.time()
            if is_wingpu() and self.is_wingpu_dry_run():
                # wingpu dry run: no attack is ever sent, so the timeout says nothing
                # about the game; acting on it would block the loop (channel_change
                # waits up to 60 s for a login button) or stop the bot (go_home)
                logger.info(f"[Attack Timeout] dry run: skip '{cfg_action}'")
            elif is_wingpu() and cfg_action == "change_channel" and \
                not self.cfg["channel_change"]["enable"]:
                logger.info("[Attack Timeout] channel_change disabled on wingpu: skip")
            elif cfg_action == "change_channel":
                logger.info("[Attack Timeout] Change channel!")
                self.channel_change()
            elif cfg_action == "go_home":
                logger.info("[Attack Timeout] Return home!")
                press_key(self.cfg["key"]["return_home"])
                # Terminate Autobot
                self.is_terminated = True
                self.kb.is_terminated = True
            else:
                logger.info(f"Unsupported timeout mode: {cfg_action}")

        self.profiler.mark("Attack WatchDog")

        ######################
        ### State Behavior ###
        ######################
        self.fsm.do_state_stuff()

        self.is_first_frame = False

        self.profiler.mark("State per-frame behavior")

        #####################
        ### Debug Windows ###
        #####################
        # Don't show debug window to save system resource
        if not self.is_show_debug_window:
            return 0 # frame done

        # Print text on debug image
        self.update_info_on_img_frame_debug()
        # finished debug image (img_frame_debug is a fresh, unannotated copy
        # while the next frame is being processed)
        self.img_frame_debug_done = self.img_frame_debug

        # Save debug window to video
        if self.video_writer:
            self.video_writer.write(self.img_frame_debug)

        # Resize img_route_debug for better visualization
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route_debug = cv2.resize(
                        self.img_route_debug, (0, 0),
                        fx=self.cfg["minimap"]["debug_window_upscale"],
                        fy=self.cfg["minimap"]["debug_window_upscale"],
                        interpolation=cv2.INTER_NEAREST)

        self.profiler.mark("Debug Window Show")

        # Update FPS timer
        self.t_last_frame = time.time()

        # Print profiler result
        if self.cfg["profiler"]["enable"] and \
            self.profiler.total_frames % self.cfg["profiler"]["print_frequency"] == 0:
            logger.info('\n' + self.profiler.report())

        return 0 # frame done

    def loop(self):
        '''
        Auto Bot main loop
        Only run when call autobot from UI framework and AutoBotController
        '''
        # Make sure player is in party
        # (wingpu: only with wingpu.ensure_party_on_start, it presses keys/clicks
        #  before anything is detected)
        if is_wingpu():
            if (self.cfg.get("wingpu", {}) or {}).get("ensure_party_on_start", False):
                self.ensure_is_in_party()
        elif not is_mac():
            activate_game_window(self.capture.window_title)
            time.sleep(0.3)
            self.ensure_is_in_party()

        while not self.kb.is_terminated:

            t_start = time.time()

            # Process one game window frame
            self.is_frame_done = False
            try:
                ret = self.run_once()
            except Exception as e: # a detector bug must not leave keys held or kill the loop
                ret = -1
                logger.error(f"[loop] run_once failed: {type(e).__name__}: {e}")
                if not getattr(self, "_logged_trace", False):
                    import traceback
                    logger.error(traceback.format_exc())
                    self._logged_trace = True
                self.kb.set_command("none none none")
                self.kb.release_all_key()
                time.sleep(1.0)

            # Only proceed if the frame is valid
            if ret == 0:
                # Draw image on debug window
                if self.is_show_debug_window and self.is_ui:
                    img_frame_debug_emit = self.img_frame_debug[:
                        self.cfg["ui_coords"]["ui_y_start"], :].copy()
                    img_route_debug_emit = self.img_route_debug.copy()
                    self.image_debug_signal.emit(img_frame_debug_emit)
                    self.route_map_viz_signal.emit(img_route_debug_emit)
            else:
                pass
                # logger.warning("Skipped debug window update due to invalid frame.")

            self.is_frame_done = True
            self.frame_count += 1

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / self.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

def main(args):
    '''
    This main function works as a fake autoBotController
    This function will only be called when the using terminal to
    run this script
    '''
    #####################
    ### Init Auto Bot ###
    #####################
    try:
        mapleStoryAutoBot = MapleStoryAutoBot(args)
    except Exception as e:
        logger.error(f"MapleStoryAutoBot Init failed: {e}")
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Init Successfully")

    ####################
    ### Apply Config ###
    ####################
    # Config layers: default -> platform -> config_<cfg>.yaml -> user override
    # (/work/msbot_config.yaml on wingpu, written by the web UI)
    from src.utils.config_layers import build_cfg
    cfg = build_cfg(args.cfg)
    # OpenCV's thread pool burns CPU on the small images the bot works on
    # (measured: 66 -> 21 ms CPU per frame with 1 thread, same wall time)
    n_threads = cfg["system"].get("opencv_threads", 1 if is_wingpu() else -1)
    if n_threads is not None and n_threads >= 0:
        cv2.setNumThreads(int(n_threads))
    # Dump config to log for debugging
    logger.debug(yaml.dump(cfg, sort_keys=False,
                 indent=2, default_flow_style=False))
    # autoBot load config
    mapleStoryAutoBot.load_config(cfg)

    #####################
    ### Start AutoBot ###
    #####################
    try:
        mapleStoryAutoBot.start() # Start all threads in autoBot
    except Exception as e:
        logger.error(f"MapleStoryAutoBot start failed: {e}")
        mapleStoryAutoBot.terminate_threads() # Terminate all threads
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Start Successfully")

    # Start record game window for debugging
    if args.record:
        mapleStoryAutoBot.start_record()

    # Headless (wingpu script container): control file / HTTP instead of
    # hotkeys, debug images on disk instead of OpenCV windows
    if is_wingpu() or args.headless:
        import signal
        from src.input.WingpuControl import WingpuControl
        def on_sigterm(signum, frame):
            raise KeyboardInterrupt # stop from the manager page / docker: release keys first
        signal.signal(signal.SIGTERM, on_sigterm)
        control = WingpuControl(mapleStoryAutoBot, mapleStoryAutoBot.cfg)
        try:
            while not mapleStoryAutoBot.is_terminated and not control.quit:
                control.tick()
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("Interrupted (SIGINT/SIGTERM)")
        control.stop()
        mapleStoryAutoBot.terminate_threads()
        if mapleStoryAutoBot.kb is not None:
            mapleStoryAutoBot.kb.release_all_key()
        if is_wingpu():
            from src.input.WingpuKeyboard import get_keyboard
            get_keyboard().release_all()
        if control.restart:
            # Settings applied from the web UI: start this process again in place
            # (same PID, so docker exec / the manager's script runner keep it).
            # --wingpu_live is dropped: a restart always comes back in dry run
            # unless the user file says remember_live.
            argv = [a for a in sys.argv[1:] if a != "--wingpu_live"]
            cmd = [sys.executable, "-m", "src.engine.MapleStoryAutoLevelUp"] + argv
            logger.info(f"[restart] exec {' '.join(cmd)}")
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, cmd)
        return

    from src.input.KeyBoardListener import KeyBoardListener
    kb_listener = KeyBoardListener(is_autobot=True)
    kb_listener.register_func_key_handler('f1', mapleStoryAutoBot.kb.toggle_enable)
    kb_listener.register_func_key_handler('f2', mapleStoryAutoBot.screenshot_img_frame)
    kb_listener.register_func_key_handler('f12', mapleStoryAutoBot.terminate_threads)

    # While loop
    while not mapleStoryAutoBot.is_terminated:
        # Show debug image on window
        if mapleStoryAutoBot.is_frame_done:
            if mapleStoryAutoBot.img_frame_debug is not None:
                cv2.imshow("Game Window Debug",
                    mapleStoryAutoBot.img_frame_debug[:
                        mapleStoryAutoBot.cfg["ui_coords"]["ui_y_start"], :])

            if mapleStoryAutoBot.img_route_debug is not None:
                cv2.imshow("Route Map Debug", mapleStoryAutoBot.img_route_debug)

        cv2.waitKey(1)

        time.sleep(0.01)

    #########################
    ### Terminate AutoBot ###
    #########################
    mapleStoryAutoBot.terminate_threads() # Terminate all threads

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--debug',
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        '--record',
        action="store_true",
        help="Record debug window"
    )

    parser.add_argument(
        '--disable_viz',
        action="store_true",
        help="Disable viz debug window"
    )

    parser.add_argument(
        '--test_image',
        default="",
        help="Pass in image in test/XXX.png"
    )

    parser.add_argument(
        '--init_state',
        default="",
        help="choose the init_state"
    )

    parser.add_argument(
        '--headless',
        action="store_true",
        help="No OpenCV windows / hotkeys; use control file + debug images "
             "(always on for the wingpu platform)"
    )

    parser.add_argument(
        '--wingpu_live',
        action="store_true",
        help="wingpu: really send key/mouse input to the VM (default: dry run)"
    )

    args = parser.parse_args()
    args.is_ui = False # Always set False for command line

    # Set logger level
    if args.debug:
        logger.set_level(logging.DEBUG)

    main(args)
