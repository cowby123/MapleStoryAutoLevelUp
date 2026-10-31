# wingpu 平台（遊戲在 wingpu VM 裡，bot 在該 VM 的腳本容器裡）

遊戲（MapleStory Worlds / Artale）跑在 wingpu 主機上的 Windows VM；bot 跑在那台 VM 的**腳本容器**（Linux，2 CPU / 2 GB）。
畫面從 wingpu SDK 讀（`/wingpu/frame`，整個 guest 螢幕），按鍵 / 滑鼠經 `/wingpu/input.sock` 送進 guest。
上游的結構不變：和 macOS 一樣多一組 `src/input/` 的實作，其餘只是幾個平台開關。

| 上游（Windows） | wingpu |
|---|---|
| `GameWindowCapturor`（windows-capture 抓視窗） | `src/input/GameWindowCapturorForWingpu.py` `WingpuCapturor`：整個 guest 畫面 → 找遊戲 client 區 → 裁切、縮放成 `game_window.size`，上面補 `title_bar_height` 行（和 Windows 抓到的版面一樣） |
| pyautogui `keyDown/keyUp`、`press_key` | `src/input/WingpuKeyboard.py`：pyautogui 鍵名 → `KeyboardEvent.code`（`KEY_MAP`），**dry run 預設開** |
| `click_in_game_window`（pygetwindow + pyautogui） | 視窗座標 × 縮放 + client 位移 → guest 座標，`vm.click`（同樣受 dry run 管） |
| `activate_game_window` / `resize_window` | 不做事（容器看不到也動不了 guest 的視窗；遊戲要自己在前景） |
| pynput 熱鍵 F1/F2/F12、OpenCV 視窗 | `src/input/WingpuControl.py`：`control.json` / HTTP，debug 圖存檔 |
| — | `src/input/WingpuVM.py`：共用一條 SDK 連線 |

平台選擇（`src/utils/common.py` `get_platform()`）：環境變數 `WINGPU=1/0` > config `system.platform`（`auto`/`windows`/`mac`/`wingpu`）> auto（Linux 上有 `/wingpu/frame` 就是 wingpu）。
是 wingpu 時，config 依序疊：`config_default.yaml` → `config_wingpu.yaml` → `config_<--cfg>.yaml`。
Windows-only 的 import（pywin32、windows-capture、pygetwindow、pynput、pyautogui）在 Linux 上不載入。

## 安裝（容器裡，一次）

容器映像已經有 Python 3.12、numpy、opencv-headless、pyyaml 和 `import wingpu`。缺的裝在 `/work`（永久）的 venv：

```sh
# 把程式放進 /work（code-server 的終端機、SSH，或 host 上 docker exec 都可以）
cd /work && git clone <this repo> MapleStoryAutoLevelUp      # 或上傳 / tar 進來
python3 -m venv --system-site-packages /work/venv-msbot       # system-site-packages：沿用映像裡的 wingpu SDK、cv2
/work/venv-msbot/bin/pip install -r /work/MapleStoryAutoLevelUp/requirements-wingpu.txt
```

`requirements-wingpu.txt` = opencv-python-headless、numpy、pyyaml、ruamel.yaml、requests（PySide6 的 UI 不用）。

## 執行

```sh
cd /work/MapleStoryAutoLevelUp
/work/venv-msbot/bin/python -m src.engine.MapleStoryAutoLevelUp --cfg wingpu_maple
```

- 一律 headless（`--headless` 在 Windows 上也能用）。`Ctrl+C` / SIGINT / SIGTERM 會放開所有按著的鍵再結束。
- 也可以從管理頁「腳本容器」執行：上傳一個 `run_bot.sh`（內容就是上面兩行）。
- 檢查（不需要 VM、不送任何輸入）：`WINGPU=1 /work/venv-msbot/bin/python tools/wingpu_check.py`。

### 控制（取代 F1 / F12）

`/work/control.json`（每 0.5 s 檢查，改了才生效）：

```sh
echo '{"run": false}' > /work/control.json   # 暫停：放開所有鍵，只看不動
echo '{"run": true}'  > /work/control.json   # 繼續
echo '{"quit": true}' > /work/control.json   # 結束
```

沒有這個檔案時，開始狀態 = `wingpu.start_running`（預設 true）。

### Debug 輸出（取代 OpenCV 視窗）

每 `wingpu.debug_interval` 秒寫到 `/work/debug/`：

| 檔案 | 內容 |
|---|---|
| `latest.png` | 原本的 debug 視窗（攻擊範圍、怪物框、party red bar、小地圖框、指令、FPS） |
| `latest_route.png` | route 地圖（normal 模式） |
| `latest_guest.jpg` | 整個 guest 畫面（半尺寸）＋偵測到的 client 區 |
| `status.json` | client 區、玩家位置、小地圖、怪物、HP/MP/EXP、bot 指令、**送出（或 dry run 時會送）的動作**、bot loop fps、process CPU |

`wingpu.http_port: 8765` 時容器裡另有 `GET /status`、`/debug.png`、`/guest.jpg`、`/mjpeg`，`POST /run`、`/pause`、`/quit`。
容器網路不對外：從 host 用 `curl http://10.231.0.<x>:8765/status`，或在 code-server 裡開。

## config（`wingpu:` 區段，預設值在 `config/config_wingpu.yaml`，在自己的 config 覆蓋）

| key | 預設 | |
|---|---|---|
| `dry_run` | `True` | 只記錄、不送輸入（見下） |
| `detect` | `auto` | `auto`：有 `client_rect` 用它，否則 title template，否則整個畫面；`fixed` / `template` / `fullscreen` |
| `client_rect` | `[]` | `[x, y, w, h]`，遊戲 client 區（不含標題列）的 guest 座標 |
| `title_template` | `""` | 視窗標題列的圖（`tools/wingpu_calibrate.py --rect` 產生），找得到就跟著視窗移動 |
| `title_thres` | `0.05` | template 比對門檻（TM_SQDIFF_NORMED） |
| `title_to_client` | `[0, 0]` | template 左上角 → client 左上角 |
| `client_size_guest` | `[1282, 693]` | template 找到時 client 的大小（guest 像素，[w, h]） |
| `redetect_interval` | `5` | 秒；重找 client 區（視窗移動 / 換解析度） |
| `interpolation` | `nearest` | client 不是 `game_window.size` 時的縮放法 |
| `start_running` / `control_file` / `debug_dir` / `debug_interval` / `http_port` | | 見上 |
| `url` / `vm` / `token` | | 只在容器外：走管理頁的 network API |

其他沿用上游：`game_window.size: [693, 1282]`（bot 的工作尺寸）、`title_bar_height: 59`（點擊座標的 y 含標題列，和 Windows 一樣）、
`system.fps_limit_window_capturor`（讀 guest 畫面的 fps，也是向 host 要的 fps）。

## 遊戲視窗要怎麼擺（重要）

bot 的偵測大量用**精確顏色**：小地圖的純白（255,255,255）邊框、小地圖上玩家點的精確 BGR、route 圖的顏色、party red bar 的 HSV 範圍、
HP/MP 條的白色外框。**遊戲全螢幕（1920x1080）時，畫面看起來是把約 1280x720 的畫面平滑放大 1.5 倍**（UI 大小剛好是 1.5 倍、邊緣全部被混色）
（小地圖邊框最白只有 247），這些偵測全部失效——縮回 1282x693 也救不回來。2026-10-31 在 maple 上量到的就是這樣（下面「驗證」）。

所以和上游一樣：**遊戲要用視窗模式、最小解析度**（上游在 Windows 上把視窗設成 1296x759，client = 1282x693），guest 的 DPI 縮放 100%。
然後擇一：
1. 固定：`tools/wingpu_calibrate.py --show` 存下 guest 畫面，讀出 client 區，寫 `wingpu.client_rect: [x, y, 1282, 693]`；
2. 跟著視窗：`python tools/wingpu_calibrate.py --rect X,Y,1282,693` 會存標題列 template，照它印的幾行設定。

視窗被別的東西蓋住、或不在前景時，bot 看到的就是被蓋住的畫面、按鍵也會送到前景的程式——`is_game_window_active()` 在 wingpu 上永遠是 True。

## Dry run 與開啟真的輸入

預設**不送任何鍵盤 / 滑鼠到 guest**：`WingpuKeyboard._send()` 是唯一碰 input socket 的地方，dry run 時直接 return；
每個動作記一行 `[DRY-RUN] key_down ArrowLeft ('left')` / `press KeyZ ('z') 0.05s` / `click window(1140,730) -> guest(1440,871)`，
也在 `status.json` 的 `input.recent`。（SDK 每秒送的 `watch` 只是向 host 要畫面，不是輸入。）

要真的送，二擇一（都是明確的動作）：
- 自己的 config：`wingpu: {dry_run: false}`
- 或命令列 `--wingpu_live`

開了以後 log 會印 `[WingpuKeyboard] LIVE: key/mouse input WILL be sent to the guest`。
注意：輸入法要是英數（中文輸入法會吃掉字母鍵）；有人用瀏覽器 pointer lock 玩的時候，點擊會被 host 拒絕（鍵盤照送）。

## 驗證（2026-10-31，maple，dry run）

- `tools/wingpu_check.py`：key 表涵蓋 config/*.yaml 用到的 21 個鍵 + 常用鍵，全部是 guest 認得的 code；裁切 / 縮放 / 標題列 /
  點擊座標（全螢幕 1.498x1.558、固定 1:1、標題 template）；dry run 時所有輸入路徑（KeyBoardController、HealthMonitor、RuneSolver、
  near_rune 的 `press_key`、`release_all_key`、`click_in_game_window`）送到 VM 0 次，live 對假 VM 送的訊息正確；
  `src/` 裡每個用到 pyautogui/win32gui/pygetwindow/Quartz 的函式不是有 `is_wingpu()` 擋就是 wingpu 不會執行。全過。
- 在 maple 的容器裡跑 `--cfg wingpu_maple`（gon_pig）：client 區 = 整個 1920x1080（fullscreen），bot frame 752x1282 正確；
  bot loop 10 fps（每幀 68 ms，77% 在上游的 hunting 狀態），容器 CPU 約 1.2–1.5 core（其中讀畫面 15 fps ≈ 0.3 core），記憶體 ~290 MB；
  因為全螢幕，小地圖、HP/MP 條、party red bar 都找不到 → bot 一直在 watchdog 的隨機動作（log 裡的 DRY-RUN 動作就是這些）。
  `control.json` 的暫停（放開按著的鍵、動作停止）和結束都驗過。
