_LANG = 'en'

_TRANSLATIONS = {
    'zh': {
        'MapleStory AutoLevelUp': '楓之谷自動升級',
        'Main': '主頁',
        'Advanced Settings': '進階設定',
        'Game Window Viz': '遊戲視窗預覽',
        'Route Map Viz': '路線地圖預覽',
        '⚔️ Attack Settings': '⚔️ 攻擊設定',
        'Basic': '基本',
        'AOE Skill': '範圍技能',
        'Attack Mode:': '攻擊模式:',
        'Range X:': '範圍 X:',
        'Cooldown (s):': '冷卻時間(秒):',
        'Range Y:': '範圍 Y:',
        '🎮 Key Bindings': '🎮 快捷鍵',
        'Basic Attack:': '普通攻擊:',
        'Teleport:': '瞬移:',
        'Party:': '組隊:',
        'AOE Skill:': '範圍技能:',
        'Jump:': '跳躍:',
        'Home:': '回家:',
        'Auto Buff': '自動增益',
        '+ Add Buff Key': '+ 新增增益鍵',
        '❤️ Pet Skills': '❤️ 寵物技能',
        'Auto Add HP': '自動補血',
        'Auto Add MP': '自動補魔',
        '🗺️ Map': '🗺️ 地圖',
        'Please select a map:': '請選擇地圖:',
        '📜 Log': '📜 日誌',
        '📂 Load Config': '📂 載入設定',
        '(No config loaded)': '未載入設定',
        '🕹️ Bot Control': '🕹️ 機器人控制',
        '▶ Start (F1)': '▶ 開始 (F1)',
        '⏸ Pause (F1)': '⏸ 暫停 (F1)',
        '📸 Screenshot (F2)': '📸 截圖 (F2)',
        '⏺ Record (F3)': '⏺ 錄製 (F3)',
        '⏹ Stop (F3)': '⏹ 停止 (F3)',
        'Bot Mode:': '機器人模式:',
        'normal': '一般',
        'aux': '輔助',
        'patrol': '巡邏',
        'Selected map: {map_path}': '已選擇地圖: {map_path}',
        '🚧 Work in Progress\nEdit config/config_default.yaml to modify advance settings': '🚧 開發中\n編輯 config/config_default.yaml 以修改進階設定',
        'Attack Key:': '攻擊按鍵:',
        'CoolDown (s):': '冷卻 (秒):',
        'When {title} is below:': '當{title}低於:',
        '%, press ': '%，按下',
        'key.': '鍵。',
        'key, for every ': '鍵，每',
        'seconds.': '秒。',
        'Press': '按下',
        'Only .yaml files are supported.': '僅支援 .yaml 檔案。',
        '{path} cannot be loaded as customized yaml': '無法載入自訂 yaml: {path}',
        'Please enter number between {val_lowest} and {val_highest}.': '請輸入介於 {val_lowest} 到 {val_highest} 的數字。',
        "Press start or 'F1' to start AutoBot": "按下開始或 F1 啟動自動機器人",
        'Select Config File': '選擇設定檔',
        'YAML Files (*.yaml);;All Files (*)': 'YAML 檔 (*.yaml);;所有檔案 (*)'
    }
}

def set_language(lang: str):
    global _LANG
    _LANG = lang


def tr(text: str) -> str:
    return _TRANSLATIONS.get(_LANG, {}).get(text, text)
