"""生成确定性演示 SFT / 评估数据（纯标准库，可重复执行）。

产出（默认写到项目 ``data/``）：

* ``demo_sft_en.jsonl`` —— 500 条英文问答（q/a/group/category/source）；
* ``demo_sft_zh.jsonl`` —— 500 条中文问答；
* ``demo_eval_en.jsonl`` —— 30 条英文评估问题（10 in_distribution + 10 edge + 10 unrelated）；
* ``demo_eval_zh.jsonl`` —— 30 条中文评估问题（同上比例）。

数据全部由简单事实/模板确定性生成：固定 seed、固定顺序、无网络、无第三方库。
``prepare_sft.py --source synthetic_en/zh`` 直接读取这些文件；``evaluate.py`` 可读取
``demo_eval_*`` 做定性评估。运行::

    python scripts/build_demo_data.py               # 写入项目 data/
    python scripts/build_demo_data.py --out-dir /tmp/demo   # 写入其他目录（测试用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# 允许从任意 cwd 运行：把项目根加入 sys.path（脚本位于 <root>/scripts/）。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.data.formatters import normalize_group  # noqa: E402

#: 默认每条语言的 SFT / 评估条数。
SFT_TOTAL = 500
EVAL_TOTAL = 30


# ======================================================================
# 工具
# ======================================================================
def _shuffled(items: list, seed: str) -> list:
    """确定性打乱（同一 seed 永远同序）。"""
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def _clock12(hour: int) -> int:
    """把任意小时数映射到 1..12 的钟点表示。"""
    return ((int(hour) - 1) % 12) + 1


def _make(q: str, a: str, category: str, source: str) -> dict:
    """把单条问答规范化为统一记录结构（q/a 去首尾空白，group 由问题归一化得到）。"""
    return {
        "q": q.strip(),
        "a": a.strip(),
        "group": normalize_group(q),
        "category": category,
        "source": source,
    }


def _take_families(
    families: list[tuple[str, int, list[tuple[str, str]]]],
    source: str,
    seen: set[str],
) -> list[dict]:
    """按 (category, 目标条数, 候选对) 依次收取记录，q 去重。"""
    records: list[dict] = []
    for category, count, candidates in families:
        added = 0
        for question, answer in candidates:
            key = question.strip().casefold()
            if not key or not answer.strip() or key in seen:
                continue
            seen.add(key)
            records.append(_make(question, answer, category, source))
            added += 1
            if added >= count:
                break
        if added < count:
            raise RuntimeError(
                f"类别 {category} 候选不足：需要 {count} 条，只生成 {added} 条"
            )
    return records


def _fill_math(
    records: list[dict],
    seen: set[str],
    target: int,
    source: str,
    chinese: bool,
) -> None:
    """不足 target 条时用简单算式补足（确定性，类别 math_extra）。"""
    pairs: list[tuple[str, str]] = []
    for a in range(100, 160):
        for b in range(2, 20):
            if chinese:
                pairs.append((f"{a} + {b} 等于多少？", str(a + b)))
                pairs.append((f"{a} - {b} 等于多少？", str(a - b)))
            else:
                pairs.append((f"What is {a} + {b}?", str(a + b)))
                pairs.append((f"What is {a} - {b}?", str(a - b)))
    for question, answer in _shuffled(pairs, f"fill:{source}"):
        if len(records) >= target:
            return
        key = question.casefold()
        if key in seen:
            continue
        seen.add(key)
        records.append(_make(question, answer, "math_extra", source))
    if len(records) < target:
        raise RuntimeError(f"补足失败：{len(records)} < {target}")


# ======================================================================
# 英文候选池
# ======================================================================
EN_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Brasilia"),
    ("Egypt", "Cairo"), ("India", "New Delhi"), ("China", "Beijing"), ("Russia", "Moscow"),
    ("Mexico", "Mexico City"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Turkey", "Ankara"),
    ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"), ("Kenya", "Nairobi"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
    ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Switzerland", "Bern"),
    ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Hungary", "Budapest"),
    ("Ireland", "Dublin"), ("New Zealand", "Wellington"), ("Morocco", "Rabat"),
    ("United Kingdom", "London"), ("Israel", "Jerusalem"), ("Indonesia", "Jakarta"),
    ("Philippines", "Manila"),
]

EN_OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("happy", "sad"),
    ("light", "dark"), ("old", "new"), ("yes", "no"), ("open", "closed"), ("full", "empty"),
    ("long", "short"), ("high", "low"), ("wet", "dry"), ("hard", "soft"), ("loud", "quiet"),
    ("early", "late"), ("rich", "poor"), ("near", "far"), ("clean", "dirty"), ("strong", "weak"),
    ("thick", "thin"), ("young", "old"), ("in", "out"), ("on", "off"), ("left", "right"),
    ("true", "false"), ("day", "night"), ("summer", "winter"), ("morning", "evening"),
    ("question", "answer"), ("begin", "end"), ("buy", "sell"),
]

EN_ANIMAL_LEGS = {
    "spider": 8, "dog": 4, "cat": 4, "bird": 2, "horse": 4, "cow": 4, "ant": 6,
    "octopus": 8, "rabbit": 4, "chicken": 2, "elephant": 4, "bee": 6, "frog": 4,
    "pig": 4, "duck": 2, "sheep": 4, "crab": 10, "centipede": 100, "tiger": 4,
    "butterfly": 6,
}

EN_ANIMAL_SOUNDS = {
    "cow": "moo", "dog": "woof", "cat": "meow", "duck": "quack", "sheep": "baa",
    "horse": "neigh", "pig": "oink", "chicken": "cluck", "bee": "buzz", "frog": "croak",
    "bird": "tweet", "owl": "hoot", "lion": "roar", "snake": "hiss", "goat": "bleat",
    "mouse": "squeak", "wolf": "howl", "rooster": "cock-a-doodle-doo", "turkey": "gobble",
    "donkey": "hee-haw",
}

EN_LETTER_WORDS = [
    "apple", "banana", "orange", "grape", "lemon", "peach", "cherry", "melon",
    "computer", "keyboard", "monitor", "window", "garden", "flower", "river", "mountain",
    "elephant", "giraffe", "penguin", "dolphin", "tiger", "monkey", "rabbit", "squirrel",
    "school", "teacher", "student", "library", "pencil", "notebook", "bicycle", "airplane",
    "morning", "evening", "summer", "winter", "friend", "family", "music", "picture",
    "yellow", "purple", "silver", "golden",
]

EN_COLORS = [
    ("red", "yellow", "orange"), ("blue", "yellow", "green"), ("red", "blue", "purple"),
    ("black", "white", "gray"), ("red", "white", "pink"), ("blue", "white", "light blue"),
    ("yellow", "green", "lime"), ("red", "green", "brown"),
]

EN_GREETINGS = [
    ("Hello!", "Hello! How can I help you?"),
    ("Hi!", "Hi! Nice to meet you."),
    ("Good morning!", "Good morning! Have a nice day."),
    ("Good afternoon!", "Good afternoon!"),
    ("Good evening!", "Good evening!"),
    ("How are you?", "I am fine, thank you."),
    ("What is your name?", "I am a small language model."),
    ("Nice to meet you.", "Nice to meet you too."),
    ("Thank you.", "You are welcome."),
    ("Thanks a lot!", "You are welcome."),
    ("Sorry.", "That is all right."),
    ("Excuse me.", "Yes, can I help you?"),
    ("Goodbye!", "Goodbye! See you next time."),
    ("See you later.", "See you later."),
    ("Good night!", "Good night! Sleep well."),
    ("Please help me.", "Of course. What do you need?"),
    ("What can you do?", "I can answer simple questions."),
    ("How old are you?", "I am a computer program, so I have no age."),
    ("Where are you from?", "I live in a computer."),
    ("Do you like music?", "I like reading about music."),
    ("What day is it today?", "I do not know the date."),
    ("Can you speak Chinese?", "I am learning one character at a time."),
    ("What is your favorite color?", "I do not have a favorite color."),
    ("Tell me something.", "Learning a little every day is powerful."),
    ("Let us learn together.", "Great idea! Let us start."),
    ("Are you a robot?", "I am a language model, not a robot."),
    ("Do you sleep?", "No, I do not need sleep."),
    ("How is the weather?", "I cannot see the weather."),
    ("One more question.", "Sure, go ahead."),
    ("That is all for now.", "Thank you. Goodbye!"),
]

EN_SPELLING = [
    "cat", "dog", "bird", "fish", "book", "tree", "house", "water", "light", "music",
    "green", "happy", "table", "phone", "train", "cloud", "bread", "apple", "sugar",
    "river", "stone", "dream", "quick", "brave", "smile",
]

EN_SYNONYMS = [
    ("happy", "glad"), ("big", "large"), ("small", "tiny"), ("fast", "quick"),
    ("smart", "clever"), ("beautiful", "pretty"), ("angry", "mad"), ("afraid", "scared"),
    ("begin", "start"), ("end", "finish"), ("buy", "purchase"), ("help", "assist"),
    ("show", "display"), ("choose", "select"), ("close", "shut"), ("difficult", "hard"),
    ("easy", "simple"), ("quiet", "silent"), ("rich", "wealthy"), ("strong", "powerful"),
    ("tired", "weary"), ("funny", "amusing"), ("calm", "peaceful"), ("cold", "chilly"),
    ("bright", "shiny"), ("correct", "right"), ("famous", "well-known"), ("gift", "present"),
    ("journey", "trip"), ("reply", "answer"),
]

EN_SEASONS = [
    ("Which season comes after spring?", "summer"),
    ("Which season comes after summer?", "autumn"),
    ("Which season comes after autumn?", "winter"),
    ("Which season comes after winter?", "spring"),
    ("Which season is the hottest?", "summer"),
    ("Which season is the coldest?", "winter"),
    ("In which season do leaves turn yellow?", "autumn"),
    ("In which season do flowers bloom?", "spring"),
    ("In which season does it snow most?", "winter"),
    ("Which season comes before winter?", "autumn"),
    ("Which season comes before summer?", "spring"),
    ("Which season is between spring and autumn?", "summer"),
    ("Which season brings harvest?", "autumn"),
    ("Which season has the longest days?", "summer"),
    ("Which season is best for planting?", "spring"),
    ("Which season do children like for snowballs?", "winter"),
    ("Which season follows the rainy season?", "autumn"),
    ("Which season starts after December?", "winter"),
    ("Which season starts after March?", "spring"),
    ("Which season starts after June?", "summer"),
]

EN_FACTS = [
    ("Where does the sun rise?", "in the east"),
    ("At what temperature does water boil?", "100 degrees Celsius"),
    ("What is frozen water called?", "ice"),
    ("What does the earth go around?", "the sun"),
    ("What goes around the earth?", "the moon"),
    ("How many colors are in a rainbow?", "seven"),
    ("How many eyes does a person have?", "two"),
    ("How many hours are in a day?", "24"),
    ("How many days are in a week?", "seven"),
    ("What do birds use to fly?", "wings"),
    ("Where do fish live?", "in water"),
    ("What do plants need to grow?", "sunlight and water"),
    ("What do cats like to catch?", "mice"),
    ("What do butterflies come from?", "caterpillars"),
    ("What do bees make?", "honey"),
    ("Can penguins fly?", "no"),
    ("Where do camels live?", "in the desert"),
    ("Which shape has three sides?", "a triangle"),
    ("Which planet do we live on?", "earth"),
    ("What is the largest animal in the ocean?", "the blue whale"),
]

EN_PLACES = [
    ("What is the largest ocean on earth?", "the Pacific Ocean"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
    ("What is the longest river in the world?", "the Nile"),
    ("What is the largest desert in the world?", "the Sahara"),
    ("What is the largest country by area?", "Russia"),
    ("What is the smallest country in the world?", "Vatican City"),
    ("Which continent is Egypt in?", "Africa"),
    ("Which continent is Brazil in?", "South America"),
    ("Which continent is Japan in?", "Asia"),
    ("Which ocean is between America and Europe?", "the Atlantic Ocean"),
    ("What is the capital city of France?", "Paris"),
    ("What is the capital city of Japan?", "Tokyo"),
    ("Which country is famous for pizza?", "Italy"),
    ("Which country is famous for the Great Wall?", "China"),
    ("Where is the Eiffel Tower?", "in Paris"),
    ("Where is the Great Pyramid?", "in Egypt"),
    ("Which country has the city of Sydney?", "Australia"),
    ("Which country has the city of Toronto?", "Canada"),
    ("Which sea is between Europe and Africa?", "the Mediterranean Sea"),
    ("What is the coldest continent?", "Antarctica"),
]

EN_GRAMMAR_VERBS = [
    ("go", "went"), ("eat", "ate"), ("drink", "drank"), ("run", "ran"), ("see", "saw"),
    ("take", "took"), ("make", "made"), ("write", "wrote"), ("swim", "swam"), ("fly", "flew"),
    ("sing", "sang"), ("speak", "spoke"), ("buy", "bought"), ("bring", "brought"),
    ("think", "thought"), ("teach", "taught"), ("catch", "caught"), ("build", "built"),
    ("send", "sent"), ("spend", "spent"), ("leave", "left"), ("sleep", "slept"),
    ("feel", "felt"), ("win", "won"), ("lose", "lost"),
]

EN_SHAPES = [
    ("triangle", 3), ("square", 4), ("pentagon", 5), ("hexagon", 6),
    ("heptagon", 7), ("octagon", 8), ("rectangle", 4), ("rhombus", 4),
]

EN_NEXT_NUMBERS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]


def _en_families() -> list[tuple[str, int, list[tuple[str, str]]]]:
    """英文 20 个类别，合计 500 条。"""
    addition = [
        (f"What is {a} + {b}?", str(a + b))
        for a in range(12, 32)
        for b in range(12, 32)
    ]
    subtraction = [
        (f"What is {a} - {b}?", str(a - b))
        for a in range(30, 60)
        for b in range(2, 16)
    ]
    multiplication = [
        (f"What is {a} times {b}?", str(a * b))
        for a in range(2, 11)
        for b in range(2, 11)
    ]
    hours = [
        (
            f"What time is it {offset} hours after {hour} o'clock?",
            f"{_clock12(hour + offset)} o'clock",
        )
        for hour in range(1, 13)
        for offset in (2, 3, 4)
    ]
    return [
        ("capitals", 34, _shuffled([(f"What is the capital of {c}?", cap) for c, cap in EN_CAPITALS], "en:cap")),
        ("math_addition", 45, _shuffled(addition, "en:add")),
        ("math_subtraction", 25, _shuffled(subtraction, "en:sub")),
        ("math_multiplication", 30, _shuffled(multiplication, "en:mul")),
        ("opposites", 30, _shuffled([(f"What is the opposite of {w}?", opp) for w, opp in EN_OPPOSITES], "en:opp")),
        ("animal_legs", 20, _shuffled([(f"How many legs does a {animal} have?", str(legs)) for animal, legs in EN_ANIMAL_LEGS.items()], "en:legs")),
        ("animal_sounds", 20, _shuffled([(f"What sound does a {animal} make?", sound) for animal, sound in EN_ANIMAL_SOUNDS.items()], "en:sound")),
        ("days_months", 19, _shuffled(
            [(f"Which day comes after {day}?", EN_DAYS[(i + 1) % 7]) for i, day in enumerate(EN_DAYS)]
            + [(f"Which month comes after {month}?", EN_MONTHS[(i + 1) % 12]) for i, month in enumerate(EN_MONTHS)],
            "en:day",
        )),
        ("letter_count", 38, _shuffled([(f"How many letters are in the word '{w}'?", str(len(w))) for w in EN_LETTER_WORDS], "en:letter")),
        ("next_number", 20, _shuffled([(f"What number comes after {n}?", str(n + 1)) for n in EN_NEXT_NUMBERS], "en:next")),
        ("colors", 8, _shuffled([(f"What color do you get by mixing {one} and {two}?", mixed) for one, two, mixed in EN_COLORS], "en:color")),
        ("greetings", 30, _shuffled(EN_GREETINGS, "en:greet")),
        ("spelling", 25, _shuffled([(f"How do you spell '{w}'?", "-".join(w)) for w in EN_SPELLING], "en:spell")),
        ("synonyms", 30, _shuffled([(f"What is a synonym of {w}?", syn) for w, syn in EN_SYNONYMS], "en:syn")),
        ("seasons", 20, _shuffled(EN_SEASONS, "en:season")),
        ("facts", 20, _shuffled(EN_FACTS, "en:fact")),
        ("time", 25, _shuffled(hours, "en:time")),
        ("shapes", 16, _shuffled(
            [(f"How many sides does a {shape} have?", str(sides)) for shape, sides in EN_SHAPES]
            + [(f"Which shape has {sides} sides?", shape) for shape, sides in EN_SHAPES
               if sum(1 for _s, n in EN_SHAPES if n == sides) == 1]
            + [
                ("How many sides does a circle have?", "0"),
                ("What shape has four equal sides?", "a square"),
                ("What shape is round?", "a circle"),
            ],
            "en:shape",
        )),
        ("grammar", 25, _shuffled([(f"What is the past tense of '{verb}'?", past) for verb, past in EN_GRAMMAR_VERBS], "en:gram")),
        ("places", 20, _shuffled(EN_PLACES, "en:place")),
    ]


EN_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EN_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

EN_EDGE_EVAL = [
    ("?", "That is not a complete question."),
    ("!!!!!", "That is not a complete question."),
    ("What is the meaning of life?", "There is no single answer."),
    ("1234567890", "That is a number, not a question."),
    ("Repeat after me: hello", "hello"),
    ("What is 2 + 2 + 2?", "6"),
    ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side."),
    ("你好", "This demo dataset is in English."),
    ("What is the capital of Atlantis?", "Atlantis is not a real country."),
    ("How do you spell 'supercalifragilisticexpialidocious'?", "It is a very long word."),
]

EN_UNRELATED_EVAL = [
    ("Explain gravity in one sentence.", "Gravity pulls objects toward each other."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare."),
    ("What is the boiling point of water in Fahrenheit?", "212 degrees Fahrenheit."),
    ("Translate 'hello' into French.", "bonjour"),
    ("How many minutes are in an hour?", "60"),
    ("What color is the sky on a clear day?", "blue"),
    ("Name three planets.", "Mercury, Venus, and Earth."),
    ("What is the square root of 81?", "9"),
    ("What is 100 divided by 4?", "25"),
    ("What language is spoken in Brazil?", "Portuguese."),
]


# ======================================================================
# 中文候选池
# ======================================================================
ZH_CAPITALS = [
    ("法国", "巴黎"), ("日本", "东京"), ("意大利", "罗马"), ("西班牙", "马德里"),
    ("德国", "柏林"), ("加拿大", "渥太华"), ("澳大利亚", "堪培拉"), ("巴西", "巴西利亚"),
    ("埃及", "开罗"), ("印度", "新德里"), ("中国", "北京"), ("俄罗斯", "莫斯科"),
    ("墨西哥", "墨西哥城"), ("希腊", "雅典"), ("葡萄牙", "里斯本"), ("土耳其", "安卡拉"),
    ("韩国", "首尔"), ("泰国", "曼谷"), ("越南", "河内"), ("阿根廷", "布宜诺斯艾利斯"),
    ("智利", "圣地亚哥"), ("秘鲁", "利马"), ("肯尼亚", "内罗毕"), ("挪威", "奥斯陆"),
    ("瑞典", "斯德哥尔摩"), ("丹麦", "哥本哈根"), ("芬兰", "赫尔辛基"), ("波兰", "华沙"),
    ("奥地利", "维也纳"), ("瑞士", "伯尔尼"), ("荷兰", "阿姆斯特丹"), ("比利时", "布鲁塞尔"),
    ("匈牙利", "布达佩斯"), ("爱尔兰", "都柏林"), ("新西兰", "惠灵顿"), ("摩洛哥", "拉巴特"),
    ("英国", "伦敦"), ("以色列", "耶路撒冷"), ("印度尼西亚", "雅加达"), ("菲律宾", "马尼拉"),
]

ZH_OPPOSITES = [
    ("大", "小"), ("高", "矮"), ("快", "慢"), ("冷", "热"), ("黑", "白"),
    ("上", "下"), ("前", "后"), ("左", "右"), ("多", "少"), ("新", "旧"),
    ("长", "短"), ("宽", "窄"), ("深", "浅"), ("重", "轻"), ("远", "近"),
    ("明", "暗"), ("开", "关"), ("早", "晚"), ("甜", "苦"), ("软", "硬"),
    ("干", "湿"), ("胖", "瘦"), ("强", "弱"), ("忙", "闲"), ("对", "错"),
    ("真", "假"), ("美", "丑"), ("富", "穷"), ("安静", "吵闹"), ("年轻", "年老"),
]

ZH_ANIMAL_LEGS = {
    "蜘蛛": 8, "狗": 4, "猫": 4, "鸟": 2, "马": 4, "牛": 4, "蚂蚁": 6,
    "章鱼": 8, "兔子": 4, "鸡": 2, "大象": 4, "蜜蜂": 6, "青蛙": 4,
    "猪": 4, "鸭子": 2, "绵羊": 4, "螃蟹": 10, "老虎": 4, "蝴蝶": 6, "蜈蚣": 100,
}

ZH_ANIMAL_SOUNDS = {
    "狗": "汪汪", "猫": "喵喵", "牛": "哞哞", "羊": "咩咩", "鸡": "咯咯",
    "鸭": "嘎嘎", "青蛙": "呱呱", "蜜蜂": "嗡嗡", "鸟": "啾啾", "猪": "哼哼",
    "马": "咴咴", "老鼠": "吱吱", "狼": "嗷呜", "狮子": "吼吼", "老虎": "啸啸",
    "山羊": "咩咩叫", "鸽子": "咕咕", "猫头鹰": "呼呼", "蝉": "知了知了", "蟋蟀": "蛐蛐",
}

ZH_CHAR_WORDS = [
    "苹果", "香蕉", "西瓜", "葡萄", "铅笔", "书包", "太阳", "月亮", "学校", "老师",
    "学生", "中国", "北京", "老虎", "熊猫", "图书", "朋友", "家庭", "音乐", "图画",
    "图书馆", "计算机", "巧克力", "天安门", "电视机", "游戏机", "中国人", "自行车",
    "星期五", "电影院", "公共汽车", "中华人民共和国", "北京天安门", "人工智能", "计算机科学",
    "春天", "夏天", "天空", "大海", "森林",
]

ZH_COLORS = [
    ("红色", "黄色", "橙色"), ("蓝色", "黄色", "绿色"), ("红色", "蓝色", "紫色"),
    ("黑色", "白色", "灰色"), ("红色", "白色", "粉色"), ("蓝色", "白色", "浅蓝色"),
    ("黄色", "绿色", "黄绿色"), ("红色", "绿色", "棕色"),
]

ZH_GREETINGS = [
    ("你好！", "你好！有什么可以帮你的吗？"),
    ("您好！", "您好！很高兴见到您。"),
    ("早上好！", "早上好！祝你一天顺利。"),
    ("下午好！", "下午好！"),
    ("晚上好！", "晚上好！"),
    ("你好吗？", "我很好，谢谢。"),
    ("你叫什么名字？", "我是一个小型语言模型。"),
    ("很高兴认识你。", "我也很高兴认识你。"),
    ("谢谢你。", "不客气。"),
    ("非常感谢！", "不用谢。"),
    ("对不起。", "没关系。"),
    ("打扰一下。", "好的，请问需要什么帮助？"),
    ("再见！", "再见！下次再聊。"),
    ("回头见。", "回头见。"),
    ("晚安！", "晚安！祝你做个好梦。"),
    ("请帮帮我。", "当然，你需要什么帮助？"),
    ("你会做什么？", "我可以回答简单的问题。"),
    ("你多大了？", "我是一个程序，没有年龄。"),
    ("你来自哪里？", "我住在计算机里。"),
    ("你喜欢音乐吗？", "我喜欢读关于音乐的知识。"),
    ("今天是星期几？", "我不知道今天的日期。"),
    ("你会说英语吗？", "我正在一个字符一个字符地学习。"),
    ("你最喜欢什么颜色？", "我没有最喜欢的颜色。"),
    ("给我讲点东西吧。", "每天学习一点点，力量就会很大。"),
    ("我们一起学习吧。", "好主意！我们开始吧。"),
    ("你是机器人吗？", "我是语言模型，不是机器人。"),
    ("你需要睡觉吗？", "不需要，我不睡觉。"),
    ("天气怎么样？", "我看不到天气。"),
    ("再问一个问题。", "好的，请说。"),
    ("就先到这里。", "谢谢，再见！"),
]

ZH_FESTIVALS = [
    ("春节是农历几月几日？", "正月初一"),
    ("中秋节是农历几月几日？", "八月十五"),
    ("端午节是农历几月几日？", "五月初五"),
    ("元宵节是农历几月几日？", "正月十五"),
    ("重阳节是农历几月几日？", "九月初九"),
    ("七夕是农历几月几日？", "七月初七"),
    ("除夕是农历几月几日？", "腊月最后一天"),
    ("腊八节是农历几月几日？", "十二月初八"),
    ("清明节大约在公历几月几日？", "4月5日前后"),
    ("国庆节是几月几日？", "10月1日"),
    ("儿童节是几月几日？", "6月1日"),
    ("劳动节是几月几日？", "5月1日"),
    ("教师节是几月几日？", "9月10日"),
    ("元旦是几月几日？", "1月1日"),
    ("圣诞节是几月几日？", "12月25日"),
    ("建军节是几月几日？", "8月1日"),
    ("植树节是几月几日？", "3月12日"),
    ("全国爱眼日是几月几日？", "6月6日"),
    ("世界地球日是几月几日？", "4月22日"),
    ("世界读书日是几月几日？", "4月23日"),
    ("国际护士节是几月几日？", "5月12日"),
    ("消防宣传日是几月几日？", "11月9日"),
]

ZH_SEASONS = [
    ("春天之后是什么季节？", "夏天"),
    ("夏天之后是什么季节？", "秋天"),
    ("秋天之后是什么季节？", "冬天"),
    ("冬天之后是什么季节？", "春天"),
    ("哪个季节最热？", "夏天"),
    ("哪个季节最冷？", "冬天"),
    ("哪个季节树叶会变黄？", "秋天"),
    ("哪个季节会开花？", "春天"),
    ("哪个季节下雪最多？", "冬天"),
    ("冬天之前是哪个季节？", "秋天"),
    ("夏天之前是哪个季节？", "春天"),
    ("春天和秋天之间是哪个季节？", "夏天"),
    ("哪个季节是收获的季节？", "秋天"),
    ("哪个季节白天最长？", "夏天"),
    ("哪个季节最适合播种？", "春天"),
    ("哪个季节最适合打雪仗？", "冬天"),
    ("雨季之后是哪个季节？", "秋天"),
    ("十二月之后是哪个季节？", "冬天"),
    ("三月之后是哪个季节？", "春天"),
    ("六月之后是哪个季节？", "夏天"),
]

ZH_FACTS = [
    ("太阳从哪边升起？", "东边"),
    ("水在多少摄氏度沸腾？", "100摄氏度"),
    ("水冻成冰是什么状态？", "固体"),
    ("地球围绕什么转？", "太阳"),
    ("月亮围绕什么转？", "地球"),
    ("彩虹有几种颜色？", "七种"),
    ("人有几只眼睛？", "两只"),
    ("一天有多少小时？", "24小时"),
    ("一周有几天？", "七天"),
    ("鸟用什么飞行？", "翅膀"),
    ("鱼生活在哪里？", "水里"),
    ("植物生长需要什么？", "阳光和水"),
    ("猫喜欢抓什么？", "老鼠"),
    ("蝴蝶是由什么变来的？", "毛毛虫"),
    ("蜜蜂会做什么？", "采蜜"),
    ("企鹅会飞吗？", "不会"),
    ("骆驼生活在哪里？", "沙漠里"),
    ("什么形状有三条边？", "三角形"),
    ("我们生活在哪个星球上？", "地球"),
    ("海洋里最大的动物是什么？", "蓝鲸"),
]

ZH_TIME = [
    (f"{hour}点之后{offset}小时是几点？", f"{_clock12(hour + offset)}点")
    for hour in range(1, 13)
    for offset in (2, 3, 4)
]

ZH_SHAPES = [
    ("三角形", 3), ("正方形", 4), ("五边形", 5), ("六边形", 6),
    ("七边形", 7), ("八边形", 8), ("长方形", 4), ("菱形", 4),
]

ZH_QUANTIFIERS = [
    ("马", "匹"), ("牛", "头"), ("花", "朵"), ("书", "本"), ("人", "个"),
    ("鱼", "条"), ("树", "棵"), ("房子", "间"), ("衣服", "件"), ("鞋", "双"),
    ("铅笔", "支"), ("纸", "张"), ("车", "辆"), ("船", "艘"),
    ("狗", "条"), ("猫", "只"), ("鸡", "只"), ("鸟", "只"), ("杯子", "个"),
    ("碗", "个"), ("伞", "把"), ("刀", "把"), ("钥匙", "把"), ("桥", "座"),
    ("山", "座"), ("河", "条"), ("云", "朵"), ("星星", "颗"), ("米", "粒"),
]

ZH_CITIES = [
    ("上海位于哪个国家？", "中国"),
    ("纽约位于哪个国家？", "美国"),
    ("伦敦位于哪个国家？", "英国"),
    ("巴黎位于哪个国家？", "法国"),
    ("东京位于哪个国家？", "日本"),
    ("柏林位于哪个国家？", "德国"),
    ("罗马位于哪个国家？", "意大利"),
    ("首尔位于哪个国家？", "韩国"),
    ("莫斯科位于哪个国家？", "俄罗斯"),
    ("悉尼位于哪个国家？", "澳大利亚"),
    ("多伦多位于哪个国家？", "加拿大"),
    ("开罗位于哪个国家？", "埃及"),
    ("新德里位于哪个国家？", "印度"),
    ("马德里位于哪个国家？", "西班牙"),
    ("曼谷位于哪个国家？", "泰国"),
    ("新加坡位于哪个国家？", "新加坡"),
    ("中国最大的城市是哪里？", "上海"),
    ("日本最大的城市是哪里？", "东京"),
]

ZH_UNITS = [
    ("1米等于多少厘米？", "100厘米"),
    ("1千米等于多少米？", "1000米"),
    ("1公斤等于多少克？", "1000克"),
    ("1小时等于多少分钟？", "60分钟"),
    ("1分钟等于多少秒？", "60秒"),
    ("1天等于多少小时？", "24小时"),
    ("1年等于多少个月？", "12个月"),
    ("1米等于多少分米？", "10分米"),
    ("1分米等于多少厘米？", "10厘米"),
    ("1吨等于多少公斤？", "1000公斤"),
    ("1升等于多少毫升？", "1000毫升"),
    ("1周等于多少天？", "7天"),
    ("1世纪等于多少年？", "100年"),
    ("1元等于多少角？", "10角"),
    ("1角等于多少分？", "10分"),
    ("半年等于多少个月？", "6个月"),
    ("一季度等于多少个月？", "3个月"),
    ("1小时等于多少秒？", "3600秒"),
    ("1公里等于多少里？", "2里"),
    ("1公斤等于多少斤？", "2斤"),
    ("1平方米等于多少平方分米？", "100平方分米"),
    ("1立方米等于多少升？", "1000升"),
]


def _zh_families() -> list[tuple[str, int, list[tuple[str, str]]]]:
    """中文 20 个类别，合计 480 条（余下 20 条由算式补足）。"""
    addition = [
        (f"{a} + {b} 等于多少？", str(a + b))
        for a in range(12, 32)
        for b in range(12, 32)
    ]
    subtraction = [
        (f"{a} - {b} 等于多少？", str(a - b))
        for a in range(30, 60)
        for b in range(2, 16)
    ]
    multiplication = [
        (f"{a} 乘以 {b} 等于多少？", str(a * b))
        for a in range(2, 11)
        for b in range(2, 11)
    ]
    return [
        ("capitals", 34, _shuffled([(f"{country}的首都是哪里？", city) for country, city in ZH_CAPITALS], "zh:cap")),
        ("math_addition", 45, _shuffled(addition, "zh:add")),
        ("math_subtraction", 22, _shuffled(subtraction, "zh:sub")),
        ("math_multiplication", 30, _shuffled(multiplication, "zh:mul")),
        ("opposites", 30, _shuffled([(f"“{word}”的反义词是什么？", opp) for word, opp in ZH_OPPOSITES], "zh:opp")),
        ("animal_legs", 20, _shuffled([(f"{animal}有几条腿？", str(legs)) for animal, legs in ZH_ANIMAL_LEGS.items()], "zh:legs")),
        ("animal_sounds", 20, _shuffled([(f"{animal}怎么叫？", sound) for animal, sound in ZH_ANIMAL_SOUNDS.items()], "zh:sound")),
        ("days_months", 19, _shuffled(
            [(f"星期{day}之后是星期几？", f"星期{ZH_WEEK[(i + 1) % 7]}") for i, day in enumerate(ZH_WEEK)]
            + [(f"{month}之后是几月？", f"{ZH_MONTH[(i + 1) % 12]}月") for i, month in enumerate(ZH_MONTH)],
            "zh:day",
        )),
        ("char_count", 35, _shuffled([(f"“{word}”这个词有几个汉字？", str(len(word))) for word in ZH_CHAR_WORDS], "zh:char")),
        ("next_number", 20, _shuffled([(f"{n}之后是多少？", str(n + 1)) for n in range(9, 209, 10)], "zh:next")),
        ("colors", 8, _shuffled([(f"{one}和{two}混合是什么颜色？", mixed) for one, two, mixed in ZH_COLORS], "zh:color")),
        ("greetings", 27, _shuffled(ZH_GREETINGS, "zh:greet")),
        ("festivals", 22, _shuffled(ZH_FESTIVALS, "zh:fest")),
        ("seasons", 20, _shuffled(ZH_SEASONS, "zh:season")),
        ("facts", 20, _shuffled(ZH_FACTS, "zh:fact")),
        ("time", 25, _shuffled(ZH_TIME, "zh:time")),
        ("shapes", 16, _shuffled(
            [(f"{shape}有几条边？", str(sides)) for shape, sides in ZH_SHAPES]
            + [(f"有几条边的形状是{shape}？", shape) for shape, sides in ZH_SHAPES
               if sum(1 for _s, n in ZH_SHAPES if n == sides) == 1]
            + [
                ("圆形有几条边？", "0条"),
                ("什么形状是圆的？", "圆形"),
                ("什么形状有四条相等的边？", "正方形"),
            ],
            "zh:shape",
        )),
        ("quantifiers", 27, _shuffled([(f"“{noun}”的量词是什么？", unit) for noun, unit in ZH_QUANTIFIERS], "zh:quant")),
        ("cities", 18, _shuffled(ZH_CITIES, "zh:city")),
        ("units", 22, _shuffled(ZH_UNITS, "zh:unit")),
    ]


ZH_WEEK = ("一", "二", "三", "四", "五", "六", "日")
ZH_MONTH = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二")

ZH_EDGE_EVAL = [
    ("？", "这不是一个完整的问题。"),
    ("！！！！！", "这不是一个完整的问题。"),
    ("人生的意义是什么？", "没有唯一答案。"),
    ("1234567890", "这是一个数字，不是问题。"),
    ("跟我重复一遍：你好", "你好"),
    ("2 + 2 + 2 等于多少？", "6"),
    ("讲个笑话。", "为什么鸡要过马路？为了到对面去。"),
    ("Hello", "这个演示数据集是中文的。"),
    ("亚特兰蒂斯的首都是哪里？", "亚特兰蒂斯不是真实存在的国家。"),
    ("“饕餮”这两个字怎么写？", "这是一个很少见的词。"),
]

ZH_UNRELATED_EVAL = [
    ("用一句话解释万有引力。", "万有引力让物体互相吸引。"),
    ("《红楼梦》的作者是谁？", "曹雪芹。"),
    ("水在多少华氏度沸腾？", "212华氏度。"),
    ("把“你好”翻译成英文。", "hello"),
    ("一小时有多少分钟？", "60分钟"),
    ("晴朗的天空是什么颜色？", "蓝色"),
    ("说出三个行星的名字。", "水星、金星、地球。"),
    ("81的平方根是多少？", "9"),
    ("100除以4等于多少？", "25"),
    ("巴西说什么语言？", "葡萄牙语。"),
]


# ======================================================================
# 组装与写出
# ======================================================================
def build_sft(language: str) -> list[dict]:
    """生成 500 条 SFT 记录（english / chinese）。"""
    if language == "english":
        families, source, chinese = _en_families(), "demo_sft_en", False
    elif language == "chinese":
        families, source, chinese = _zh_families(), "demo_sft_zh", True
    else:  # pragma: no cover - 内部调用保证
        raise ValueError(f"未知语言：{language}")
    seen: set[str] = set()
    records = _take_families(families, source, seen)
    _fill_math(records, seen, SFT_TOTAL, source, chinese)
    if len(records) != SFT_TOTAL:
        raise RuntimeError(f"{language} SFT 记录数不是 {SFT_TOTAL}：{len(records)}")
    return records


def build_eval(language: str) -> list[dict]:
    """生成 30 条评估记录（10 in_distribution + 10 edge + 10 unrelated）。"""
    if language == "english":
        sft_source, eval_source, chinese = "demo_sft_en", "demo_eval_en", False
        base_families = _en_families()
        edge, unrelated = EN_EDGE_EVAL, EN_UNRELATED_EVAL
    elif language == "chinese":
        sft_source, eval_source, chinese = "demo_sft_zh", "demo_eval_zh", True
        base_families = _zh_families()
        edge, unrelated = ZH_EDGE_EVAL, ZH_UNRELATED_EVAL
    else:  # pragma: no cover
        raise ValueError(f"未知语言：{language}")

    sft_questions = {record["q"].strip().casefold() for record in build_sft(language)}
    # in_distribution：同分布但未出现在 SFT 的候选，跨类别轮转均衡采样，
    # 避免评测集几乎全部落在第一个有余量的类别（如算术题）。
    target = EVAL_TOTAL // 3
    pools: list[list[tuple[str, str]]] = []
    for _category, _count, candidates in base_families:
        pool = [
            (question, answer)
            for question, answer in candidates
            if question.strip().casefold() not in sft_questions
        ]
        if pool:
            pools.append(pool)
    in_dist: list[tuple[str, str]] = []
    cursor = 0
    while len(in_dist) < target and pools:
        progressed = False
        for pool in pools:
            if cursor < len(pool):
                in_dist.append(pool[cursor])
                progressed = True
                if len(in_dist) >= target:
                    break
        if not progressed:
            break
        cursor += 1
    if len(in_dist) < target:
        raise RuntimeError(f"{language} 评估 in_distribution 候选不足：{len(in_dist)}")

    records = [_make(q, a, "in_distribution", eval_source) for q, a in in_dist]
    records += [_make(q, a, "edge", eval_source) for q, a in edge]
    records += [_make(q, a, "unrelated", eval_source) for q, a in unrelated]
    if len(records) != EVAL_TOTAL:
        raise RuntimeError(f"{language} 评估记录数不是 {EVAL_TOTAL}：{len(records)}")
    # 边缘/无关问题不应与 SFT 训练问题完全相同。
    for record in records:
        if record["q"].strip().casefold() in sft_questions:
            raise RuntimeError(f"评估问题与 SFT 训练问题重叠：{record['q']!r}")
    return records


def _write_jsonl(path: Path, records: list[dict]) -> str:
    """按 ``\\n`` 换行写出 UTF-8 JSONL（中文不转义），返回文件 sha256。

    显式 ``newline="\\n"`` 是为了让 Windows 上的输出与 Linux 完全一致，
    保证「同一条命令重复执行得到相同字节」。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out_dir: Path) -> dict[str, dict]:
    """生成四个文件并返回统计信息（路径 -> {count, sha256}）。"""
    reports: dict[str, dict] = {}
    for language, sft_name, eval_name in (
        ("english", "demo_sft_en.jsonl", "demo_eval_en.jsonl"),
        ("chinese", "demo_sft_zh.jsonl", "demo_eval_zh.jsonl"),
    ):
        sft_records = build_sft(language)
        eval_records = build_eval(language)
        for name, records in ((sft_name, sft_records), (eval_name, eval_records)):
            path = out_dir / name
            digest = _write_jsonl(path, records)
            reports[name] = {"count": len(records), "sha256": digest, "path": str(path)}
    return reports


def main(argv: list[str] | None = None) -> int:
    """命令行入口：把四个演示数据文件生成到 ``--out-dir``。"""
    parser = argparse.ArgumentParser(description="生成 MiniGPT 确定性演示 SFT/评估数据（纯标准库）")
    parser.add_argument("--out-dir", default=str(ROOT / "data"), help="输出目录（默认项目 data/）")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    reports = generate(out_dir)
    for name, info in reports.items():
        print(f"[build_demo_data] {name}: {info['count']} 条  sha256={info['sha256'][:16]}...  -> {info['path']}")
    print("[build_demo_data] 完成（确定性生成：同一条命令重复执行得到相同文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
