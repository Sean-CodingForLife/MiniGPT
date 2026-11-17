"""Web 控制台「表单 / JSON 双模式」配置编辑器的静态结构守卫。

只读取 web/ 下的静态资源做文本断言，不启动服务；守住表单与 JSON 编辑器
两个入口同时存在、默认隐藏 JSON 编辑器、以及每个配置项都有对应表单控件，
防止新增配置项后前端出现「无法编辑/显示 undefined」的回归。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FormUiStaticTests(unittest.TestCase):
    """校验前端配置编辑器在 HTML/JS/CSS 三处的挂载点保持完整。"""

    def test_form_and_json_modes_are_wired(self) -> None:
        """表单与 JSON 编辑器都必须接线，且编辑器默认隐藏、样式类不得缺失。"""
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        for marker in ("id=\"config-form\"", "id=\"editor-form-btn\"", "id=\"editor-json-btn\"", "id=\"config-editor\""):
            self.assertIn(marker, html)
        self.assertRegex(html, r'id="config-editor"[^>]*hidden')
        for marker in ("FIELD_GROUPS", "FIELD_META", "setEditorMode", "renderConfigForm", "onFormChanged"):
            self.assertIn(marker, app)
        for marker in (".config-form", ".form-grid", ".form-field", ".editor-toolbar"):
            self.assertIn(marker, css)

    def test_every_config_field_has_a_form_entry(self) -> None:
        """config.example.json 中的每个键都必须同时出现在表单分组与字段标签中，避免新增配置漏配 UI。"""
        import json

        config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        app = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        # 直接从 JS 源码里截取 FIELD_GROUPS / FIELD_META 两段声明做键名比对，
        # 省去引入浏览器或解析 JS 的开销。
        groups_block = app.split("var FIELD_GROUPS = [", 1)[1].split("  var FIELD_META =", 1)[0]
        meta_block = app.split("var FIELD_META = {", 1)[1].split("\n  };", 1)[0]
        group_fields = set(re.findall(r'"([a-z0-9_]+)"', groups_block))
        meta_keys = set(re.findall(r"^    ([a-z0-9_]+): \{", meta_block, re.M))

        self.assertEqual(set(config) - group_fields, set(), "config fields missing from form groups")
        self.assertEqual(set(config) - meta_keys, set(), "config fields missing form labels")


if __name__ == "__main__":
    unittest.main()
