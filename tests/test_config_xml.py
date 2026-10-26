"""config.xml 配置加载 / 序列化测试.

覆盖:
    1. 仓库根目录 config.xml 模板解析无异常
    3. 类型转换: str / int / float / bool / optional_str / list_str
    4. 缺字段时沿用 dataclass 默认
    5. to_xml → from_xml 圆环回 (round-trip) 一致性
    6. 未知字段跳过不抛错
    7. 解析失败 (非法类型) 抛 ValueError 并指明字段名

运行: python tests/test_config_xml.py
"""
import os, tempfile, traceback, sys

# 允许在仓库根目录直接 python tests/test_config_xml.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgf.config import LGFConfig


PASS = 0; FAIL = 0; FAILED = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def test_repo_template_loads():
    """仓库根目录的 config.xml 应该能加载出非空 LLM 配置."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "config.xml")
    cfg = LGFConfig.from_xml(path)
    check("repo-config-loads-backend",
          cfg.llm_backend == "heuristic",
          f"got {cfg.llm_backend!r}")
    check("repo-config-loads-url-None",
          cfg.llm_url is None,
          f"got {cfg.llm_url!r}")
    check("repo-config-loads-debug-false",
          cfg.llm_debug is False,
          f"got {cfg.llm_debug!r}")
    check("repo-config-loads-temperature",
          cfg.llm_temperature == 0.7,
          f"got {cfg.llm_temperature!r}")
    check("repo-config-loads-population",
          cfg.population == 50,
          f"got {cfg.population!r}")


def test_type_coercion():
    """各 kind 类型转换符合预期."""
    xml = """<lgf>
        <llm_backend>openai</llm_backend>
        <llm_url>https://api.openai.com/v1</llm_url>
        <llm_api_key>sk-test-123</llm_api_key>
        <llm_model>gpt-4o-mini</llm_model>
        <llm_temperature>0.3</llm_temperature>
        <llm_timeout>120.0</llm_timeout>
        <llm_debug>true</llm_debug>
        <population>80</population>
        <generations>10</generations>
        <case_names>a, b, d</case_names>
    </lgf>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml); path = f.name
    try:
        cfg = LGFConfig.from_xml(path)
        check("type-str-backend", cfg.llm_backend == "openai")
        check("type-str-url",      cfg.llm_url == "https://api.openai.com/v1")
        check("type-str-api_key",  cfg.llm_api_key == "sk-test-123")
        check("type-float-temp",   cfg.llm_temperature == 0.3)
        check("type-float-timeout", cfg.llm_timeout == 120.0)
        check("type-bool-true",    cfg.llm_debug is True)
        check("type-int-pop",      cfg.population == 80)
        check("type-int-gen",      cfg.generations == 10)
        check("type-list-str",     cfg.case_names == ["a", "b", "d"])
    finally:
        os.unlink(path)


def test_missing_fields_use_defaults():
    """XML 中缺失的字段, 沿用 dataclass 默认值 (heuristic / None / False / 0.7 / 60)."""
    xml = "<lgf><llm_url>http://x/v1</llm_url></lgf>"
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml); path = f.name
    try:
        cfg = LGFConfig.from_xml(path)
        check("default-backend",     cfg.llm_backend == "heuristic")
        check("default-api-key",     cfg.llm_api_key is None)
        check("default-temperature", cfg.llm_temperature == 0.7)
        check("default-timeout",     cfg.llm_timeout == 60.0)
        check("default-debug-false", cfg.llm_debug is False)
        check("default-population",  cfg.population == 50)
        check("default-use_llm",     cfg.use_llm is True)
    finally:
        os.unlink(path)


def test_round_trip():
    """cfg → to_xml → from_xml → cfg 字段一致."""
    cfg0 = LGFConfig(
        llm_backend="openai",
        llm_url="https://api.openai.com/v1",
        llm_api_key="sk-abc",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        llm_timeout=90.0,
        llm_debug=True,
        population=120,
        generations=50,
        case_names=["alpha", "beta"],
        seed=42,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        path = f.name
    try:
        cfg0.to_xml(path)
        cfg1 = LGFConfig.from_xml(path)
        check("roundtrip-backend",  cfg1.llm_backend == cfg0.llm_backend)
        check("roundtrip-url",      cfg1.llm_url == cfg0.llm_url)
        check("roundtrip-api-key",  cfg1.llm_api_key == cfg0.llm_api_key)
        check("roundtrip-model",    cfg1.llm_model == cfg0.llm_model)
        check("roundtrip-temp",     cfg1.llm_temperature == cfg0.llm_temperature)
        check("roundtrip-timeout",  cfg1.llm_timeout == cfg0.llm_timeout)
        check("roundtrip-debug",    cfg1.llm_debug == cfg0.llm_debug)
        check("roundtrip-pop",      cfg1.population == cfg0.population)
        check("roundtrip-gen",      cfg1.generations == cfg0.generations)
        check("roundtrip-case",     cfg1.case_names == cfg0.case_names)
        check("roundtrip-seed",     cfg1.seed == cfg0.seed)
    finally:
        os.unlink(path)


def test_unknown_field_skipped():
    """未知字段跳过不抛错 (向前兼容)."""
    xml = """<lgf>
        <llm_backend>openai</llm_backend>
        <future_field_xxx>some value</future_field_xxx>
    </lgf>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml); path = f.name
    try:
        cfg = LGFConfig.from_xml(path)
        check("unknown-skip-backend", cfg.llm_backend == "openai")
        check("unknown-skip-no-attr", not hasattr(cfg, "future_field_xxx"))
    finally:
        os.unlink(path)


def test_invalid_type_raises():
    """非法类型应抛 ValueError 并指明字段."""
    xml = "<lgf><llm_temperature>not_a_number</llm_temperature></lgf>"
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml); path = f.name
    try:
        raised = False
        try:
            LGFConfig.from_xml(path)
        except ValueError as e:
            raised = "llm_temperature" in str(e)
        check("invalid-type-raises", raised)
    finally:
        os.unlink(path)


def test_wrong_root_raises():
    """非 <lgf> 根节点抛 ValueError."""
    xml = "<config><llm_backend>openai</llm_backend></config>"
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml); path = f.name
    try:
        raised = False
        try:
            LGFConfig.from_xml(path)
        except ValueError as e:
            raised = "lgf" in str(e).lower()
        check("wrong-root-raises", raised)
    finally:
        os.unlink(path)


def test_file_not_found():
    """不存在的路径抛 FileNotFoundError."""
    raised = False
    try:
        LGFConfig.from_xml("Z:/nonexistent/path/xxx.xml")
    except FileNotFoundError:
        raised = True
    check("missing-file-raises", raised)


def main():
    global PASS, FAIL, FAILED
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n[TEST] {fn.__name__}")
        try:
            fn()
        except Exception:
            FAIL += 1
            FAILED.append(fn.__name__)
            traceback.print_exc()
    print(f"\n=== summary: {PASS} passed, {FAIL} failed ===")
    if FAIL:
        for n in FAILED: print("  FAILED:", n)
        sys.exit(1)


if __name__ == "__main__":
    main()