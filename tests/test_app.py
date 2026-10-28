"""End-to-end: train in quick mode on the SYNTHETIC fixture inside a temporary
project root, then start every website page and run the demo prediction flow.
(~1-2 minutes; skip with  pytest -m "not slow".)"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_training_script_and_website_end_to_end(synthetic_dir, tmp_path):
    env = {**os.environ, "NETGUARD_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "src.train", "--data", str(synthetic_dir), "--quick",
                        "--tune-rows", "2000", "--skip-holdout"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    for rel in ["models/final_pipeline.joblib", "models/model_metadata.json",
                "models/feature_schema.json", "reports/results/model_comparison.json",
                "data/samples/demo_traffic_sample.csv"]:
        assert (tmp_path / rel).exists(), rel

    script = textwrap.dedent(f"""
        import sys
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(r"{ROOT / 'app' / 'app.py'}", default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        print("home ok")
    """)
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]

    # Every page function runs without raising, including the demo prediction
    page_script = textwrap.dedent(f"""
        import sys, types
        sys.path.insert(0, r"{ROOT / 'app'}")
        from streamlit.testing.v1 import AppTest

        def run_page(name):
            src = open(r"{ROOT / 'app' / 'app.py'}", encoding="utf-8").read()
            src = src.replace("st.navigation(pages).run()", name + "()")
            at = AppTest.from_string(src, default_timeout=120)
            at.run()
            assert not at.exception, (name, at.exception)
            return at

        for n in ["page_home", "page_dataset", "page_performance", "page_explain",
                  "page_method", "page_about"]:
            run_page(n)
        at = run_page("page_detect")
        demo_btn = [b for b in at.button if b.label == "Classify the demo sample"][0]
        demo_btn.click().run()
        assert not at.exception, at.exception
        assert any("flagged as attack" in m.value for m in at.markdown), "verdict not shown"
        print("pages ok")
    """)
    r = subprocess.run([sys.executable, "-c", page_script], cwd=ROOT / "app", env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
