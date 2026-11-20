"""Run the n8n workflow against local fixture reports.

The structural tests always run. The execution tests need the n8n CLI (`npm install -g n8n`, or set N8N_BIN)
and are skipped without it. The email node is removed before executing, since tests have no SMTP account;
the tests check the alert the flow would send.

  python -m unittest discover -s automation/n8n -v
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKFLOW = json.loads((HERE / "model-health-alert.json").read_text())
N8N = os.environ.get("N8N_BIN") or shutil.which("n8n")


def node(name: str) -> dict:
    return next(n for n in WORKFLOW["nodes"] if n["name"] == name)


class WorkflowShapeTest(unittest.TestCase):
    def test_every_connection_points_at_a_node(self):
        names = {n["name"] for n in WORKFLOW["nodes"]}
        for source, outputs in WORKFLOW["connections"].items():
            self.assertIn(source, names)
            for branch in outputs["main"]:
                for target in branch:
                    self.assertIn(target["node"], names)

    def test_reads_the_report_the_weekly_job_publishes(self):
        url = node("Fetch health report")["parameters"]["url"]
        self.assertTrue(url.endswith("/main/output/monitoring_report.json"))
        monitor_job = (HERE.parent.parent / ".github/workflows/monitor.yml").read_text()
        self.assertIn("output/monitoring_report.json", monitor_job)

    def test_runs_weekly_on_monday_morning(self):
        rule = node("Every Monday 10:00")["parameters"]["rule"]["interval"][0]
        self.assertEqual((rule["field"], rule["triggerAtDay"], rule["triggerAtHour"]), ("weeks", [1], 10))

    def test_no_real_credentials_are_committed(self):
        self.assertEqual(node("Send email")["credentials"]["smtp"]["id"], "REPLACE_WITH_YOUR_SMTP_CREDENTIAL")


@unittest.skipUnless(N8N, "n8n CLI not installed")
class WorkflowRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handler = functools.partial(QuietHandler, directory=str(HERE / "fixtures"))
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.home = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.home, ignore_errors=True)

    def run_workflow(self, fixture: str) -> dict:
        flow = json.loads(json.dumps(WORKFLOW))
        flow["id"] = "test" + fixture.replace("_", "").replace(".json", "")
        flow["nodes"] = [n for n in flow["nodes"] if n["name"] != "Send email"]
        del flow["connections"]["Compose alert"]
        port = self.server.server_address[1]
        next(n for n in flow["nodes"] if n["name"] == "Fetch health report")["parameters"]["url"] = (
            f"http://127.0.0.1:{port}/{fixture}"
        )
        path = Path(self.home) / f"{flow['id']}.json"
        path.write_text(json.dumps(flow))

        env = {**os.environ, "N8N_USER_FOLDER": self.home, "N8N_DIAGNOSTICS_ENABLED": "false",
               "N8N_RUNNERS_ENABLED": "false"}
        subprocess.run([N8N, "import:workflow", f"--input={path}"], env=env, check=True, capture_output=True)
        out = subprocess.run([N8N, "execute", f"--id={flow['id']}", "--rawOutput"], env=env, check=True,
                             capture_output=True, text=True).stdout
        result = json.loads(out[out.index("{"):])
        self.assertEqual(result["status"], "success")
        return result["data"]["resultData"]

    def items(self, result: dict, name: str) -> list[dict]:
        return [i["json"] for i in result["runData"][name][0]["data"]["main"][0]]

    def test_alert_report_builds_an_email_of_flagged_batches(self):
        result = self.run_workflow("report_alert.json")
        self.assertEqual(result["lastNodeExecuted"], "Compose alert")
        self.assertEqual([b["batch"] for b in self.items(result, "Keep flagged batches")], ["mix_shift", "claims_trend"])
        [alert] = self.items(result, "Compose alert")
        self.assertEqual(alert["subject"], "Claims model: ALERT (2 batch(es) flagged)")
        self.assertIn("claims_trend", alert["html"])
        self.assertIn("1.204", alert["html"])
        self.assertNotIn("steady", alert["html"])

    def test_healthy_report_sends_nothing(self):
        result = self.run_workflow("report_ok.json")
        self.assertEqual(result["lastNodeExecuted"], "Healthy, no alert")
        self.assertNotIn("Compose alert", result["runData"])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
