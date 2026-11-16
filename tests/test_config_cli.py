import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from faceid.config import DEFAULTS, load_config
from faceid.cli import main, _open_browser_when_ready


class ConfigTests(unittest.TestCase):
    def test_relative_paths_resolve_from_configuration_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            path.write_text(json.dumps({"data_dir": "photos", "training": {"epochs": 2}}))
            config = load_config(path)
            self.assertEqual(Path(config["data_dir"]), Path(directory) / "photos")
            self.assertEqual(config["training"]["epochs"], 2)
            self.assertEqual(config["splits"], DEFAULTS["splits"])
            config["splits"]["train"].clear()
            self.assertEqual(DEFAULTS["splits"]["train"], [1, 2, 3, 4])

    def test_invalid_batch_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"training": {"batch_size": 0}}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_init_creates_18_folders_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"data_dir": "photos", "legacy_dir": "missing_weights"}))
            self.assertEqual(main(["--config", str(path), "init"]), 0)
            self.assertEqual(len(list((Path(directory) / "photos").glob("*/phien_*"))), 18)

    def test_browser_opens_only_after_server_can_answer_http(self):
        import socket
        import threading
        from urllib.request import urlopen
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/ready")
        def ready():
            return {"ready": True}

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}/ready"
            server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            statuses = []

            def browser_open(address):
                with urlopen(address, timeout=3) as response:
                    statuses.append(response.status)
                return True

            try:
                with patch("webbrowser.open", side_effect=browser_open) as opened:
                    thread.start()
                    _open_browser_when_ready(server, url)
                    opened.assert_called_once_with(url)
                self.assertEqual(statuses, [200])
            finally:
                server.should_exit = True
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
