from typing import Any
import json
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


class WorkerApiClient:
    def __init__(self, base_url: str, api_token: str | None = None) -> None:
        self.base_url = base_url
        self.api_token = api_token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8")
                data = json.loads(content) if content else {}
                return data if isinstance(data, dict) else {}
        except HTTPError as error:
            message = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {error.code}: {message}") from error
        except URLError as error:
            raise RuntimeError(f"Network error: {error}") from error

    def claim(self, worker_payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/tasks/claim", worker_payload)

    def start(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/tasks/{task_id}/start", payload)

    def heartbeat(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/tasks/{task_id}/heartbeat", payload)

    def complete(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/tasks/{task_id}/complete", payload)

    def fail(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/tasks/{task_id}/fail", payload)

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/workers/register", payload)
