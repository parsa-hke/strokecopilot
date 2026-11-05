"""Vast.ai REST client: offers, create, wait, SSH details, destroy, balance (DEC-21).

Endpoints, checked against docs.vast.ai on 5 Nov 2026: POST /api/v0/bundles (search), PUT /api/v0/asks/{id} (create),
GET /api/v1/instances (list; /api/v0/instances/ answers 410 Gone), DELETE /api/v0/instances/{id} (destroy),
GET /api/v0/users/current/ (credit). The key comes from $VAST_API_KEY only and is never printed.
"""

import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://console.vast.ai/api"

# DEC-21: one on-demand RTX 4090 on a verified host (datacenter not required; amended 5 Nov 2026). Units as the REST API uses them: cpu_ram in MB,
# disk_space in GB, inet_down in Mbps (shown as Mbps in the console; the docs label it MB/s, checked live on 5 Nov).
FILTERS = {
    "gpu_name": {"eq": "RTX 4090"}, "num_gpus": {"eq": 1}, "verified": {"eq": True},
    "rentable": {"eq": True}, "rented": {"eq": False}, "reliability": {"gte": 0.98}, "cpu_cores_effective": {"gte": 16},
    "cpu_ram": {"gte": 32_000}, "disk_space": {"gte": 100}, "inet_down": {"gte": 200}, "dph_total": {"lte": 0.50},
    "cuda_max_good": {"gte": 12.8},  # the host driver must run the CUDA 12.8 image (pod.IMAGE)
}
OFFER_TYPE = "ondemand"


class VastError(RuntimeError):
    pass


def _http(method, url, key, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode(errors="ignore")
        raise VastError(f"{method} {url.split('/api', 1)[1]} -> HTTP {e.code}: {detail}") from None


class Vast:
    name = "vast"

    def __init__(self, key=None, http=_http, sleep=time.sleep, clock=time.monotonic):
        self.key = key or os.environ.get("VAST_API_KEY")
        if not self.key:
            raise VastError("VAST_API_KEY is not set in this shell.")
        self.http, self.sleep, self.clock = http, sleep, clock

    def _call(self, method, path, body=None, version="v0"):
        return self.http(method, f"{BASE}/{version}{path}", self.key, body)

    def balance(self):
        return float(self._call("GET", "/users/current/").get("credit") or 0.0)

    def offers(self, filters=None, limit=10):
        body = dict(filters or FILTERS)
        body.update({"type": OFFER_TYPE, "limit": limit, "order": [["dph_total", "asc"]]})
        found = self._call("POST", "/bundles/", body).get("offers", [])
        found = found if isinstance(found, list) else [found]
        return sorted(found, key=lambda o: o.get("dph_total", 1e9))

    def create(self, offer_id, image, disk_gb, label, onstart=None):
        body = {"image": image, "disk": disk_gb, "label": label, "runtype": "ssh_direct ssh_proxy"}
        if onstart:
            body["onstart"] = onstart
        out = self._call("PUT", f"/asks/{offer_id}/", body)
        if not out.get("success") or not out.get("new_contract"):
            raise VastError(f"Create was refused: {str(out)[:300]}")
        return int(out["new_contract"])

    def instances(self):
        return self._call("GET", "/instances?limit=25", version="v1").get("instances", [])

    def get(self, instance_id):
        return next((i for i in self.instances() if int(i.get("id", -1)) == int(instance_id)), None)

    def wait_running(self, instance_id, timeout_s=900, poll_s=15):
        """The instance once actual_status is running; raises if it ends up exited, offline or times out."""
        start = self.clock()
        while True:
            inst = self.get(instance_id)
            status = (inst or {}).get("actual_status")
            if status == "running":
                return inst
            if status in ("exited", "offline", "unknown") and self.clock() - start > 120:
                raise VastError(f"Instance {instance_id} is {status}; it will not reach running. Destroy it and retry.")
            if self.clock() - start > timeout_s:
                raise VastError(f"Instance {instance_id} not running after {timeout_s} s (last status {status}).")
            self.sleep(poll_s)

    @staticmethod
    def ssh(inst):
        """(host, port, user): the direct port when the host maps one, else Vast's SSH proxy."""
        ports = (inst.get("ports") or {}).get("22/tcp") or []
        if inst.get("public_ipaddr") and ports:
            return inst["public_ipaddr"].strip(), int(ports[0]["HostPort"]), "root"
        return inst["ssh_host"], int(inst["ssh_port"]), "root"

    def destroy(self, instance_id):
        self._call("DELETE", f"/instances/{instance_id}/")

    def describe(self, offer):
        return (f"offer {offer.get('id')}: ${offer.get('dph_total', 0):.3f}/h, {offer.get('gpu_name')}, "
                f"{offer.get('cpu_cores_effective', 0):.0f} CPU cores, {offer.get('cpu_ram', 0) / 1000:.0f} GB RAM, "
                f"{offer.get('disk_space', 0):.0f} GB disk, {offer.get('inet_down', 0):.0f} Mbps down, "
                f"reliability {offer.get('reliability', 0):.3f}, {offer.get('geolocation') or '?'}")
