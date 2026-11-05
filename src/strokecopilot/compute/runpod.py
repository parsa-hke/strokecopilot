"""RunPod client with the same interface as vast.Vast: the DEC-21 fallback (Community Cloud, RTX 4090).

Thin on purpose: used only if no Vast offer passes the filters under 0.50 USD/h. GraphQL at api.runpod.io; the key
comes from $RUNPOD_API_KEY only. Tested against a stub; not yet run against RunPod itself.
"""

import json
import os
import time
import urllib.error
import urllib.request

URL = "https://api.runpod.io/graphql"
GPU = "NVIDIA GeForce RTX 4090"


class RunPodError(RuntimeError):
    pass


def _http(query, key, variables=None, timeout=60):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RunPodError(f"HTTP {e.code}: {e.read()[:300].decode(errors='ignore')}") from None
    if out.get("errors"):
        raise RunPodError(str(out["errors"])[:300])
    return out["data"]


class RunPod:
    name = "runpod"

    def __init__(self, key=None, http=_http, sleep=time.sleep, clock=time.monotonic):
        self.key = key or os.environ.get("RUNPOD_API_KEY")
        if not self.key:
            raise RunPodError("RUNPOD_API_KEY is not set in this shell.")
        self.http, self.sleep, self.clock = http, sleep, clock

    def balance(self):
        return float(self.http("query { myself { clientBalance } }", self.key)["myself"]["clientBalance"])

    def offers(self, filters=None, limit=10):
        q = 'query { gpuTypes(input: {id: "%s"}) { id displayName memoryInGb lowestPrice(input: {gpuCount: 1, secureCloud: false}) { minimumBidPrice uninterruptablePrice } } }' % GPU
        types = self.http(q, self.key)["gpuTypes"]
        out = []
        for t in types:
            price = (t.get("lowestPrice") or {}).get("uninterruptablePrice")
            if price is not None and price <= 0.50:
                out.append({"id": t["id"], "dph_total": price, "gpu_name": t["displayName"], "cpu_cores_effective": None,
                            "cpu_ram": None, "disk_space": None, "inet_down": None, "reliability": None, "geolocation": "community"})
        return out[:limit]

    def create(self, offer_id, image, disk_gb, label, onstart=None):
        m = ('mutation($i: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $i) { id } }')
        variables = {"i": {"cloudType": "COMMUNITY", "gpuCount": 1, "gpuTypeId": offer_id, "imageName": image,
                           "containerDiskInGb": disk_gb, "volumeInGb": 0, "name": label, "ports": "22/tcp",
                           "minVcpuCount": 16, "minMemoryInGb": 32, "dockerArgs": onstart or ""}}
        return self.http(m, self.key, variables)["podFindAndDeployOnDemand"]["id"]

    def instances(self):
        q = "query { myself { pods { id name desiredStatus costPerHr runtime { ports { ip isIpPublic privatePort publicPort } } } } }"
        return [{"id": p["id"], "label": p["name"], "actual_status": "running" if p.get("runtime") else (p.get("desiredStatus") or "").lower(),
                 "dph_total": p.get("costPerHr"), "_ports": (p.get("runtime") or {}).get("ports") or []}
                for p in self.http(q, self.key)["myself"]["pods"]]

    def get(self, instance_id):
        return next((i for i in self.instances() if str(i["id"]) == str(instance_id)), None)

    def wait_running(self, instance_id, timeout_s=900, poll_s=15):
        start = self.clock()
        while True:
            inst = self.get(instance_id)
            if inst and inst["actual_status"] == "running" and self.ssh_or_none(inst):
                return inst
            if self.clock() - start > timeout_s:
                raise RunPodError(f"Pod {instance_id} not running after {timeout_s} s.")
            self.sleep(poll_s)

    @staticmethod
    def ssh_or_none(inst):
        for p in inst.get("_ports", []):
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                return p["ip"], int(p["publicPort"]), "root"
        return None

    def ssh(self, inst):
        found = self.ssh_or_none(inst)
        if not found:
            raise RunPodError("The pod has no public SSH port.")
        return found

    def destroy(self, instance_id):
        self.http('mutation($id: String!) { podTerminate(input: {podId: $id}) }', self.key, {"id": str(instance_id)})

    def describe(self, offer):
        return f"RunPod community {offer.get('gpu_name')}: ${offer.get('dph_total', 0):.3f}/h"
