"""Download only the pinned public RNN assets, verifying Git blob and SHA256 hashes.

No downloaded program is executed. Public checkpoints should be loaded with
torch.load(weights_only=True). The binary assets remain outside Git history.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"docs/research/public_internal_pgc_sources.json"
DEST=ROOT/"results/ring_public_assets"


def download(item,repository,commit):
    path=(DEST/item["path"]).resolve()
    if not path.is_relative_to(DEST.resolve()): raise ValueError("Path outside asset folder")
    path.parent.mkdir(parents=True,exist_ok=True)
    expected=item["size"]
    if path.exists():
        data=path.read_bytes()
    else:
        url=f"https://raw.githubusercontent.com/{repository}/{commit}/"+urllib.parse.quote(item["path"],safe="/")
        request=urllib.request.Request(url,headers={"User-Agent":"predictive-grid-mechanism-research"})
        with urllib.request.urlopen(request,timeout=120) as response:
            data=response.read(expected+1)
    if len(data)!=expected: raise ValueError(f"Size differs: {item['path']}")
    blob=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
    if blob!=item["git_blob_sha1"]: raise ValueError(f"Git blob hash differs: {item['path']}")
    if not path.exists():
        temp=path.with_suffix(path.suffix+".part")
        temp.write_bytes(data)
        temp.replace(path)
    result={**item,"sha256":hashlib.sha256(data).hexdigest(),"local_path":str(path.relative_to(ROOT))}
    if path.suffix==".pth": print(json.dumps({"event":"verified_checkpoint","path":item["path"]}),flush=True)
    return result


def main():
    manifest=json.loads(MANIFEST.read_text())
    if sum(row["size"] for row in manifest["files"])>1024**3: raise ValueError("Pinned assets exceed1GiB")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda item:download(item,manifest["repository"],manifest["commit"]),manifest["files"]))
    manifest["files"]=results
    manifest["download_verified"]=True
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"event":"complete","files":len(results),"bytes":sum(r["size"] for r in results)}),flush=True)


if __name__=="__main__": main()
