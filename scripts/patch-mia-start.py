#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Desktop/Qwen3.8-27B-SGLang-DGX-Spark/start.sh")
text = p.read_text()
bak = p.with_name("start.sh.bak-pre-mia-20260819")
if not bak.exists():
    bak.write_text(text)
    print("backup", bak)

old = """# GDN state pool: S=4 (extra_buffer_lazy) + D=4 (MTP draft tokens) = 8
# slots per request; DSpark would be 12. Sized from MAX_CONCURRENT_REQUESTS.
MAMBA_SLOTS_PER_REQ=8
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))
"""
new = """# GDN state pool: slots = concurrency x S. S=4 for extra_buffer_lazy
# (Mia: engine divides the pool by S alone; x(S+D) over-provisions 2x).
# Pin container to Cortex-X925 (5-9,15-19); A725 little cores are 0-4,10-14.
MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-0}"
CPUSET="${CPUSET:-5-9,15-19}"
MAMBA_SLOTS_PER_REQ=$(( 4 - MAMBA_SKIP_DECODE_LOCK ))
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))
"""
if old not in text:
    raise SystemExit("GDN block not found")
text = text.replace(old, new, 1)

needle = "  --shm-size 32g \\\n  -e HF_HOME=/root/.cache/huggingface \\"
insert = (
    "  --shm-size 32g \\\n"
    '  "${PIN_ARGS[@]}" \\\n'
    "  -e HF_HOME=/root/.cache/huggingface \\\n"
    '  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK}" \\'
)
if needle not in text:
    raise SystemExit("docker env needle not found")
text = text.replace(needle, insert, 1)

old_run = "docker run -d \\\n"
new_run = 'PIN_ARGS=()\n[[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")\n\ndocker run -d \\\n'
if old_run not in text:
    raise SystemExit("docker run not found")
if "PIN_ARGS=()" not in text:
    text = text.replace(old_run, new_run, 1)

p.write_text(text)
print("patched", p)
