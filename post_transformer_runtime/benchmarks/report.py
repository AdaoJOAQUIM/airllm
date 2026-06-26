"""Human-readable formatting for BenchmarkResult (no fabrication of n/a fields)."""

from __future__ import annotations

from .metrics import BenchmarkResult, human_bytes


def _fmt_float(x, unit="", nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def format_result(r: BenchmarkResult) -> str:
    host = r.host or {}
    lines = [
        "=" * 64,
        f"ENGINE        : {r.engine}",
        f"MODEL         : {r.model_id}",
        f"DEVICE        : {r.device}   COMPRESSION: {r.compression or 'none'}",
        f"HOST          : {host.get('platform', 'n/a')}",
        f"                torch={host.get('torch_version')} "
        f"cuda={host.get('cuda_available')} gpu={host.get('gpu_name')}",
        "-" * 64,
        f"prompt tokens : {r.prompt_tokens}    generated: {r.generated_tokens}",
        f"load time     : {_fmt_float(r.load_seconds, ' s')}",
        f"generate time : {_fmt_float(r.generate_seconds, ' s')}",
        f"throughput    : {_fmt_float(r.tokens_per_second, ' tok/s')}  "
        f"({_fmt_float(r.seconds_per_token, ' s/tok')})",
        "-" * 64,
        f"peak RAM      : {human_bytes(r.peak_rss_bytes)}",
        f"peak VRAM     : {human_bytes(r.peak_gpu_bytes)}",
        f"model on disk : {human_bytes(r.model_disk_bytes)}",
        f"splitted disk : {human_bytes(r.splitted_disk_bytes)}",
        "-" * 64,
        f"CPU energy    : {_fmt_float(r.cpu_energy_joules, ' J')}",
        f"GPU energy    : {_fmt_float(r.gpu_energy_joules, ' J')}",
        f"energy/token  : {_fmt_float(r.joules_per_token, ' J/tok')}",
        "-" * 64,
        f"perplexity    : {_fmt_float(r.perplexity, '', 4)}  "
        f"(eval tokens: {r.eval_text_tokens})",
    ]
    if r.notes:
        lines.append("-" * 64)
        for n in r.notes:
            lines.append(f"note: {n}")
    lines.append("=" * 64)
    return "\n".join(lines)


_MD_HEADER = (
    "| engine | model | device | comp | tok/s | s/tok | peak RAM | peak VRAM "
    "| disk | J/tok | ppl |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def markdown_header() -> str:
    return _MD_HEADER


def result_to_markdown(r: BenchmarkResult) -> str:
    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    return (
        f"| {r.engine} | {r.model_id} | {r.device} | {r.compression or '-'} "
        f"| {f(r.tokens_per_second)} | {f(r.seconds_per_token)} "
        f"| {human_bytes(r.peak_rss_bytes)} | {human_bytes(r.peak_gpu_bytes)} "
        f"| {human_bytes(r.splitted_disk_bytes or r.model_disk_bytes)} "
        f"| {f(r.joules_per_token)} | {f(r.perplexity)} |"
    )
