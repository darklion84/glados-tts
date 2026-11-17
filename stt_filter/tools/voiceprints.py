#!/usr/bin/env python3
"""Образцы голосов для аддона stt_filter: посмотреть запросы, записать человека, проверить порог.

Работает с папкой аддона на общем диске HA (addon_configs/local_stt_filter):
  utterances.jsonl — отпечаток каждого запроса (пишет аддон);
  speakers.json    — образцы людей (пишет этот скрипт; аддон перечитывает сам, перезапуск не нужен).

  voiceprints.py list [-n 20]                      последние запросы: время, кто, сходство, текст
  voiceprints.py enroll ИМЯ --since 21:40 [--until 21:45] [--date 2026-11-17] [--yes]
                                                   запросы за это время → образец человека
  voiceprints.py forget ИМЯ                        убрать человека
  voiceprints.py report                            сходство «свой/чужой» по журналу — для выбора порога
"""

from __future__ import annotations

import argparse
import array
import base64
import json
import math
import os
import statistics
import sys
from datetime import date, datetime, time

DEFAULT_DIR = "/Volumes/addon_configs/local_stt_filter"


def vec(b64: str) -> list[float]:
    a = array.array("f")
    a.frombytes(base64.b64decode(b64))
    if sys.byteorder != "little":
        a.byteswap()
    return list(a)


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def centroid(vectors: list[list[float]]) -> list[float]:
    return unit([sum(col) / len(vectors) for col in zip(*(unit(v) for v in vectors))])


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def utterances(folder: str) -> list[dict]:
    rows = []
    for name in ("utterances.jsonl.1", "utterances.jsonl"):
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            pass
    return rows


def load_speakers(folder: str) -> dict:
    try:
        with open(os.path.join(folder, "speakers.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"model": None, "speakers": {}}


def save_speakers(folder: str, data: dict) -> None:
    path = os.path.join(folder, "speakers.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def local(row: dict) -> datetime:
    return datetime.fromisoformat(row["ts"]).astimezone()


def show(row: dict) -> str:
    scores = ", ".join(f"{k} {v:.2f}" for k, v in sorted(row["scores"].items(), key=lambda kv: -kv[1]))
    return (f"{local(row):%d.%m %H:%M:%S}  {row['seconds']:5.1f} с {row['rms_dbfs']:4d} дБ  "
            f"{row['speaker'] or '—':10s} {scores:24s} «{row['text']}»")


def cmd_list(a) -> None:
    for row in utterances(a.dir)[-a.n:]:
        print(show(row))


def cmd_enroll(a) -> None:
    day = date.fromisoformat(a.date) if a.date else date.today()
    since = datetime.combine(day, time.fromisoformat(a.since)).astimezone()
    until = datetime.combine(day, time.fromisoformat(a.until)).astimezone() if a.until else datetime.now().astimezone()
    picked = [r for r in utterances(a.dir) if since <= local(r) <= until and r["seconds"] >= a.min_seconds]
    if not picked:
        sys.exit("за это время запросов нет")
    models = {r["model"] for r in picked}
    data = load_speakers(a.dir)
    if len(models) != 1 or (data["speakers"] and data["model"] not in models):
        sys.exit(f"отпечатки сняты разными моделями: {models}, в speakers.json — {data['model']}")
    for row in picked:
        print(show(row))
    have = data["speakers"].setdefault(a.name, {"embeddings": []})["embeddings"]
    new = [r["embedding"] for r in picked if r["embedding"] not in have]
    print(f"\n{a.name}: было образцов {len(have)}, добавится {len(new)}")
    if not a.yes:
        print("ничего не записано; чтобы записать, повтори с --yes")
        return
    have += new
    data["model"] = models.pop()
    save_speakers(a.dir, data)
    print("записано в speakers.json")


def cmd_forget(a) -> None:
    data = load_speakers(a.dir)
    if data["speakers"].pop(a.name, None) is None:
        sys.exit(f"нет такого: {a.name}")
    save_speakers(a.dir, data)
    print(f"{a.name} убран")


def cmd_report(a) -> None:
    data = load_speakers(a.dir)
    rows = [r for r in utterances(a.dir) if r["model"] == data["model"]]
    for name, spk in data["speakers"].items():
        own_b64 = spk["embeddings"]
        own = [unit(vec(e)) for e in own_b64]
        # Свои фразы: каждая против образца из остальных, иначе сходство завышено.
        own_scores = [cos(own[i], centroid(own[:i] + own[i + 1:])) for i in range(len(own))] if len(own) > 1 else []
        c = centroid(own)
        others = sorted(((cos(unit(vec(r["embedding"])), c), r) for r in rows if r["embedding"] not in own_b64),
                        key=lambda t: -t[0])
        print(f"\n{name}: образцов {len(own)}")
        if own_scores:
            print(f"  свои фразы:   мин {min(own_scores):.2f}  медиана {statistics.median(own_scores):.2f}  макс {max(own_scores):.2f}")
        print(f"  прочие запросы в журнале ({len(others)}), от самых похожих:")
        for score, row in others[:a.n]:
            print(f"    {score:.2f}  {show(row)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.environ.get("VOICEPRINTS_DIR", DEFAULT_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("-n", type=int, default=20); s.set_defaults(fn=cmd_list)
    s = sub.add_parser("enroll"); s.add_argument("name"); s.add_argument("--since", required=True)
    s.add_argument("--until"); s.add_argument("--date"); s.add_argument("--min-seconds", type=float, default=2.0)
    s.add_argument("--yes", action="store_true"); s.set_defaults(fn=cmd_enroll)
    s = sub.add_parser("forget"); s.add_argument("name"); s.set_defaults(fn=cmd_forget)
    s = sub.add_parser("report"); s.add_argument("-n", type=int, default=15); s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
