"""dictionary.txt (word phonemes...) -> SQLite (word, rank, phonemes), file order kept as rank."""
import json, sqlite3, sys
src, cfg_path, dst = sys.argv[1:4]
valid = set(json.load(open(cfg_path))["phoneme_id_map"])
db = sqlite3.connect(dst)
db.execute("CREATE TABLE d (word TEXT NOT NULL, rank INTEGER NOT NULL, ph TEXT NOT NULL)")
rows, ranks, bad = [], {}, 0
with open(src, encoding="utf-8") as f:
    for line in f:
        it = line.split()
        if len(it) < 2:
            continue
        if any(p not in valid for p in it[1:]):
            bad += 1; continue
        r = ranks.get(it[0], 0); ranks[it[0]] = r + 1
        rows.append((it[0], r, " ".join(it[1:])))
db.executemany("INSERT INTO d VALUES (?,?,?)", rows)
db.execute("CREATE INDEX i ON d(word, rank)")
db.commit()
multi = sum(1 for v in ranks.values() if v > 1)
print(f"words={len(ranks)} rows={len(rows)} homographs={multi} skipped_invalid={bad}")
db.execute("VACUUM"); db.close()
