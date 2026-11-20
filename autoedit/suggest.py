"""Step 2: a first draft of what to cut, plus the human-readable cut list.

The draft only catches the obvious things (empty islands, "um/uh", a sentence that is restarted in the next
few islands, a phrase repeated back-to-back inside one island). The AI assistant then reads the transcript,
fixes the draft with judgment (cuts.json), and shows the user 剪掉清單.md before anything is cut.

cuts.json:
  {"keep": [{"id": 0}, {"id": 5, "from": 3, "to": 12}, ...],   # from/to = word index in that island, `to` exclusive
   "why":  {"4": "重講", "5": "句中重講：最狠的是"}}
"""
import difflib
import re
from pathlib import Path

from .common import load, norm, save

FILLER = re.compile(r"^(嗯|哼|啊|呃|欸|誒|哦|喔|噢|唉|那個|就是|然後|好|對|ok|okay|um|uh|hmm)+$", re.I)


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _char_to_word(words):
    """For each normalized char position, the index of the word it came from."""
    idx = []
    for k, w in enumerate(words):
        idx += [k] * len(norm(w["w"]))
    return idx


def inner_repeat(words, min_len=3, max_len=10):
    """A phrase said twice back to back inside one island ('最狠的是最狠的是還能') -> (cut_from, cut_to) word
    indexes of the first take, or None. Only when both takes start on a word boundary."""
    n = "".join(norm(w["w"]) for w in words)
    c2w = _char_to_word(words)
    if len(c2w) != len(n):
        return None
    for L in range(max_len, min_len - 1, -1):
        for p in range(0, len(n) - 2 * L + 1):
            if n[p:p + L] == n[p + L:p + 2 * L]:
                w0, w1 = c2w[p], c2w[p + L]
                starts_ok = (p == 0 or c2w[p - 1] != w0) and c2w[p + L - 1] != w1
                if starts_ok and w1 > w0:
                    return w0, w1, n[p:p + L]
    return None


def draft(d):
    isl = load(d, "transcript.json")["islands"]
    N = [norm(s["text"]) for s in isl]
    keep, why = [], {}
    for i, s in enumerate(isl):
        n = N[i]
        if s.get("unclear"):
            why[str(i)] = "聽不清楚（可能是笑聲或雜音），先剪掉；如果是有講話的，請跟我說"
            continue
        if not s["words"] or not n:
            why[str(i)] = "沒有字（換氣、雜音）"
            continue
        if FILLER.match(n):
            why[str(i)] = f"語助詞：{s['text']}"
            continue
        restarted = None
        for j in range(i + 1, min(len(isl), i + 4)):
            if isl[j]["a"] - s["b"] > 20 or not N[j]:
                continue
            m = N[j]
            if len(n) <= len(m) + 2:
                # same opening as a later island = said again
                if len(n) >= 2 and _ratio(n, m[:len(n) + 1]) >= 0.72:
                    restarted = j
                    break
                # or this island sits (almost) entirely inside a later one
                if len(n) >= 4:
                    best = max((_ratio(n, m[k:k + len(n)]) for k in range(0, max(1, len(m) - len(n) + 1))), default=0)
                    if best >= 0.8:
                        restarted = j
                        break
        # short island that opens like the next one ('影片也先讓我們' -> '影片也能先排好…')
        if restarted is None and i + 1 < len(isl) and N[i + 1] and len(n) <= 8 and len(N[i + 1]) > len(n):
            common = 0
            while common < min(len(n), len(N[i + 1])) and n[common] == N[i + 1][common]:
                common += 1
            if common >= 3:
                restarted = i + 1
        if restarted is not None:
            why[str(i)] = f"重講（第 {restarted} 段又講了一次）"
            continue
        # a very short island that trails off ('你…有…') = false start
        if len(n) <= 4 and re.search(r"(\.\.\.|…)\s*$", s["text"]):
            why[str(i)] = f"講到一半停住：{s['text']}"
            continue
        # the island ends with the opening of the next island ('…還附…' + '還附一段免費 prompt')
        nxt = N[i + 1] if i + 1 < len(isl) else ""
        if nxt and len(s["words"]) > 2:
            c2w = _char_to_word(s["words"])
            for L in range(min(6, len(n) - 2), 1, -1):
                if len(c2w) == len(n) and n[-L:] == nxt[:L]:
                    w0 = c2w[len(n) - L]
                    if w0 > 0 and (len(n) - L == 0 or c2w[len(n) - L - 1] != w0):
                        keep.append({"id": i, "from": 0, "to": w0})
                        why[str(i)] = f"句尾先講了下一句的開頭「{n[-L:]}」，剪掉"
                        break
            else:
                L = 0
            if L:
                continue
        rep = inner_repeat(s["words"])
        if rep:
            w0, w1, phrase = rep
            if w0 > 0:
                keep.append({"id": i, "from": 0, "to": w0})
            keep.append({"id": i, "from": w1, "to": len(s["words"])})
            why[str(i)] = f"句中重講：「{phrase}」講了兩次，留第二次"
            continue
        keep.append({"id": i})
    save(d, "cuts.json", {"keep": keep, "why": why})
    write_list(d)


def write_list(d):
    """剪掉清單.md from transcript.json + cuts.json: everything that is NOT kept, in time order."""
    isl = load(d, "transcript.json")["islands"]
    cuts = load(d, "cuts.json")
    why = cuts.get("why", {})
    kept = {}
    for k in cuts["keep"]:
        s = isl[k["id"]]
        kept.setdefault(k["id"], set()).update(range(k.get("from", 0), k.get("to", len(s["words"]))))
    rows, total_cut = [], 0.0
    for s in isl:
        ks = kept.get(s["id"])
        if not ks:
            total_cut += s["b"] - s["a"]
            if s["text"] or s.get("unclear"):
                rows.append(f"| {s['a']:.1f}–{s['b']:.1f} | {s['id']} | {s['text'][:40]} | {why.get(str(s['id']), '剪掉')} |")
            continue
        if str(s["id"]) in why and len(ks) == len(s["words"]):
            rows.append(f"| {s['a']:.1f}–{s['b']:.1f} | {s['id']}（留著） | {s['text'][:40]} | ⚠️ {why[str(s['id'])]} |")
            continue
        gone = [k for k in range(len(s["words"])) if k not in ks]
        if gone:
            txt = "".join(s["words"][k]["w"] for k in gone)
            t0 = s["words"][gone[0]]["t0"]
            rows.append(f"| {t0:.1f} | {s['id']}（部分） | {txt[:40]} | {why.get(str(s['id']), '剪掉這幾個字')} |")
    src_len = isl[-1]["b"] if isl else 0
    kept_len = sum(s["b"] - s["a"] for s in isl if s["id"] in kept)
    md = ["# 剪掉清單", "",
          f"- 原片有聲音的部分約 {src_len / 60:.1f} 分鐘；留下 {len([1 for s in isl if s['id'] in kept])} 段、約 {kept_len / 60:.1f} 分鐘（還沒算停頓縮短和加速）。",
          "- 下面是會被剪掉的內容。沒有字的換氣、雜音片段不列出來。",
          "- 請確認：有沒有哪一句其實是要留的？有沒有漏剪的重講？", "",
          "| 原片秒數 | 段落 | 剪掉的內容 | 原因 |", "|---|---|---|---|", *rows]
    (Path(d) / "剪掉清單.md").write_text("\n".join(md), encoding="utf-8")
    print(f"剪掉清單.md：列出 {len(rows)} 處")
