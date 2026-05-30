"""
情绪动作标签的剥离与锚点定位。

LLM 生成的文本中可以包含 ``[happy] / [shy] / [apologize] / [scared]`` 四种
方括号标签。TTS 合成前必须先把它们从文本里剥掉（否则会被读出来），但同时
要记下每个标签的"锚点"——即在剥离后的纯文本里的字符偏移——以便后面把它们
重新挂到对应的 sentence 上、与音频包顺序对齐。

设计要点：

* 仅 4 个情绪 key 命中白名单；其它方括号片段（路名/缩写等）原样保留，
  避免误伤。
* 锚点是"在剥离后的 clean_text 里、它原来所在位置之前已有的字符数"。这样
  后续断句不需要关心标签曾经存在过。
* 模块对外暴露纯函数，方便在 TTS handler 与单元测试里复用，不依赖项目内
  其它运行时对象。

未来扩展运动标签（mf/mb/ml/mr）时，可以在 ``_TAG_RE`` 命中后基于 key 走两
条分支：情绪 key 走 emotion 通道；运动 key 解析参数。本次只实现情绪。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


EMOTION_TAGS = ("happy", "shy", "apologize", "scared")
WHITELIST = frozenset(EMOTION_TAGS)


@dataclass(frozen=True)
class AnchoredTag:
    """剥离后保留的标签信息。

    Attributes:
        name: 标签 key，已规范化为小写，例如 ``"happy"``。
        offset: 在 ``clean_text`` 中的字符偏移。``offset == len(clean_text)``
            表示标签出现在末尾、其后再无任何字符；这种 tag 通常作为
            ``tail_tags`` 派发到 ``avatar_speech_end`` 的音频包上。
    """
    name: str
    offset: int


# 允许 key 内出现下划线，为将来留扩展空间；但目前只接受 ASCII 字母。
_TAG_RE = re.compile(r"\[\s*([a-zA-Z_]+)\s*\]")

# 常见「中式」书名号/全角方括号 → ASCII，便于模型输出 【happy】 时仍能识别。
_BRACKET_TRANS = str.maketrans(
    {
        "\u3010": "[",  # LEFT BLACK LENTICULAR BRACKET 【
        "\u3011": "]",  # RIGHT BLACK LENTICULAR BRACKET 】
        "\uff3b": "[",  # FULLWIDTH LEFT SQUARE BRACKET ［
        "\uff3d": "]",  # FULLWIDTH RIGHT SQUARE BRACKET ］
    }
)

# clean_text 中的合并空白：标签剥离后可能留下双空格或"空格 标点"，做轻量
# 折叠以避免 TTS 合成出现奇怪停顿。中文场景空格本来就少，不需要更激进。
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def extract_action_tags(text: str) -> Tuple[str, List[AnchoredTag]]:
    """从 ``text`` 中剥离白名单情绪标签，返回 ``(clean_text, anchored_tags)``。

    非白名单的方括号片段（例如 ``[A1]`` 路名缩写、或未来未实现的运动标签）
    原样保留进 ``clean_text``，但会被 TTS 朗读，应该靠 system prompt 约束
    模型不要乱产生。

    ``anchored_tags`` 的 ``offset`` 是相对 ``clean_text`` 而言的。多个 tag 在
    同一位置时按出现顺序保留。
    """
    if not text:
        return text or "", []
    text = text.translate(_BRACKET_TRANS)
    out_parts: List[str] = []
    out_len = 0  # len(''.join(out_parts))，避免反复 join 取长度
    tags: List[AnchoredTag] = []
    last_end = 0
    for m in _TAG_RE.finditer(text):
        # 先把 tag 之前的原文写出去
        if m.start() > last_end:
            chunk = text[last_end:m.start()]
            out_parts.append(chunk)
            out_len += len(chunk)
        key = m.group(1).lower()
        if key in WHITELIST:
            tags.append(AnchoredTag(name=key, offset=out_len))
            # 命中白名单：剥掉，不写入 out_parts；out_len 不增加
        else:
            # 不命中：原样保留方括号（包括内部空格还原成原写法是过度雕琢，这里就保
            # 留 m.group(0) 即可）。
            piece = m.group(0)
            out_parts.append(piece)
            out_len += len(piece)
        last_end = m.end()
    if last_end < len(text):
        tail = text[last_end:]
        out_parts.append(tail)
        out_len += len(tail)
    clean = "".join(out_parts)
    clean = _MULTI_SPACE_RE.sub(" ", clean)
    # 注意：折叠空白会让 offset 不再精确对应 clean 的实际字符；但只有连续空白
    # 段落里的字符索引会偏移，对句首/句尾归属判定不会出错（情绪 tag 不会被
    # 模型放进连续空白里）。简单起见不做 offset 重映射。
    return clean, tags


@dataclass(frozen=True)
class SegmentTagAssignment:
    """``split_tags_to_segments`` 的返回单元。

    Attributes:
        prefix_tags: 落在该 sentence 的 ``[start, end)`` 字符区间内、应作为
            该 sentence 首帧 ``action_tags`` 派发的 tag 列表。
        tail_tags: 仅最后一个元素的 ``tail_tags`` 可能非空；对应所有偏移落
            在 ``len(full_clean_text)`` 末尾、没有归属到任何 sentence 的
            标签，应该作为 ``avatar_speech_end=True`` 包的 ``tail_action_tags``。
    """
    prefix_tags: List[AnchoredTag]
    tail_tags: List[AnchoredTag]


def split_tags_to_segments(
    sentences: Sequence[str],
    anchored_tags: Iterable[AnchoredTag],
) -> List[SegmentTagAssignment]:
    """按字符偏移把 ``anchored_tags`` 派发到 ``sentences``。

    ``sentences`` 应当是按顺序拼接起来等于 clean_text 全文的若干段（典型来源
    是 ``re.split(r'(?<=[,.~!?，。！？])', clean_text)``）。每个 tag 的偏移落
    在 ``[sum(len(s[:i])), sum(len(s[:i+1])))`` 区间内即归属到第 ``i`` 段；
    偏移恰好等于 ``sum(len(s))`` 即派发到最后一段的 ``tail_tags``。

    返回 list 长度与 ``sentences`` 一致；若 ``sentences`` 为空但有 tag，会返
    回一个长度 1 的占位列表，所有 tag 都进 ``tail_tags``。
    """
    tags_sorted = sorted(anchored_tags, key=lambda t: t.offset)
    if not sentences:
        if not tags_sorted:
            return []
        return [SegmentTagAssignment(prefix_tags=[], tail_tags=list(tags_sorted))]

    boundaries: List[Tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        boundaries.append((cursor, cursor + len(sent)))
        cursor += len(sent)
    total_len = cursor

    result = [SegmentTagAssignment(prefix_tags=[], tail_tags=[]) for _ in sentences]

    last_seg = len(sentences) - 1
    for tag in tags_sorted:
        off = tag.offset
        if off >= total_len:
            result[last_seg].tail_tags.append(tag)
            continue
        # 二分查找 sentence index；句数通常很少（≤ 几十），线性也够，但保持
        # 整洁用 bisect-like：找到第一个 end > off 的段。
        idx = 0
        for i, (s, e) in enumerate(boundaries):
            if off < e:
                idx = i
                break
        result[idx].prefix_tags.append(tag)
    return result


def _self_test() -> None:
    # case 1: 标签在开头
    clean, tags = extract_action_tags("[happy]你好啊")
    assert clean == "你好啊", clean
    assert tags == [AnchoredTag("happy", 0)], tags

    # case 2: 标签在中间
    clean, tags = extract_action_tags("我叫小黑！[happy]很高兴认识你。")
    assert clean == "我叫小黑！很高兴认识你。", clean
    assert tags == [AnchoredTag("happy", len("我叫小黑！"))], tags

    # case 3: 标签在结尾
    clean, tags = extract_action_tags("没事啦，我皮实着呢。[happy]")
    assert clean == "没事啦，我皮实着呢。", clean
    assert tags == [AnchoredTag("happy", len(clean))], tags

    # case 4: 非白名单方括号保留
    clean, tags = extract_action_tags("到 [B1] 出口右拐。")
    assert clean == "到 [B1] 出口右拐。", clean
    assert tags == [], tags

    # case 5: 大小写 + 多余空格容忍
    clean, tags = extract_action_tags("[ Happy ]嗨")
    assert clean == "嗨", clean
    assert tags == [AnchoredTag("happy", 0)], tags

    # case 6: split 派发（中文逗号也是分隔符，所以下面是 3 段）
    text = "[apologize]对不起，我没看清。[happy]下次会小心的。"
    clean, tags = extract_action_tags(text)
    assert clean == "对不起，我没看清。下次会小心的。", clean
    sents = [s for s in re.split(r"(?<=[,.~!?，。！？])", clean) if s]
    # 期望：["对不起，", "我没看清。", "下次会小心的。"]
    assert sents == ["对不起，", "我没看清。", "下次会小心的。"], sents
    assign = split_tags_to_segments(sents, tags)
    assert len(assign) == len(sents), (assign, sents)
    # apologize 在 offset 0 → 第 1 段 prefix
    assert assign[0].prefix_tags == [AnchoredTag("apologize", 0)], assign[0]
    # happy 在 offset len("对不起，我没看清。") = 9 → 第 3 段（idx 2）prefix
    happy_off = len("对不起，我没看清。")
    assert any(t.name == "happy" and t.offset == happy_off
               for t in assign[2].prefix_tags), assign
    # 中间段没 tag
    assert assign[1].prefix_tags == []
    assert all(a.tail_tags == [] for a in assign), assign

    # case 7: tail
    text = "全部说完啦。[scared]"
    clean, tags = extract_action_tags(text)
    sents = [s for s in re.split(r"(?<=[,.~!?，。！？])", clean) if s]
    assign = split_tags_to_segments(sents, tags)
    assert assign[-1].tail_tags == [AnchoredTag("scared", len(clean))], assign

    # case 8: 只有标签
    clean, tags = extract_action_tags("[shy]")
    assert clean == ""
    assert tags == [AnchoredTag("shy", 0)]
    sents = [s for s in re.split(r"(?<=[,.~!?，。！？])", clean) if s]
    assign = split_tags_to_segments(sents, tags)
    # sentences 为空 → 1 段占位 tail
    assert len(assign) == 1
    assert assign[0].tail_tags == [AnchoredTag("shy", 0)]
    assert assign[0].prefix_tags == []

    # case 9: 中文书名号式括号 → 与 ASCII 等价
    clean, tags = extract_action_tags("【happy】你好")
    assert clean == "你好", clean
    assert tags == [AnchoredTag("happy", 0)], tags

    print("action_tags self-test ok")


if __name__ == "__main__":
    _self_test()
