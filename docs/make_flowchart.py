# -*- coding: utf-8 -*-
"""生成项目二（变化检测）流程图：两张 PNG。

  图1  项目二路线图_一屏总览.png   —— 6 周在干什么、每周门槛、拿不到怎么办
  图2  项目二流程图_详细步骤.png   —— 每一步具体做什么、产出什么、约几天

只用 Pillow + 系统字体（msyh/simhei），不依赖 graphviz / matplotlib。
用法：python docs/make_flowchart.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_R = r"C:\Windows\Fonts\msyh.ttc"

C_BG = (247, 248, 250)
C_INK = (17, 24, 39)
C_GRAY = (90, 100, 115)
C_LINE = (156, 163, 175)
C_GATE = (180, 83, 9)
C_GATE_BG = (254, 243, 199)
C_OUT_BG = (236, 253, 245)
C_OUT_EDGE = (16, 185, 129)
C_RED = (185, 28, 28)
C_OK = (5, 120, 87)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def F(path, size):
    return ImageFont.truetype(path, size)


def wrap(text, font, maxw):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        t = cur + ch
        if font.getlength(t) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def dashed(d, p1, p2, color, w=2, dash=7, gap=6):
    (x1, y1), (x2, y2) = p1, p2
    n = abs(x2 - x1) + abs(y2 - y1)
    if n == 0:
        return
    sx = (x2 - x1) / n
    sy = (y2 - y1) / n
    pos = 0.0
    while pos < n:
        a, b = pos, min(pos + dash, n)
        d.line([(x1 + sx * a, y1 + sy * a), (x1 + sx * b, y1 + sy * b)], fill=color, width=w)
        pos += dash + gap


# =====================================================================
# 图 1：一屏总览
# =====================================================================
def draw_overview(path):
    W = 1940
    X0 = 44
    COL = [("阶段", 150), ("这一周要干成什么", 470), ("门槛（拿不到怎么办）", 500),
           ("周结束时手里必须有", 470), ("不达标的处理", 260)]
    HEAD_H = 52
    ROW_H = 122

    ROWS = [
        dict(badge="准备", date="10/27", color=(75, 85, 99),
             goal="摸清数据真实形态 + 搭目录/实验表（已完成）",
             gate="无门槛（已完成）", mark=None,
             out="docs/data_prior.md、EXPERIMENTS.md、目录骨架",
             fail="—"),
        dict(badge="W1", date="10/27–11/02", color=(37, 99, 235),
             goal="数据能流进模型；基线 B0 跑通",
             gate="B0 的 test F1 ≥ 0.82", mark="G1",
             out="crops 裁片、Dataset、可信评估脚本、B0 数字",
             fail="查数据与指标 bug，别换模型"),
        dict(badge="W2", date="11/03–11/09", color=(124, 58, 237),
             goal="做出自己的方法 M1（特征差分融合）+ 消融",
             gate="M1 的 F1 ≥ 0.86 且高于 B0", mark="G2",
             out="M1 代码、损失三行消融、PR 曲线",
             fail="调参，不换模型（W2 内不许改方案）"),
        dict(badge="W3", date="11/10–11/16", color=(13, 148, 136),
             goal="BIT 对照 + 3 seed 稳定性 + 调优",
             gate="有模型 F1 ≥ 0.88（目标线）", mark="G3",
             out="同协议对照表、mean±std、调优记录",
             fail="接受 0.86–0.88，直接进 W4 用业务线补分"),
        dict(badge="W4", date="11/17–11/23", color=(234, 88, 12),
             goal="图斑化 + 部署对比 + 错误分析",
             gate="闸门 A：10 条全绿 → 当天投递（关键）", mark="GA",
             out="矢量图斑、图斑级 P/Boundary IoU、部署三档表、错误册",
             fail="补缺项；不许硬投（会被追问穿）"),
        dict(badge="W5", date="11/24–11/30", color=(22, 163, 74),
             goal="报告 + Demo + GitHub + 简历三行",
             gate="外行读 README 能懂；3 张图能直接贴简历", mark=None,
             out="报告 10 页、Demo、仓库、简历三行",
             fail="砍细节保主线（宁可少，不可混）"),
        dict(badge="W6", date="12/01–12/07", color=(220, 38, 38),
             goal="8 分钟叙事 + 20 问答案卡 + 项目卡片",
             gate="闸门 B：20 问 ≥16，8 个★全对", mark="GB",
             out="讲稿、答题卡、A4 卡片",
             fail="继续练——硬投只会消耗机会"),
    ]

    f_h1 = F(FONT_B, 44)
    f_sub = F(FONT_R, 23)
    f_head = F(FONT_B, 23)
    f_badge = F(FONT_B, 30)
    f_date = F(FONT_R, 18)
    f_cell = F(FONT_R, 21)
    f_gate = F(FONT_B, 21)
    f_fail = F(FONT_R, 19)
    f_foot = F(FONT_R, 20)
    f_footb = F(FONT_B, 21)

    top = 150
    H = top + HEAD_H + len(ROWS) * ROW_H + 250

    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)

    d.text((X0, 30), "项目二｜变化检测（LEVIR-CD）· 6 周路线图（一屏看懂）", font=f_h1, fill=C_INK)
    d.text((X0 + 2, 88),
           "读法：每行 = 一周，从上往下走。橙色字是「每周必须达到的数字」，红字是「没达到怎么办」——这是防止你在某一周无限打转的刹车。",
           font=f_sub, fill=C_GRAY)

    # 表头
    xs = []
    x = X0
    for name, w in COL:
        xs.append((x, w))
        x += w
    d.rounded_rectangle([X0, top, X0 + sum(w for _, w in COL), top + HEAD_H - 8], 8, fill=(226, 232, 240))
    for (name, w), (cx, _) in zip(COL, xs):
        d.text((cx + 14, top + 9), name, font=f_head, fill=(51, 65, 85))

    y = top + HEAD_H
    for r in ROWS:
        col = r["color"]
        # 交替底色
        if ROWS.index(r) % 2 == 0:
            d.rounded_rectangle([X0, y - 4, X0 + sum(w for _, w in COL), y + ROW_H - 14], 10,
                                fill=(255, 255, 255))
        # 阶段徽标
        bx, bw = xs[0]
        d.rounded_rectangle([bx + 8, y + 10, bx + bw - 22, y + ROW_H - 34], 12, fill=col)
        d.text((bx + (bw - 22 + 8) / 2 - f_badge.getlength(r["badge"]) / 2, y + 22),
               r["badge"], font=f_badge, fill=(255, 255, 255))
        d.text((bx + (bw - 22 + 8) / 2 - f_date.getlength(r["date"]) / 2, y + 66),
               r["date"], font=f_date, fill=(255, 255, 255))
        # 目标
        gx, gw = xs[1]
        yy = y + 18
        for s in wrap(r["goal"], f_cell, gw - 30):
            d.text((gx + 14, yy), s, font=f_cell, fill=C_INK)
            yy += 27
        # 门槛
        qx, qw = xs[2]
        d.rounded_rectangle([qx + 6, y + 14, qx + qw - 12, y + ROW_H - 30], 8,
                            fill=C_GATE_BG, outline=C_GATE, width=2)
        yy = y + 22
        for s in wrap(r["gate"], f_gate, qw - 40):
            d.text((qx + 18, yy), s, font=f_gate, fill=(120, 53, 15))
            yy += 26
        # 产出
        ox, ow = xs[3]
        yy = y + 18
        for s in wrap(r["out"], f_cell, ow - 28):
            d.text((ox + 14, yy), s, font=f_cell, fill=(6, 78, 59))
            yy += 26
        # 不达标
        fx, fw = xs[4]
        yy = y + 18
        for s in wrap(r["fail"], f_fail, fw - 24):
            d.text((fx + 12, yy), s, font=f_fail, fill=C_RED)
            yy += 24
        y += ROW_H

        # 行间箭头
        if ROWS.index(r) < len(ROWS) - 1:
            d.line([(bx + bw / 2 - 7, y - 14), (bx + bw / 2 - 7, y + 2)], fill=C_LINE, width=3)
            d.polygon([(bx + bw / 2 - 7, y + 6), (bx + bw / 2 - 14, y - 6), (bx + bw / 2, y - 6)], fill=C_LINE)

    # 底部三块
    fy = y + 24
    bw3 = (sum(w for _, w in COL) - 40) // 3
    # ① 精度三层线
    d.rounded_rectangle([X0, fy, X0 + bw3, fy + 150], 12, fill=(255, 255, 255), outline=(214, 219, 227), width=2)
    d.text((X0 + 18, fy + 12), "① 精度三层线（F1，test 2048 个 256 裁片）", font=f_footb, fill=C_INK)
    lines = [("0.86  下限", "过了才允许往简历上写", (120, 53, 15)),
             ("0.88  目标", "写进简历的数字（与 STANet 0.889 同档）", C_OK),
             ("0.90  冲刺", "有余力才追，讲不清就别报", (37, 99, 235))]
    yy = fy + 46
    for a, b, c in lines:
        d.text((X0 + 18, yy), a, font=f_footb, fill=c)
        d.text((X0 + 130, yy + 1), b, font=f_foot, fill=C_GRAY)
        yy += 33
    # ② 两个闸门
    x2 = X0 + bw3 + 20
    d.rounded_rectangle([x2, fy, x2 + bw3, fy + 150], 12, fill=(255, 255, 255), outline=(214, 219, 227), width=2)
    d.text((x2 + 18, fy + 12), "② 两个闸门（时间点不能挪）", font=f_footb, fill=C_INK)
    d.text((x2 + 18, fy + 48), "闸门 A · 11/23", font=f_footb, fill=(234, 88, 12))
    d.text((x2 + 186, fy + 49), "10 条全绿 → 开始投递", font=f_foot, fill=C_GRAY)
    d.text((x2 + 18, fy + 82), "闸门 B · 12/07", font=f_footb, fill=(220, 38, 38))
    d.text((x2 + 186, fy + 83), "20 问 ≥16 且 8★全对", font=f_foot, fill=C_GRAY)
    d.text((x2 + 18, fy + 116), "依据：技术面通过率约 30% → 60–70%", font=f_foot, fill=C_GRAY)
    # ③ 现在在哪
    x3 = x2 + bw3 + 20
    d.rounded_rectangle([x3, fy, x3 + bw3, fy + 150], 12, fill=(239, 246, 255), outline=(37, 99, 235), width=2)
    d.text((x3 + 18, fy + 12), "③ 你现在在这里", font=f_footb, fill=(30, 64, 175))
    d.text((x3 + 18, fy + 48), "今天 10/27 → W1 第 1 天", font=f_footb, fill=(30, 64, 175))
    d.text((x3 + 18, fy + 82), "下一步只有 4 件事：裁片 → Dataset →", font=f_foot, fill=C_GRAY)
    d.text((x3 + 18, fy + 108), "评估脚本自检 → B0 跑 3 epoch", font=f_foot, fill=C_GRAY)

    d.text((X0, fy + 182), "缓冲 12/08–12/20　│　闸门 2：12/31 前完成简历与面试准备　│　落后 >3 天就砍：SNUNet 对照、跨数据集泛化 —— 业务线（W4）永远不砍",
           font=f_foot, fill=C_GRAY)

    img.save(path, "PNG")
    print(f"saved: {path}  size={img.size}")


# =====================================================================
# 图 2：详细步骤
# =====================================================================
SECTIONS = [
    dict(no="0", name="准备", color=(75, 85, 99),
         title="准备（10/27，已完成）　摸清家底：数据长什么样、目录和表格先搭好",
         steps=[
             ("0.1", "核查数据的真实形态", "445/64/128 对 1024×1024；256 裁片必须自己切（旧文档写的 1280/256/384 是错的）",
              "docs/data_prior.md\n正样本 4.59/4.20/5.09%", "0.5 天"),
             ("0.2", "搭目录骨架 + 实验登记表", "12 条实验 ID 先排好；跑之前先写「假设」，没有假设的实验不许跑",
              "EXPERIMENTS.md\nconfigs/ data/ models/ …", "0.5 天"),
         ]),
    dict(no="1", name="W1", color=(37, 99, 235),
         title="W1（10/27–11/02）　打地基：让数据能流进模型（最容易崩的一周，别急）",
         steps=[
             ("1.1", "把 1024 大图切成 256 小片", "像把大地图裁成一格格：train 随机裁、val/test 不重叠切 16 块",
              "crops/\n7120 / 1024 / 2048 片", "0.5 天"),
             ("1.2", "写 Dataset：三张图配对 + 同步增强", "A、B、label 必须同步翻转/旋转，否则会人为造出「假变化」",
              "能迭代的 DataLoader", "1 天"),
             ("1.3", "改评估脚本 + 用极端值自检", "先让它预测「全 0」（F1 应=0、OA≈95%）再预测「全 1」（F1≈0.1）",
              "可信的 OA/F1/IoU 脚本", "0.5 天"),
             ("G1", "B0 小规模跑 3 epoch：test F1 ≥ 0.82？",
              "达到=管线通了，可以正式开始；没达到先查数据与指标 bug，别急着换模型",
              None, "0.5 天", "否 ▶ 回到 1.1–1.3 查数据/指标"),
             ("1.4", "训 B0 / B1 基线 30 epoch", "B0 从零 U-Net(7.85M) + B1 ResNet34(24.5M)；单变量阶梯，记显存/耗时",
              "B0 的 F1/IoU + loss 曲线", "1 天"),
         ]),
    dict(no="2", name="W2", color=(124, 58, 237),
         title="W2（11/03–11/09）　做出你自己的方法（简历主打从这周来）",
         steps=[
             ("2.1", "实现「特征差分融合」M1", "让网络显式看 A 与 B 差在哪，而不是自己从两张图里猜",
              "M1 代码 + 结构图", "1 天"),
             ("2.2", "损失函数三行消融", "BCE / Dice / 组合：看谁在 1:19 的不平衡下不塌向全 0",
              "消融表 3 行数字", "1.5 天"),
             ("2.3", "训 M1 + 阈值扫描", "0.2–0.8 扫一遍画 PR 曲线，为「按业务选阈值」留证据",
              "PR 曲线 + 最优阈值", "1 天"),
             ("G2", "M1 的 F1 ≥ 0.86（下限线）且高于 B0？",
              "没到就在本周内调参；W2 结束时不许还在改方案，否则整个 6 周会崩",
              None, "0.5 天", "否 ▶ 回 2.3 调参（不换模型）"),
         ]),
    dict(no="3", name="W3", color=(13, 148, 136),
         title="W3（11/10–11/16）　对照与稳定性：让数字站得住，别人一问不塌",
         steps=[
             ("3.1", "评测 BIT（本地已有权重）", "同一套裁片、同一个脚本；比你好就如实写「这是我理解的工程折中」",
              "BIT 同协议 1 行数字", "0.5 天"),
             ("3.2", "3 个 seed 重跑主模型", "报 mean±std；标准差 >0.8 点说明训练不稳，先修稳再谈提升",
              "mean±std 三行", "1.5 天"),
             ("3.3", "有针对性调优", "融合位置 / 损失权重 / 学习率：一次只改一个变量",
              "调优记录（含失败的）", "1.5 天"),
             ("G3", "有模型 F1 ≥ 0.88（目标线）？",
              "到了很好；没到就接受 0.86–0.88 的现实，用 W4 的业务线补分——不要让数字卡住整个项目",
              None, "0.5 天", "否 ▶ 接受现况，直接进 W4"),
         ]),
    dict(no="4", name="W4", color=(234, 88, 12),
         title="W4（11/17–11/23）　从「模型」变成「能交付的成果」——投递闸门就在这周",
         steps=[
             ("4.1", "像素 → 矢量图斑", "连通域提取 + MMU 40/80/200 三档 + 碎斑合并前后对比",
              "shp/geojson + 面积分布图", "1.5 天"),
             ("4.2", "算两个业务指标", "图斑级 Precision、Boundary IoU（大斑块占 96% 像素，边界差 1px 就掉分）",
              "图斑级 P / Boundary IoU", "0.5 天"),
             ("4.3", "1024 大图滑窗拼接", "重叠切块再拼回去、检查接缝；顺便测单张时延",
              "无接缝大图 + 时延", "1 天"),
             ("4.4", "导出 ONNX / TensorRT", "fp32 / fp16 / int8 三档：记掉点与加速比（JD 里最缺的加分项）",
              "部署对比表", "1 天"),
             ("4.5", "错误分析：四类错误各 3 例", "新增漏检 / 拆除漏检 / 伪变化 / 边界；配图卡与占比",
              "错误图册 + 各类占比", "1 天"),
             ("GA", "闸门 A：10 条准入清单全绿？",
              "全绿=当天开始投递（边投边补）；缺项就补，不许硬投——面试官 3 分钟就能问穿",
              None, "0.5 天", "否 ▶ 补缺项，顺延 2–3 天"),
         ]),
    dict(no="5", name="W5", color=(22, 163, 74),
         title="W5（11/24–11/30）　把做过的事，讲成别人看得懂的东西",
         steps=[
             ("5.1", "技术报告 10 页内", "任务→数据→方法→结果→消融→错误→部署→局限（局限要自己先写）",
              "报告 PDF", "1.5 天"),
             ("5.2", "做一个对比 Demo", "A / B / 预测 / 图斑 四联图，最好能自己换张图跑一遍",
              "静态页或 gradio", "1 天"),
             ("5.3", "GitHub README", "首屏 3 张图 + 一条命令复现 + 指标表；外行能读懂才算过",
              "可点开的仓库", "0.5 天"),
             ("5.4", "简历三行", "每行都要能被追问 3 层，数字全部能在仓库里查到",
              "简历三行", "0.5 天"),
         ]),
    dict(no="6", name="W6", color=(220, 38, 38),
         title="W6（12/01–12/07）　面试准备：能不能拿到 offer，决定在这周",
         steps=[
             ("6.1", "8 分钟叙事稿", "问题→难在哪→我的选择→数字→失败与反思（不许从「我用了个 U-Net」开头）",
              "讲稿 1 页", "0.5 天"),
             ("6.2", "20 问答案卡 + 录音自测", "8 个★必须全对；每题口述 60–90 秒，卡住的标红重练",
              "答题卡 + 自测记录", "1.5 天"),
             ("6.3", "1 页项目卡片", "面试时放手边，随时能指着一张图讲清楚",
              "A4 卡片", "0.5 天"),
             ("GB", "闸门 B：20 问 ≥16 且 8 个★全对？",
              "没到就继续练；闸门 B 通过后技术面通过率约从 30% 提到 60–70%",
              None, "0.5 天", "否 ▶ 继续练，不硬投"),
         ]),
]


def draw_detail(path):
    W = 1760
    X_BAND, W_BAND = 44, 104
    X0, W_STEP = 250, 940
    X_OUT, W_OUT = 1230, 420
    PAD = 24
    GAP_STEP = 30
    GAP_SEC = 46

    f_h1 = F(FONT_B, 40)
    f_h1s = F(FONT_R, 22)
    f_sec = F(FONT_B, 28)
    f_num = F(FONT_B, 24)
    f_t = F(FONT_B, 26)
    f_d = F(FONT_R, 21)
    f_out = F(FONT_R, 20)
    f_gate = F(FONT_B, 22)
    f_lab = F(FONT_B, 20)
    f_leg = F(FONT_R, 20)
    f_band = F(FONT_B, 26)
    f_ban2 = F(FONT_R, 19)
    f_dur = F(FONT_R, 19)
    f_fail = F(FONT_R, 19)
    f_footb = F(FONT_B, 22)

    def step_height(st):
        no, title, detail, out = st[0], st[1], st[2], st[3]
        fail = st[5] if len(st) > 5 else None
        if no.startswith("G"):
            n = len(wrap(title, f_gate, 470)) + len(wrap(detail, f_out, 470))
            if fail:
                n += len(wrap(fail, f_fail, 470))
            return 56 + n * 25 + 46
        tl = wrap(title, f_t, W_STEP - 2 * PAD - 64)
        dl = wrap(detail, f_d, W_STEP - 2 * PAD - 64)
        return PAD + len(tl) * 33 + 6 + len(dl) * 28 + PAD

    def out_height(out):
        if not out:
            return 0
        lines = []
        for seg in out.split("\n"):
            lines += wrap(seg, f_out, W_OUT - 32)
        return max(72, len(lines) * 26 + 34)

    layout, y = [], 168
    for sec in SECTIONS:
        sec_y = y
        y += 50
        items = []
        for st in sec["steps"]:
            h = step_height(st)
            items.append(dict(st=st, y=y, h=h, oh=out_height(st[3])))
            y += h + GAP_STEP
        y += GAP_SEC
        layout.append(dict(sec=sec, y0=sec_y, y1=y - GAP_SEC + 8, items=items))

    H = layout[-1]["y1"] + 250
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)

    d.text((X_BAND, 34), "项目二｜变化检测（LEVIR-CD）· 详细步骤（每周拆到可执行的小步）", font=f_h1, fill=C_INK)
    d.text((X_BAND + 2, 84), "读法：菱形是「判定点」，橙色问句=周门槛，红字=没达标怎么办；右侧绿框=这一步必须产出的东西（没有产出就等于没做）。",
           font=f_h1s, fill=C_GRAY)

    # 图例
    lx, ly = X_BAND + 2, 122
    for kind, txt in [("box", "步骤"), ("dia", "判定点"), ("out", "产出物"), ("dur", "预计耗时")]:
        if kind == "box":
            d.rounded_rectangle([lx, ly, lx + 24, ly + 18], 4, fill=(255, 255, 255),
                                outline=(37, 99, 235), width=3)
            lx += 32
        elif kind == "dia":
            d.polygon([(lx, ly + 9), (lx + 12, ly), (lx + 24, ly + 9), (lx + 12, ly + 18)],
                      fill=C_GATE_BG, outline=C_GATE)
            lx += 32
        elif kind == "out":
            d.rounded_rectangle([lx, ly, lx + 24, ly + 18], 4, fill=C_OUT_BG, outline=C_OUT_EDGE, width=3)
            lx += 32
        else:
            d.text((lx, ly - 2), "≈0.5 天", font=f_dur, fill=C_GRAY)
            lx += 62
        d.text((lx, ly - 2), txt, font=f_leg, fill=C_GRAY)
        lx += d.textlength(txt, font=f_leg) + 30

    def arrow(x, y1, y2, color=C_LINE, w=3):
        d.line([(x, y1), (x, y2 - 12)], fill=color, width=w)
        d.polygon([(x, y2), (x - 9, y2 - 15), (x + 9, y2 - 15)], fill=color)

    for L in layout:
        sec = L["sec"]
        col = sec["color"]
        # 左侧阶段带
        d.rounded_rectangle([X_BAND, L["y0"], X_BAND + W_BAND, L["y1"]], 14, fill=col)
        cx = X_BAND + W_BAND // 2
        ty = L["y0"] + 18
        for ch in sec["name"]:
            d.text((cx - f_band.getlength(ch) / 2, ty), ch, font=f_band, fill=(255, 255, 255))
            ty += 32
        # 章节标题条
        d.rounded_rectangle([X0, L["y0"], X0 + W_STEP + 300, L["y0"] + 40], 8, fill=(238, 242, 255))
        d.text((X0 + 14, L["y0"] + 6), sec["title"], font=f_sec, fill=col)

        for it in L["items"]:
            st = it["st"]
            no, title, detail, out = st[0], st[1], st[2], st[3]
            dur = st[4] if len(st) > 4 else ""
            fail = st[5] if len(st) > 5 else None
            y0, h = it["y"], it["h"]

            if no.startswith("G"):
                cxx = X0 + W_STEP // 2
                cy = y0 + h // 2
                dw, dh = 700, h - 10
                d.polygon([(cxx, cy - dh // 2), (cxx + dw // 2, cy),
                           (cxx, cy + dh // 2), (cxx - dw // 2, cy)],
                          fill=C_GATE_BG, outline=C_GATE, width=3)
                tl = wrap(title, f_gate, 520)
                dl = wrap(detail, f_out, 520)
                fl = wrap(fail, f_fail, 520) if fail else []
                th = len(tl) * 27 + len(dl) * 25 + len(fl) * 25
                yy = cy - th // 2
                for s in tl:
                    d.text((cxx - f_gate.getlength(s) / 2, yy), s, font=f_gate, fill=(120, 53, 15))
                    yy += 27
                for s in dl:
                    d.text((cxx - f_out.getlength(s) / 2, yy), s, font=f_out, fill=C_GRAY)
                    yy += 25
                for s in fl:
                    d.text((cxx - f_fail.getlength(s) / 2, yy), s, font=f_fail, fill=C_RED)
                    yy += 25
                d.text((cxx + dw // 2 + 14, cy - 12), "是 ↓", font=f_lab, fill=C_OK)
                continue

            d.rounded_rectangle([X0, y0, X0 + W_STEP, y0 + h], 12, fill=(255, 255, 255),
                                outline=(214, 219, 227), width=2)
            d.rounded_rectangle([X0, y0, X0 + 8, y0 + h], 4, fill=col)
            bx = X0 + PAD
            d.rounded_rectangle([bx, y0 + PAD - 2, bx + 52, y0 + PAD + 28], 8, fill=col)
            d.text((bx + 26 - f_num.getlength(no) / 2, y0 + PAD - 1), no, font=f_num, fill=(255, 255, 255))
            tx = bx + 64
            tw = W_STEP - (tx - X0) - PAD
            tl = wrap(title, f_t, tw)
            dl = wrap(detail, f_d, tw)
            yy = y0 + PAD - 3
            for s in tl:
                d.text((tx, yy), s, font=f_t, fill=C_INK)
                yy += 33
            yy += 6
            for s in dl:
                d.text((tx, yy), s, font=f_d, fill=C_GRAY)
                yy += 28
            # 耗时
            if dur:
                d.text((X0 + W_STEP - 14 - f_dur.getlength(dur), y0 + PAD - 1), dur, font=f_dur, fill=(148, 163, 184))
            # 产出
            if out:
                oh = max(72, it["oh"])
                oy = y0 + (h - oh) // 2
                d.rounded_rectangle([X_OUT, oy, X_OUT + W_OUT, oy + oh], 10, fill=C_OUT_BG,
                                    outline=C_OUT_EDGE, width=2)
                d.text((X_OUT + 14, oy + 6), "产出", font=f_lab, fill=C_OK)
                yy = oy + 30
                for seg in out.split("\n"):
                    for s in wrap(seg, f_out, W_OUT - 28):
                        d.text((X_OUT + 14, yy), s, font=f_out, fill=(6, 78, 59))
                        yy += 26

        # 章内箭头
        items = L["items"]
        for i in range(len(items) - 1):
            a, b = items[i], items[i + 1]
            if a["st"][0].startswith("G"):
                continue
            arrow(X0 + PAD + 26, a["y"] + a["h"], b["y"], color=(203, 213, 225))

    # 章间箭头
    for i in range(len(layout) - 1):
        a, b = layout[i], layout[i + 1]
        last = a["items"][-1]
        x = X0 + W_STEP // 2 if last["st"][0].startswith("G") else X0 + PAD + 26
        arrow(x, a["y1"] - 6, b["y0"] + 40, color=(148, 163, 184), w=4)

    # 页脚：内循环
    fy = layout[-1]["y1"] + 46
    d.rounded_rectangle([X_BAND, fy, W - 40, fy + 150], 12, fill=(255, 255, 255),
                        outline=(214, 219, 227), width=2)
    d.text((X_BAND + 20, fy + 14), "每个实验都是同一个内循环（全项目约 20 次，别跳步）", font=f_footb, fill=(37, 99, 235))
    loop_items = ["写假设\n（能被证伪）", "只改一个\n变量", "训练\n2–3 小时", "统一脚本\n评估",
                  "EXPERIMENTS\n记一行", "看结论\n定下一步"]
    bw, bg = 246, 18
    bx, by = X_BAND + 22, fy + 56
    for i, t in enumerate(loop_items):
        d.rounded_rectangle([bx, by, bx + bw, by + 78], 10, fill=(239, 246, 255),
                            outline=(191, 219, 254), width=2)
        yy = by + 16
        for s in t.split("\n"):
            d.text((bx + bw / 2 - f_d.getlength(s) / 2, yy), s, font=f_d, fill=(30, 64, 175))
            yy += 28
        if i < len(loop_items) - 1:
            d.line([(bx + bw + 2, by + 39), (bx + bw + bg - 8, by + 39)], fill=(96, 165, 250), width=3)
            d.polygon([(bx + bw + bg - 2, by + 39), (bx + bw + bg - 16, by + 32),
                       (bx + bw + bg - 16, by + 46)], fill=(96, 165, 250))
        bx += bw + bg

    img.save(path, "PNG")
    print(f"saved: {path}  size={img.size}")


if __name__ == "__main__":
    draw_overview(os.path.join(OUT_DIR, "项目二路线图_一屏总览.png"))
    draw_detail(os.path.join(OUT_DIR, "项目二流程图_详细步骤.png"))



