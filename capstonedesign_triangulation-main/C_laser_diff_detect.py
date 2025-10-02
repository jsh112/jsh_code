#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_laser_diff_detect.py
- laser_diff_detect_A.py의 핵심 로직을 함수로 재구성
- BEFORE(레이저 OFF) / AFTER(레이저 ON) 프레임 쌍에서 레이저 한 점 좌표(좌/우)를 자동 검출

사용:
    from C_laser_diff_detect import find_laser_point_pair
    ptL, ptR, debug = find_laser_point_pair(beforeL, beforeR, afterL, afterR)

반환:
    ptL, ptR: (x, y) 또는 None
    debug: {"overlayL","overlayR","heatL","heatR"} 디버그 이미지(표시용)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

# ---- 파라미터 (laser_diff_detect_A.py 기본값과 동일/유사) ----
USE_ECC_ALIGN = True
BLUR = 0                    # 0 그대로, 3 이상 홀수면 가우시안 블러
BORDER_IGNORE = 2           # 프레임 테두리 n픽셀 무시
REQUIRE_POSITIVE_DIFF = False
ALPHA = 0.6                 # 히트맵 오버레이 강도

POINT_RADIUS = 10
POINT_FILL_COLOR = (0, 0, 255)
POINT_EDGE_COLOR = (255, 255, 255)
POINT_EDGE_THICK = 2


# ------------- 내부 유틸 -------------
def _to_gray_norm(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m, s = cv2.meanStdDev(g)
    s = max(float(s[0][0]), 1e-6)
    return (g - float(m[0][0])) / s

def _align_translation(src, dst):
    """dst 기준으로 src를 translation 정합(ECC). 입력은 float32 gray."""
    try:
        src_n = cv2.normalize(src, None, 0.0, 1.0, cv2.NORM_MINMAX)
        dst_n = cv2.normalize(dst, None, 0.0, 1.0, cv2.NORM_MINMAX)
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        cv2.findTransformECC(dst_n, src_n, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
        return cv2.warpAffine(src, warp, (dst.shape[1], dst.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return src

def _diff_maps(before_bgr, after_bgr):
    gb = _to_gray_norm(before_bgr)
    ga = _to_gray_norm(after_bgr)
    if USE_ECC_ALIGN:
        ga = _align_translation(ga, gb)
    d = np.abs(ga - gb).astype(np.float32)
    if BLUR >= 3 and BLUR % 2 == 1:
        d = cv2.GaussianBlur(d, (BLUR, BLUR), 0)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return d8, d

def _max_change_pixel(d_float, border_ignore=0, require_positive=False):
    if d_float is None or d_float.size == 0:
        return None
    roi = d_float.copy()
    bi = max(0, int(border_ignore))
    if bi > 0:
        roi[:bi, :] = 0; roi[-bi:, :] = 0; roi[:, :bi] = 0; roi[:, -bi:] = 0
    _, maxV, _, maxLoc = cv2.minMaxLoc(roi)
    if require_positive and maxV <= 0:
        return None
    return maxLoc  # (x, y)

def _make_heatmap(d8):
    if hasattr(cv2, "COLORMAP_TURBO"):
        return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

def _overlay_heat(after_bgr, d8):
    heat = _make_heatmap(d8)
    return cv2.addWeighted(after_bgr, 1.0, heat, ALPHA, 0), heat

def _draw_point(frame, pt):
    if pt is None: return frame
    cv2.circle(frame, pt, POINT_RADIUS, POINT_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, POINT_RADIUS + 2, POINT_EDGE_COLOR, POINT_EDGE_THICK, cv2.LINE_AA)
    return frame


# ------------- 외부 API -------------
def find_laser_point_pair(beforeL, beforeR, afterL, afterR) -> Tuple[Optional[Tuple[int,int]],
                                                                     Optional[Tuple[int,int]],
                                                                     Dict[str, np.ndarray]]:
    """레이저 OFF/ON 프레임 쌍에서 좌/우 레이저 한 점을 찾는다."""
    d8L, dL = _diff_maps(beforeL, afterL)
    d8R, dR = _diff_maps(beforeR, afterR)

    ptL = _max_change_pixel(dL, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)
    ptR = _max_change_pixel(dR, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)

    overL, heatL = _overlay_heat(afterL, d8L)
    overR, heatR = _overlay_heat(afterR, d8R)
    overL = _draw_point(overL, ptL)
    overR = _draw_point(overR, ptR)

    debug = {
        "overlayL": overL, "overlayR": overR,
        "heatL": heatL, "heatR": heatR
    }
    return ptL, ptR, debug
