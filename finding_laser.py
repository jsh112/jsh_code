#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
두 웹캠(기본 1,2)에서 전/후 수동 캡처.
절대차 |after-before|의 전역 최대 '한 점'을 레이저로 가정해 표시.
- after 화면: 히트맵 오버레이 + 한 점(빨강 원)
- diff 화면: 순수 히트맵 + 동일 지점 마커
키: Space(전→후/후 갱신), x(초기화), q(종료)
"""

import cv2, numpy as np, platform

# ========= 설정 =========
CAM0, CAM1 = 0, 1          # 카메라 인덱스
W, H = 1280, 720           # 해상도
USE_ECC_ALIGN = True       # 전/후 미세 흔들림(translation) 정합
BLUR = 0                   # 1픽셀 유지면 0, 약간 번지면 3
BORDER_IGNORE = 2          # 프레임 테두리 n픽셀 무시
REQUIRE_POSITIVE_DIFF = False  # 최대값이 0이어도 한 점 찍을지(False=찍음)

# 히트맵/표시
ALPHA = 0.6                        # 오버레이 강도
USE_TURBO = True                   # 팔레트: TURBO(권장) / JET
POINT_RADIUS = 10                  # 표시 점 크기
POINT_FILL_COLOR = (0, 0, 255)     # 빨강(가시성↑)
POINT_EDGE_COLOR = (255, 255, 255) # 흰 테두리
POINT_EDGE_THICK = 2
# =======================

def open_cam(idx):
    be = 0
    if platform.system() == "Windows": be = cv2.CAP_DSHOW
    elif platform.system() == "Linux": be = cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, be)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def to_gray_norm(bgr):
    """BGR→gray(float32) 후 z-score 정규화로 조도 변화 영향 축소."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m, s = cv2.meanStdDev(g); s = max(float(s[0][0]), 1e-6)
    return (g - float(m[0][0])) / s

def align_translation(src, dst):
    """dst 기준으로 src를 translation 정합(ECC). 입력은 float32 gray."""
    try:
        # ECC 수렴 안정화(0..1 정규화)
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

def diff_maps(before_bgr, after_bgr):
    """
    반환:
      d8 : 0..255 uint8 절대차(히트맵용)
      d  : float32 절대차(최대점 탐색용)
    """
    gb = to_gray_norm(before_bgr)
    ga = to_gray_norm(after_bgr)
    if USE_ECC_ALIGN:
        ga = align_translation(ga, gb)
    d = np.abs(ga - gb).astype(np.float32)
    if BLUR >= 3 and BLUR % 2 == 1:
        d = cv2.GaussianBlur(d, (BLUR, BLUR), 0)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return d8, d

def max_change_pixel(d_float, border_ignore=0, require_positive=False):
    """모든 픽셀 중 전역 최대 한 점 좌표 리턴."""
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

def make_heatmap(d8):
    """순수 히트맵 BGR 이미지 반환."""
    if USE_TURBO and hasattr(cv2, "COLORMAP_TURBO"):
        return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

def overlay_heat(after_bgr, d8):
    """after 위에 히트맵 반투명 오버레이."""
    heat = make_heatmap(d8)
    return cv2.addWeighted(after_bgr, 1.0, heat, ALPHA, 0), heat

def draw_point(frame, pt):
    if pt is None: return frame
    cv2.circle(frame, pt, POINT_RADIUS, POINT_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, POINT_RADIUS + 2, POINT_EDGE_COLOR, POINT_EDGE_THICK, cv2.LINE_AA)
    return frame

def main():
    cap0, cap1 = open_cam(CAM0), open_cam(CAM1)
    cv2.namedWindow("cam0",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam1",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("diff0", cv2.WINDOW_NORMAL)  # 순수 히트맵
    cv2.namedWindow("diff1", cv2.WINDOW_NORMAL)

    before0 = before1 = None
    print("[Space] 전→후(히트맵+최대 한 점) | [x] 초기화 | [q] 종료 — 저장 없음")

    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1): break

        # 상태 표시
        v0, v1 = f0.copy(), f1.copy()
        status = "READY: Space=BEFORE 촬영" if before0 is None else "BEFORE OK: Space=AFTER 촬영→표시  /  x=초기화"
        for img in (v0, v1):
            cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("cam0", v0); cv2.imshow("cam1", v1)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('x'):
            before0 = before1 = None
            print("[초기화]")
        elif k == 32:  # Space
            if before0 is None:
                before0, before1 = f0.copy(), f1.copy()
                print("[BEFORE] 촬영")
            else:
                after0, after1 = f0.copy(), f1.copy()
                print("[AFTER]  촬영 → 히트맵+최대 변화 한 점 표시")

                # 1) 절대차 맵들
                d8_0, d0 = diff_maps(before0, after0)
                d8_1, d1 = diff_maps(before1, after1)

                # 2) 모든 픽셀 중 전역 최대 한 점
                pt0 = max_change_pixel(d0, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)
                pt1 = max_change_pixel(d1, BORDER_IGNORE, REQUIRE_POSITIVE_DIFF)

                # 3) after 위 히트맵 오버레이 + 한 점 표시
                over0, heat0 = overlay_heat(after0, d8_0)
                over1, heat1 = overlay_heat(after1, d8_1)
                over0 = draw_point(over0, pt0)
                over1 = draw_point(over1, pt1)

                # 4) 순수 히트맵에도 동일 지점 마커(가독성↑)
                if pt0 is not None:
                    cv2.drawMarker(heat0, pt0, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                if pt1 is not None:
                    cv2.drawMarker(heat1, pt1, (0,255,0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                # 5) 표출
                cv2.imshow("cam0", over0)
                cv2.imshow("cam1", over1)
                cv2.imshow("diff0", heat0)
                cv2.imshow("diff1", heat1)

    cap0.release(); cap1.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
