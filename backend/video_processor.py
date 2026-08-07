"""
VirtuCoach - 视频处理模块
MediaPipe 手部关键点检测（本地毫秒级）+ 超时保护
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback

from logging_config import get_logger

logger = get_logger(__name__)

TIMEOUT_SECONDS = 45  # 整个视频处理超时（秒），需覆盖80帧检测+截图


def _pick_best_snapshot_time(group: list, hand_issue_frames: list) -> float:
    """从 0.5s 窗口内的多个时间点中选出最优截图时刻。

    优先选有 ⚠️ 严重问题的帧，其次选问题数量最多的帧。
    """
    if len(group) == 1:
        return group[0]
    # 统计每个时间点的问题严重度
    best_t, best_score = group[0], -1
    for t in group:
        score = 0
        for h in hand_issue_frames:
            if abs(round(h.get("timestamp", 0), 2) - t) < 0.05:
                for iss in h.get("issues", []):
                    if iss.startswith("⚠️"):
                        score += 3
                    elif iss.startswith("💡"):
                        score += 1
        if score > best_score:
            best_score = score
            best_t = t
    return best_t



class VideoProcessor:
    """视频处理器：提取关键帧 + MediaPipe 手型检测"""

    def __init__(self):
        self.model_loaded = False
        self._mirror_detected = False
        self._camera_angle = "unknown"
        self._load_mediapipe()

    def _load_mediapipe(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

            self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "models", "hand_landmarker.task")
            if not os.path.exists(self._model_path):
                logger.warning(f"HandLandmarker 模型文件不存在: {self._model_path}")
                return

            # Store shared config for creating fresh HandLandmarker per video
            self._mp = mp
            self._HandLandmarker = HandLandmarker
            self._HandLandmarkerOptions = HandLandmarkerOptions
            self._RunningMode = RunningMode
            self._BaseOptions = BaseOptions
            self._mp_image = mp.Image
            self._mp_image_format = mp.ImageFormat
            self.model_loaded = True
            logger.info("MediaPipe Hands 就绪 (tasks API)")
        except ImportError:
            logger.warning("mediapipe 未安装！手型检测功能将不可用")
        except Exception as e:
            logger.warning(f"MediaPipe 加载失败: {e}")

    def _create_detector(self):
        """Create a fresh HandLandmarker instance (avoids monotonic timestamp issues)."""
        options = self._HandLandmarkerOptions(
            base_options=self._BaseOptions(model_asset_path=self._model_path),
            running_mode=self._RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return self._HandLandmarker.create_from_options(options)

    def process(self, video_path: str, error_timestamps: List[float], instrument: str,
                technique_hints: Optional[List[str]] = None,
                technique_segments: Optional[List[Dict]] = None,
                force_hand: Optional[str] = None,
                detected_chords: Optional[List[Dict]] = None) -> Dict:
        """处理视频（带超时保护）。

        Args:
            technique_hints: 可选，已知的技巧上下文列表。
            technique_segments: 可选，技巧时间段列表。
            force_hand: 可选，强制指定手别 ("left" 或 "right")。
            detected_chords: 可选，音频检测到的和弦列表 [{time, chord_id, chord_name, confidence}]，
                            用于 KB 驱动的和弦感知手型检测（Phase 2）。
        """
        result = {
            "frames_analyzed": 0, "total_frames_extracted": 0,
            "hand_landmarks": [], "hand_issues": [],
            "snapshots": [], "scored_frames": [],
            "frame_analysis_data": '{"hands_found":false,"frames_checked":0,"issues_found":0,"issues_list":[],"instrument":""}',
        }

        if not self.model_loaded:
            return result

        try:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(self._process_internal, video_path, error_timestamps, instrument, technique_hints, technique_segments, force_hand, detected_chords)
                try:
                    result = future.result(timeout=TIMEOUT_SECONDS)
                except TimeoutError:
                    logger.warning(f"处理超时（{TIMEOUT_SECONDS}s），等待线程完成...")
                    try:
                        result = future.result(timeout=15)
                    except Exception:
                        logger.warning("等待线程完成也超时，跳过手型分析")
                        result["hand_issues"] = []
                except Exception as e:
                    logger.warning(f"处理异常: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.warning(f"启动失败: {e}")

        # 如果没有检测到手，加一个友好的提示
        if not result.get("hand_issues"):
            result["hand_issues"] = []

        return result

    def _process_internal(self, video_path: str, error_timestamps: List[float], instrument: str,
                          technique_hints: Optional[List[str]] = None,
                          technique_segments: Optional[List[Dict]] = None,
                          force_hand: Optional[str] = None,
                          detected_chords: Optional[List[Dict]] = None) -> Dict:
        """内部处理逻辑"""
        self._mirror_detected = False
        self._camera_angle = "unknown"
        self._force_hand = force_hand
        self._detected_chords = detected_chords  # Phase 2b 使用

        detector = self._create_detector()  # 每次视频创建新探测器，避免时间戳不单调

        result = {
            "frames_analyzed": 0, "total_frames_extracted": 0,
            "hand_landmarks": [], "hand_issues": [],
            "snapshots": [], "scored_frames": [],
            "frame_analysis_data": '{"hands_found":false,"frames_checked":0,"issues_found":0,"issues_list":[],"instrument":""}',
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"无法打开视频: {video_path}")
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"视频: {fps:.1f}fps, {duration:.1f}s, {total_frames}帧")

        frame_timestamps = sorted(set(self._select_keyframes(duration, fps, error_timestamps, technique_segments)))
        # 新 mediapipe tasks API 要求时间戳严格单调递增
        prev_ts = -0.001
        clean_ts = []
        for ts in frame_timestamps:
            if ts > prev_ts:
                clean_ts.append(ts)
                prev_ts = ts
        frame_timestamps = clean_ts
        hand_issue_frames = []
        hands_detected_anywhere = False
        wrist_x_samples = []  # 左手腕x坐标采样，用于估算手区品数
        all_hand_landmarks = []  # 累积所有帧检测到的手部 landmarks

        # 多帧一致性追踪: {(handedness, issue_sig): [consecutive_count, gap_count]}
        # 需连续≥3帧确认才报告，容忍1帧短暂消失
        issue_frame_tracker = {}
        reported_sigs = set()
        prev_frame_sigs = set()
        MIN_CONSECUTIVE = 4  # ~1.2s sustained issue before reporting (4 frames × 0.3s interval)
        MAX_GAP = 1          # 容忍1帧间隙

        for ts in frame_timestamps:
            frame_idx = int(ts * fps)
            if frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            # 帧定位回退：某些视频格式下CAP_PROP_POS_FRAMES不准，用MSEC重试
            if not ret or (ret and frame is not None and frame.size == 0):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp_image(image_format=self._mp_image_format.SRGB, data=frame_rgb)
            results = detector.detect_for_video(mp_image, int(ts * 1000))

            cur_frame_sigs = set()
            if results.hand_landmarks:
                hands_detected_anywhere = True
                frame_landmarks = []
                # 首次检测到手时估算拍摄角度
                if self._camera_angle == "unknown" and results.hand_landmarks:
                    raw_landmarks = [[(lm.x, lm.y, lm.z) for lm in hlm] for hlm in results.hand_landmarks]
                    self._camera_angle = self._estimate_camera_angle(raw_landmarks)
                    logger.info(f"[camera] estimated angle: {self._camera_angle}")
                for idx, hlm in enumerate(results.hand_landmarks):
                    handedness = "unknown"
                    handedness_confidence = 0.0
                    if results.handedness and idx < len(results.handedness):
                        category = results.handedness[idx][0]
                        handedness = category.category_name
                        handedness_confidence = round(float(category.score), 3)

                    wrist_x = hlm[0].x

                    # 镜像自动检测：前置摄像头画面中，左手(按弦手)出现在画面右侧
                    # 如果 MediaPipe 说 "Left" 但手腕在画面右侧(wrist_x > 0.5)，说明是镜像画面
                    if not self._mirror_detected and handedness != "unknown" and handedness_confidence > 0.6:
                        label_says_left = (handedness == "Left")
                        hand_on_right = (wrist_x > 0.5)
                        if label_says_left and hand_on_right:
                            self._mirror_detected = True
                            logger.info(f"[mirror] auto-detected mirror (front camera), wrist_x={wrist_x:.2f}")
                        elif not label_says_left and not hand_on_right:
                            self._mirror_detected = True
                            logger.info(f"[mirror] auto-detected mirror (front camera), wrist_x={wrist_x:.2f}")

                    # 镜像画面翻转 handedness：前置摄像头左右颠倒，MP 解剖学分类可能误判。
                    # 例如左手（按弦手）在前置摄像头中出现在画面右侧，MP 可能标为 "Right"
                    if self._mirror_detected and handedness != "unknown":
                        handedness = "Right" if handedness == "Left" else "Left"

                    if handedness == "unknown":
                        if self._mirror_detected:
                            # 镜像画面：按弦手(左手)在画面右侧
                            handedness = "Left" if wrist_x > 0.5 else "Right"
                        else:
                            # 非镜像画面：按弦手(左手)在画面左侧
                            handedness = "Left" if wrist_x < 0.5 else "Right"
                    else:
                        # 信任 MediaPipe 解剖学分类，不做交叉验证修正
                        # MediaPipe 看的是手部解剖结构（拇指位置等），比手腕位置启发式可靠得多
                        # 仅在 handedness 与预期严重矛盾且置信度极低时才修正
                        if handedness_confidence < 0.5:
                            if self._mirror_detected:
                                wrist_says_left = (wrist_x > 0.5)
                            else:
                                wrist_says_left = (wrist_x < 0.5)
                            label_says_left = (handedness == "Left")
                            if label_says_left != wrist_says_left:
                                corrected = "Left" if wrist_says_left else "Right"
                                logger.info(f"[correct] low-confidence fix: {handedness}(conf={handedness_confidence}) -> {corrected}")
                                handedness = corrected

                    # 强制手别：和弦检查等场景始终只关心一只手。
                    # 结合手在画面中的位置判断是否为镜像画面（前置摄像头中：
                    # 按弦手/左手在画面右侧 wrist_x>0.5，弹奏手/右手在画面左侧 wrist_x<0.5）
                    if self._force_hand:
                        expected = "Left" if self._force_hand == "left" else "Right"
                        if handedness != expected:
                            force_left_on_right = (self._force_hand == "left" and wrist_x > 0.5)
                            force_right_on_left = (self._force_hand == "right" and wrist_x < 0.5)
                            if force_left_on_right or force_right_on_left:
                                # 手在画面中的位置与 force_hand 的镜像预期一致 → 检测为镜像
                                if not self._mirror_detected:
                                    self._mirror_detected = True
                                    logger.info(f"[mirror] detected via force_hand+position, wrist_x={wrist_x:.2f}")
                                handedness = expected
                            elif handedness_confidence < 0.7:
                                handedness = expected
                                logger.info(f"[force_hand] corrected handedness: {handedness}->{expected} (conf={handedness_confidence})")
                                logger.debug(f"low-confidence hand label corrected to {handedness} (conf={handedness_confidence})")

                    # 记录左手腕位置用于品位估算
                    if handedness == "Left":
                        wrist_x_samples.append(wrist_x)

                    # 使用 WorldLandmarks (3D 真实坐标) 计算角度，避免 2D 透视畸变
                    world_lm = None
                    if (results.hand_world_landmarks
                            and idx < len(results.hand_world_landmarks)):
                        world_lm = results.hand_world_landmarks[idx]
                    angles = self._calc_angles(hlm, world_lm)
                    # Phase 2b: 查找当前帧对应的和弦上下文
                    chord_ctx = {}
                    chord_id_matched = None
                    dc = getattr(self, '_detected_chords', None)
                    if dc and handedness == "Left":
                        for chord in dc:
                            ct = chord.get("time", 0)
                            c_end = chord.get("end_time", ct + 2.0)
                            if ct - 0.3 <= ts <= c_end + 0.3:
                                cid = chord.get("chord_id", "")
                                if cid:
                                    try:
                                        from knowledge_db import knowledge_db as kdb
                                        chord_ctx = kdb.get_chord_angle_standards(cid)
                                        chord_id_matched = cid
                                    except Exception:
                                        pass
                                break
                    if chord_id_matched and chord_ctx:
                        logger.debug(f"[chord-match] ts={ts:.1f}s → {chord_id_matched} kb_fingers={list(chord_ctx.keys())}")
                    elif chord_id_matched:
                        logger.debug(f"[chord-match] ts={ts:.1f}s → {chord_id_matched} (no finger angles in KB)")
                    issues = self._detect_issues_with_chord(angles, handedness, instrument, chord_ctx)
                    lm_list = [(lm.x, lm.y, lm.z) for lm in hlm]
                    frame_landmarks.append(lm_list)
                    all_hand_landmarks.append({"time": round(ts, 2), "hand": handedness, "landmarks": lm_list})

                    if issues:
                        for iss in issues:
                            sig = iss.split(" ", 1)[-1] if " " in iss else iss
                            sig_key = f"{handedness}|{sig[:30]}"
                            cur_frame_sigs.add(sig_key)

                            if sig_key in issue_frame_tracker:
                                issue_frame_tracker[sig_key]["count"] += 1
                                issue_frame_tracker[sig_key]["gap"] = 0
                            else:
                                issue_frame_tracker[sig_key] = {"count": 1, "gap": 0}

                            # 连续≥3帧时报告一次
                            if issue_frame_tracker[sig_key]["count"] >= MIN_CONSECUTIVE and sig_key not in reported_sigs:
                                reported_sigs.add(sig_key)
                                hand_label = "左手" if handedness == "Left" else "右手" if handedness == "Right" else "未知"
                                hand_issue_frames.append({
                                    "timestamp": round(ts, 2),
                                    "handedness": hand_label,
                                    "hand_label": hand_label,  # 额外字段确保AI不混淆
                                    "handedness_confidence": handedness_confidence,
                                    "issues": [iss],
                                    "landmarks": lm_list,
                                })

            # 追踪间隙：不在当前帧的问题 +1 gap，超过容忍值则移除
            for sig_key in list(issue_frame_tracker.keys()):
                if sig_key not in cur_frame_sigs:
                    issue_frame_tracker[sig_key]["gap"] += 1
                    if issue_frame_tracker[sig_key]["gap"] > MAX_GAP:
                        del issue_frame_tracker[sig_key]
            prev_frame_sigs = cur_frame_sigs

        # 释放手部检测用的cap，截图阶段用新cap避免seek状态污染
        cap.release()

        # ===== 截图：所有问题帧 + 全程均匀采样 =====
        issue_times = set()
        for h in hand_issue_frames:
            issue_times.add(round(h.get("timestamp", 0), 2))

        # 合并 0.5s 内的相邻时间点，避免同一问题产生多张几乎相同的截图
        if len(issue_times) > 1:
            sorted_times = sorted(issue_times)
            merged_times = []
            group = [sorted_times[0]]
            for t in sorted_times[1:]:
                if t - group[-1] <= 0.5:
                    group.append(t)
                else:
                    # 保留组内问题最严重的时间点（优先选有⚠️的帧）
                    merged_times.append(_pick_best_snapshot_time(group, hand_issue_frames))
                    group = [t]
            merged_times.append(_pick_best_snapshot_time(group, hand_issue_frames))
            issue_times = set(merged_times)

        # 均匀采样：从视频全程补充截图，确保时间覆盖
        even_times = set()
        if duration > 0:
            num_even = max(6, min(15, int(duration / 15)))
            for i in range(num_even):
                t = duration * (i + 1) / (num_even + 1)
                even_times.add(round(t, 2))

        # 所有问题时间点优先截图（精准匹配），均匀采样帧作补充
        snapshot_times = sorted(issue_times) + sorted(even_times)
        # 去重：问题帧优先保留，均匀帧作补充
        seen_st = set()
        snapshot_times_dedup = []
        for st in snapshot_times:
            if st not in seen_st:
                seen_st.add(st)
                snapshot_times_dedup.append(st)
        snapshot_times = snapshot_times_dedup[:16]

        # Build lookup: timestamp → list of (landmarks, issues)
        ts_to_data = {}
        for h in hand_issue_frames:
            ts = round(h.get("timestamp", 0), 2)
            if ts not in ts_to_data:
                ts_to_data[ts] = []
            ts_to_data[ts].append({
                "landmarks": h.get("landmarks", []),
                "issues": h.get("issues", []),
            })

        # 创建全新的 VideoCapture 用于截图，避免被手部检测循环的 seek 状态污染
        cap = cv2.VideoCapture(video_path)
        snapshots = []
        scored_frames = []  # 用于 VL 智能帧选择
        for st in snapshot_times:
            frame_idx = int(st * fps)
            if frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            # 帧定位回退1: 用毫秒定位
            if not ret or (ret and frame is not None and frame.size == 0):
                cap.set(cv2.CAP_PROP_POS_MSEC, st * 1000)
                ret, frame = cap.read()
            # 帧定位回退2: 向前微调时间偏移再试
            if not ret or frame is None or frame.size == 0:
                for offset in (0.3, -0.5, 1.0, -1.0, 2.0):
                    try_ms = max(0, (st + offset) * 1000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, try_ms)
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
            # 帧定位回退3: 从较早位置顺序读取到目标时间
            if not ret or frame is None or frame.size == 0:
                seek_start = max(0, st - 3.0)
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_start * 1000)
                for _ in range(int(3.0 * fps) + 10):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame.size > 0 and cap.get(cv2.CAP_PROP_POS_MSEC) >= st * 1000:
                        break
            if not ret or frame is None or frame.size == 0:
                continue

            h, w = frame.shape[:2]

            # 获取该时间点的问题数据用于标注
            frame_data_list = ts_to_data.get(round(st, 2), [])
            # 回退：搜索 ±0.3s 内的数据
            if not frame_data_list:
                best_key = None
                best_dist = float("inf")
                for k in ts_to_data:
                    if abs(k - round(st, 2)) < 0.3 and abs(k - round(st, 2)) < best_dist:
                        best_dist = abs(k - round(st, 2))
                        best_key = k
                if best_key is not None:
                    frame_data_list = ts_to_data[best_key]
            all_frame_issues = []
            all_frame_landmarks = []
            for fd in frame_data_list:
                all_frame_issues.extend(fd.get("issues", []))
                all_frame_landmarks.append(fd.get("landmarks", []))

            # 帧评分
            frame_score = self._score_frame(all_frame_issues)
            if frame_score["total_score"] > 0:
                scored_frames.append({
                    "timestamp": round(st, 2),
                    "total_score": frame_score["total_score"],
                    "top_issues": frame_score["top_issues"],
                })

            # 标注问题区域（已禁用红圈标注）
            # if all_frame_landmarks and all_frame_issues:
            #     frame = self._draw_hand_annotations(frame, all_frame_landmarks, all_frame_issues, w, h)

            # 缩放宽度到480px
            if w > 480:
                ratio = 480 / w
                frame_small = cv2.resize(frame, (480, int(h * ratio)))
            else:
                frame_small = frame

            import uuid
            snap_id = str(uuid.uuid4())[:8]
            snap_filename = f"snap_{snap_id}_{st:.2f}s.jpg"
            snap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, snap_filename)

            cv2.imwrite(snap_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            snapshots.append({
                "time": round(st, 2),
                "url": f"/snapshots/{snap_filename}",
                "score": frame_score["total_score"],
            })

        result["snapshots"] = snapshots

        cap.release()
        result["frames_analyzed"] = len(frame_timestamps)
        result["total_frames_extracted"] = len(frame_timestamps)

        # 给手型问题列表加上中文字段名，精准匹配截图（每个问题时间点都是截图时间点）
        clean_issues = []
        seen_issues = set()
        snap_by_time = {s["time"]: s["url"] for s in snapshots}
        for h in hand_issue_frames:
            for iss in h.get("issues", []):
                key = str(h["timestamp"]) + iss
                if key not in seen_issues:
                    seen_issues.add(key)
                    issue_ts = round(h["timestamp"], 2)
                    # 精准匹配：同时间截图直接对应
                    snap_url = snap_by_time.get(issue_ts, "")
                    # 回退：找最近的截图（±0.8s内）
                    if not snap_url:
                        best_dist = float("inf")
                        for sn in snapshots:
                            dist = abs(sn["time"] - issue_ts)
                            if dist < best_dist:
                                best_dist = dist
                                snap_url = sn["url"]
                        if best_dist > 0.8:
                            snap_url = ""
                    clean_issues.append({
                        "时间": issue_ts,
                        "哪只手": h.get("handedness", "未知"),
                        "问题": iss,
                        "置信度": h.get("handedness_confidence", 0.0),
                        "截图": snap_url,
                    })

        result["hand_issues"] = clean_issues
        result["hand_landmarks"] = all_hand_landmarks
        scored_frames.sort(key=lambda x: x["total_score"], reverse=True)
        result["scored_frames"] = scored_frames
        result["frame_analysis_data"] = json.dumps({
            "frames_checked": len(frame_timestamps),
            "hands_found": hands_detected_anywhere,
            "issues_found": len(clean_issues),
            "issues_list": clean_issues,
            "scored_frames": scored_frames,
            "instrument": instrument,
        }, ensure_ascii=False)

        logger.info(f"完成: {len(frame_timestamps)}帧, "
              f"{'有手部' if hands_detected_anywhere else '未检测到手部'}"
              f", {len(hand_issue_frames)}个问题")

        # 估算左手在指板上的大致品数（用于修正品位分配算法）
        hand_zone = self._estimate_hand_zone(wrist_x_samples, hands_detected_anywhere)
        if hand_zone is not None:
            result["hand_zone_center"] = hand_zone
            logger.info(f"手区估算: ~{hand_zone:.0f}品")

        # VL 证人信号：横按/品位区域/浮空指 三道判断题
        vl_signals = self._compute_vl_signals(all_hand_landmarks)
        result["vl_signals"] = vl_signals
        logger.info(f"VL证人: barre={vl_signals['barre']}({vl_signals['barre_conf']:.0%}) "
                    f"zone={vl_signals['zone']}({vl_signals['zone_conf']:.0%}) "
                    f"floating={vl_signals['floating']}")

        # 右手弹奏分析：拨弦角度 + 手腕运动
        right_hand = self._compute_right_hand_strumming(all_hand_landmarks)
        result["right_hand_analysis"] = right_hand
        if right_hand["frames_used"] >= 3:
            logger.info(f"右手分析: grip={right_hand['grip_angle']:.0f}° "
                        f"wrist_vel={right_hand['wrist_velocity_avg']:.4f} "
                        f"strum_score={right_hand['strumming_motion_score']:.0%}")

        return result

    def estimate_hand_zone(self, video_path: str, max_frames: int = 16) -> Optional[float]:
        """快速估算左手在指板上的大致品数。采样多帧取中位数，传给品位分配算法。

        改进（2026-07-22）：
        - 采样帧数 8→16，取至少 5 个有效样本
        - 同时使用手腕(landmark[0])和食指指尖(landmark[8])的 x 坐标
        - 指尖 x 位置对品位判断更精确（指尖直接指向品丝）

        Returns:
            估算品数（如 3.0=第3品附近），或 None 表示未能检测到手部
        """
        if not self.model_loaded:
            return None
        detector = self._create_detector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        if duration <= 0:
            cap.release()
            return None

        # Sample evenly spaced frames
        step = max(1, total_frames // (max_frames + 1))
        wrist_x_vals = []
        fingertip_x_vals = []  # index finger tip for finer estimation
        mirror_detected = False

        for frame_idx in range(step, total_frames - step, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp_image(image_format=self._mp_image_format.SRGB, data=frame_rgb)
            results = detector.detect_for_video(mp_image, int(frame_idx / fps * 1000))

            if results.hand_landmarks:
                for idx, hlm in enumerate(results.hand_landmarks):
                    handedness = "unknown"
                    if results.handedness and idx < len(results.handedness):
                        handedness = results.handedness[idx][0].category_name

                    wrist_x = hlm[0].x
                    fingertip_x = hlm[8].x  # index finger tip
                    # Auto-detect mirror: Left hand on right side → mirror
                    if not mirror_detected and handedness == "Left" and wrist_x > 0.5:
                        mirror_detected = True

                    if handedness == "Left" or (handedness == "unknown" and wrist_x < 0.5):
                        wrist_x_vals.append(wrist_x)
                        fingertip_x_vals.append(fingertip_x)

            if len(wrist_x_vals) >= 5:
                break

        cap.release()

        if not wrist_x_vals:
            return None

        median_wrist_x = sorted(wrist_x_vals)[len(wrist_x_vals) // 2]
        median_tip_x = sorted(fingertip_x_vals)[len(fingertip_x_vals) // 2]

        # Blend wrist and fingertip: fingertip is more indicative of actual fret position
        # Weight: 70% fingertip, 30% wrist (fingertip points directly at the fret)
        blended_x = median_tip_x * 0.7 + median_wrist_x * 0.3

        # Map x-coordinate to approximate fret number
        # Typical front camera: headstock on player's left
        # Non-mirror: x ~0.1 = fret 1, x ~0.8 = fret 15
        # Mirror: x ~0.9 = fret 1, x ~0.2 = fret 15
        if mirror_detected:
            est_fret = max(1.0, min(15.0, (1.0 - blended_x) * 17.0))
        else:
            est_fret = max(1.0, min(15.0, blended_x * 17.0))

        return round(est_fret, 1)

    def _estimate_hand_zone(self, wrist_x_samples: list, hands_detected: bool) -> Optional[float]:
        """从左手腕x坐标估算手在指板上的大致品数。

        Returns:
            估算品数（如 3.0 = 第3品附近），None 表示无法估算
        """
        if not wrist_x_samples or not hands_detected:
            return None
        median_x = sorted(wrist_x_samples)[len(wrist_x_samples) // 2]
        # 映射: 非镜像模式下，wrist_x 0.05~0.1 → 1品，0.8~0.9 → 15品
        if self._mirror_detected:
            est_fret = max(1.0, min(15.0, (1.0 - median_x) * 17.0))
        else:
            est_fret = max(1.0, min(15.0, median_x * 17.0))
        return round(est_fret, 1)

    def _select_keyframes(self, duration: float, fps: float, error_ts: List[float],
                          technique_segments: Optional[List[Dict]] = None) -> List[float]:
        from config import FRAME_INTERVAL_SEC, MAX_UNIFORM_KEYFRAMES, MAX_BONUS_KEYFRAMES, MAX_TOTAL_KEYFRAMES

        # Step 1: 均匀基线 — 保证全视频覆盖，不参与 bonus 上限
        uniform = set()
        if duration > 0:
            interval = max(0.15, FRAME_INTERVAL_SEC)
            ts = max(0.15, interval)
            while ts < duration:
                uniform.add(round(ts, 1))
                ts += interval
        # 超过上限时等比降采样
        if len(uniform) > MAX_UNIFORM_KEYFRAMES:
            factor = len(uniform) / MAX_UNIFORM_KEYFRAMES
            uniform = {t for i, t in enumerate(sorted(uniform)) if i % max(1, int(factor)) == 0}
            uniform = set(sorted(uniform)[:MAX_UNIFORM_KEYFRAMES])

        # Step 2: 错误 bonus（去重后计入 bonus 池）
        bonus = set()
        for et in error_ts:
            for offset in [t/10.0 for t in range(-5, 6)]:
                ts = round(et + offset, 1)
                if 0 <= ts < duration and ts not in uniform:
                    bonus.add(ts)

        # Step 3: 技巧 bonus（去重后计入 bonus 池）
        if technique_segments:
            for seg in technique_segments:
                t0 = seg.get("start_time", 0)
                t1 = seg.get("end_time", t0 + 0.5)
                if t1 - t0 < 0.1:
                    t1 = t0 + 0.5
                step = max(0.1, (t1 - t0) / 3)
                ts = t0
                while ts <= t1 and ts < duration:
                    if ts >= 0:
                        rt = round(ts, 1)
                        if rt not in uniform:
                            bonus.add(rt)
                    ts += step

        # Step 4: bonus 上限
        bonus = set(sorted(bonus)[:MAX_BONUS_KEYFRAMES])

        # Step 5: 合并
        all_ts = sorted(uniform | bonus)

        # Step 6: 绝对安全上限，优先保留均匀帧
        if len(all_ts) > MAX_TOTAL_KEYFRAMES:
            all_ts = sorted(uniform)[:MAX_TOTAL_KEYFRAMES - len(bonus)] + sorted(bonus)
            all_ts = sorted(set(all_ts))[:MAX_TOTAL_KEYFRAMES]

        logger.info(f"keyframes: uniform={len(uniform)}, bonus_err+tech={len(bonus)}, total={len(all_ts)}")
        return all_ts

    # 手指定义: (名称, MCP_index, PIP_index, DIP_index, TIP_index)
    # MediaPipe: 0=wrist, 1-4=thumb(CMC/MCP/IP/TIP), 5-8=index, 9-12=middle,
    #             13-16=ring, 17-20=pinky, 每指: MCP→PIP→DIP→TIP
    FINGER_DEFS = [
        ("thumb",  1, 2, 3, 4),
        ("index",  5, 6, 7, 8),
        ("middle", 9, 10, 11, 12),
        ("ring",   13, 14, 15, 16),
        ("pinky",  17, 18, 19, 20),
    ]

    def _calc_angles(self, lm_2d, lm_3d=None, min_confidence: float = 0.7) -> Dict:
        """计算手部各关节角度 + 指尖z深度。

        优先使用 lm_3d（MediaPipe WorldLandmarks, 真实3D米制坐标），
        回退到 lm_2d（图像归一化坐标, 2D投影会严重扭曲角度）。
        """
        # 选择3D或2D坐标源
        if lm_3d is not None and len(lm_3d) >= 21:
            lm = lm_3d
            use_3d = True
        else:
            lm = lm_2d
            use_3d = False

        angles = {}
        for name, mcp_idx, pip_idx, dip_idx, tip_idx in self.FINGER_DEFS:
            # 置信度门控：仅用2D坐标检查 MCP+PIP visibility
            mcp_vis = 1.0
            pip_vis = 1.0
            if not use_3d:
                mcp_vis = lm_2d[mcp_idx].visibility if hasattr(lm_2d[mcp_idx], 'visibility') and lm_2d[mcp_idx].visibility > 0 else 1.0
                pip_vis = lm_2d[pip_idx].visibility if hasattr(lm_2d[pip_idx], 'visibility') and lm_2d[pip_idx].visibility > 0 else 1.0
                if mcp_vis < 0.4 or pip_vis < 0.4:
                    continue

            # 用选定的坐标源计算向量
            v_wrist_mcp = np.array([lm[mcp_idx].x - lm[0].x, lm[mcp_idx].y - lm[0].y, lm[mcp_idx].z - lm[0].z])
            v_mcp_pip = np.array([lm[pip_idx].x - lm[mcp_idx].x, lm[pip_idx].y - lm[mcp_idx].y, lm[pip_idx].z - lm[mcp_idx].z])
            mcp_deg = float(np.degrees(self._vec_angle(v_wrist_mcp, v_mcp_pip)))

            v_pip_dip = np.array([lm[dip_idx].x - lm[pip_idx].x, lm[dip_idx].y - lm[pip_idx].y, lm[dip_idx].z - lm[pip_idx].z])
            pip_deg = float(np.degrees(self._vec_angle(v_mcp_pip, v_pip_dip)))

            # 透视缩短比: 2D手指全长 / 手掌宽度。
            # 手指指向镜头时2D手指极短，正常侧视时手指长度≈手掌宽度或更大。
            # palm_w 基于 index MCP (5) 到 pinky MCP (17) 的2D距离。
            mcp_pt = np.array([lm_2d[mcp_idx].x, lm_2d[mcp_idx].y])
            pip_pt = np.array([lm_2d[pip_idx].x, lm_2d[pip_idx].y])
            tip_pt = np.array([lm_2d[tip_idx].x, lm_2d[tip_idx].y])
            palm_w = np.linalg.norm([lm_2d[5].x - lm_2d[17].x, lm_2d[5].y - lm_2d[17].y])
            dist_mcp_tip = np.linalg.norm(tip_pt - mcp_pt)
            foreshorten = dist_mcp_tip / max(palm_w, 0.001)
            # foreshorten < 0.5 → 手指严重指向镜头，2D PIP 被透视反转
            # foreshorten 0.5-0.8 → 部分透视缩短，角度轻微不可靠
            # foreshorten > 0.8 → 侧视或正常角度，2D角度可靠

            # z_depth 用2D坐标（仅2D模式使用）
            palm_z = (lm_2d[0].z + lm_2d[5].z + lm_2d[17].z) / 3.0
            tip_z = lm_2d[tip_idx].z
            z_depth = round(float(palm_z - tip_z), 4)

            angles[name] = {
                "mcp_deg": round(mcp_deg, 1),
                "pip_deg": round(pip_deg, 1),
                "z_depth": z_depth,
                "is_standing": mcp_deg > 60 and z_depth > 0.005,
                "is_flat": pip_deg > 155 and mcp_deg > 160,
                "_conf": round(float((mcp_vis + pip_vis) / 2), 2),
                "_3d": use_3d,
                "foreshorten": round(float(foreshorten), 2),
            }

        # 手腕角度：用选定的坐标源
        palm_pts = [lm[i] for i in [5, 9, 13, 17]]
        palm_cx = sum(p.x for p in palm_pts) / 4
        palm_cy = sum(p.y for p in palm_pts) / 4
        palm_cz = sum(p.z for p in palm_pts) / 4
        v_hand_to_palm = np.array([palm_cx - lm[0].x, palm_cy - lm[0].y, palm_cz - lm[0].z])
        norm = np.linalg.norm(v_hand_to_palm)
        if norm > 1e-9:
            v_arm = -v_hand_to_palm / norm
        else:
            v_arm = np.array([0.0, -1.0, 0.0])  # fallback
        v_hand = np.array([lm[9].x - lm[0].x, lm[9].y - lm[0].y, lm[9].z - lm[0].z])
        wa = float(np.degrees(self._vec_angle(v_arm, v_hand)))
        angles["wrist"] = {"angle_deg": round(wa, 1), "is_bent": wa < 100 or wa > 170}
        return angles

    def _vec_angle(self, v1, v2) -> float:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0
        return float(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

    # MediaPipe hand skeleton connections
    HAND_CONNECTIONS = [
        (0,1), (1,2), (2,3), (3,4),         # thumb
        (0,5), (5,6), (6,7), (7,8),          # index
        (0,9), (9,10), (10,11), (11,12),      # middle
        (0,13), (13,14), (14,15), (15,16),    # ring
        (0,17), (17,18), (18,19), (19,20),    # pinky
        (5,9), (9,13), (13,17),               # palm
    ]

    # Finger name → landmark indices
    FINGER_LANDMARKS = {
        "手腕": [0],
        "拇指": [1, 2, 3, 4],
        "食指": [5, 6, 7, 8],
        "中指": [9, 10, 11, 12],
        "无名指": [13, 14, 15, 16],
        "小指": [17, 18, 19, 20],
    }

    def _find_problem_landmarks(self, issue_text: str) -> set:
        """Parse issue text to find which finger/body part is problematic."""
        indices = set()
        for name, lm_ids in self.FINGER_LANDMARKS.items():
            if name in issue_text:
                indices.update(lm_ids)
        # "手指" without specific finger → check all fingers
        if not indices and "手指" in issue_text:
            for name in ["食指", "中指", "无名指", "小指"]:
                indices.update(self.FINGER_LANDMARKS[name])
        return indices

    def _landmark_quality_ok(self, lm_list, finger_name: str) -> bool:
        """检查手指关键点是否可见且合理。返回 True 表示可以圈注。"""
        lm_ids = self.FINGER_LANDMARKS.get(finger_name, [])
        if not lm_ids:
            return False
        # 需要至少 2 个关键点来合理画圈（降低阈值，允许部分遮挡）
        available = sum(1 for i in lm_ids if i < len(lm_list))
        if available < 2:
            return False
        # 检查 z_depth：如果所有关键点 z 值都 > 0.08（严重偏离），可能是误跟踪
        z_vals = [lm_list[i][2] for i in lm_ids if i < len(lm_list)]
        if z_vals and all(abs(z) > 0.08 for z in z_vals):
            return False
        return True

    def _finger_has_real_issue(self, lm_list, finger_name: str) -> bool:
        """通过 landmark 坐标估算手指角度，判断是否真的有手型问题。
        用于 issue 泛称"手指"时，只圈真实有问题的手指，避免误圈。
        """
        lm_ids = self.FINGER_LANDMARKS.get(finger_name, [])
        if not lm_ids or len(lm_ids) < 4:
            return False
        try:
            # 取 MCP (lm_ids[0])、PIP (lm_ids[1])、DIP (lm_ids[2])、TIP (lm_ids[3])
            mcp = np.array([lm_list[lm_ids[0]][0], lm_list[lm_ids[0]][1], lm_list[lm_ids[0]][2]])
            pip = np.array([lm_list[lm_ids[1]][0], lm_list[lm_ids[1]][1], lm_list[lm_ids[1]][2]])
            dip = np.array([lm_list[lm_ids[2]][0], lm_list[lm_ids[2]][1], lm_list[lm_ids[2]][2]])
            tip = np.array([lm_list[lm_ids[3]][0], lm_list[lm_ids[3]][1], lm_list[lm_ids[3]][2]])
            # 计算 PIP 角度（mcp-pip-dip 三点夹角）
            v1 = mcp - pip
            v2 = dip - pip
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-8:
                return False
            pip_angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
            # 正常 PIP 120-150°，超出范围 = 有问题
            if pip_angle > 160 or pip_angle < 110:
                return True
            # 计算 MCP 角度 (pip-mcp 与水平方向的夹角近似)
            v_mcp = pip - mcp
            mcp_angle = np.degrees(np.arctan2(abs(v_mcp[2]), np.linalg.norm(v_mcp[:2])))
            if mcp_angle < 45:
                return True
            # z_depth 检查：指尖在掌面后方（塌指）
            if tip[2] - mcp[2] < -0.005:
                return True
        except (IndexError, ValueError):
            return False
        return False

    def _draw_hand_annotations(self, frame, landmarks_list, issues, img_w, img_h):
        """在截图上标注问题区域——只画红色圆圈，不画线和文字。
        对泛称"手指"的 issue，只标注实际检测到问题的手指而非全部。
        对质量不足的关键点跳过标注，避免画到错误位置。
        """
        RED = (0, 60, 255)

        # 收集所有被问题提到的手指
        problem_fingers = set()
        generic_finger = False  # issue 是否使用了泛称"手指"
        for iss in issues:
            matched = False
            for name in self.FINGER_LANDMARKS:
                if name in iss:
                    problem_fingers.add(name)
                    matched = True
            if not matched and "手指" in iss:
                generic_finger = True
            if "手腕" in iss:
                problem_fingers.add("手腕")

        for lm_list in landmarks_list:
            pts = {}
            for i, (x, y, z) in enumerate(lm_list):
                pts[i] = (int(x * img_w), int(y * img_h))

            # 当 issue 是泛称"手指"时，只标注确实有问题的手指
            if generic_finger:
                for fname in ["食指", "中指", "无名指", "小指"]:
                    if self._finger_has_real_issue(lm_list, fname):
                        problem_fingers.add(fname)

            for finger_name in problem_fingers:
                # 跳过质量不足的关键点
                if not self._landmark_quality_ok(lm_list, finger_name):
                    continue

                lm_ids = self.FINGER_LANDMARKS.get(finger_name, [])
                if not lm_ids:
                    continue

                xs, ys = [], []
                for lid in lm_ids:
                    if lid in pts:
                        xs.append(pts[lid][0])
                        ys.append(pts[lid][1])
                if not xs:
                    continue

                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                bbox_size = max(max(xs) - min(xs), max(ys) - min(ys))
                radius = min(int(bbox_size / 2) + 15, 40)

                overlay_ring = frame.copy()
                cv2.circle(overlay_ring, (cx, cy), radius, RED, 2)
                cv2.addWeighted(overlay_ring, 0.5, frame, 0.5, 0, frame)

        return frame

    # 手指英文→中文映射
    FINGER_CN = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小指"}

    def _detect_issues(self, angles: Dict, handedness: str, instrument: str) -> List[str]:
        """吉他手型检测 — 严格区分左手（按弦）和右手（弹奏）。"""
        return self._detect_issues_with_chord(angles, handedness, instrument, {})

    def _detect_issues_with_chord(self, angles: Dict, handedness: str, instrument: str,
                                    chord_angle_standards: Dict = None) -> List[str]:
        """Phase 2b: 带和弦上下文的检测。"""
        is_left = (handedness == "Left")
        is_right = (handedness == "Right")

        if is_left:
            issues = self._detect_left_hand_issues(angles, chord_angle_standards or {})
        elif is_right:
            issues = self._detect_right_hand_issues(angles)
        else:
            return []
        issues.sort(key=self._issue_severity, reverse=True)
        return issues[:6]

    @staticmethod
    def _issue_severity(issue_text: str) -> int:
        """返回 issue 的 severity 分数（越高越严重），用于排序和截断。"""
        if issue_text.startswith("⚠️"):
            return 3
        if issue_text.startswith("💡"):
            # moderate: 包含"趴""塌""太弯""太直""没完全"等具体问题描述
            if any(kw in issue_text for kw in ["趴", "塌", "太弯", "太直", "没完全", "翘"]):
                return 2
            return 1
        return 1

    @staticmethod
    def _estimate_camera_angle(landmarks_list: List) -> str:
        """通过手腕关键点相对位置判断拍摄角度。
        返回: "front" (正面), "side" (侧面), "unknown"
        """
        if not landmarks_list:
            return "unknown"
        # 取第一帧的手部数据
        lm = landmarks_list[0] if isinstance(landmarks_list[0], list) else landmarks_list
        if len(lm) < 21:
            return "unknown"
        # 手腕 x 坐标 (landmark 0)，MediaPipe 归一化到 [0,1]
        wrist_x = lm[0][0] if isinstance(lm[0], (list, tuple)) else getattr(lm[0], 'x', 0.5)
        # 中指 MCP x (landmark 9)
        mid_mcp_x = lm[9][0] if isinstance(lm[9], (list, tuple)) else getattr(lm[9], 'x', 0.5)
        # 侧面拍摄：手腕在画面边缘；正面：手腕在画面中间区域
        hand_deviation = abs(wrist_x - 0.5)
        if hand_deviation < 0.15:
            return "front"
        elif hand_deviation > 0.25:
            return "side"
        return "unknown"

    @staticmethod
    def _get_adaptive_thresholds(camera_angle: str) -> Dict:
        """根据拍摄角度返回自适应角度阈值。
        正面拍摄时 MCP 读数偏小（手指指向镜头，关节看起来更弯），需用更低的阈值。
        PIP 正面拍摄时读数偏大（手指看起来更平），需放宽阈值。
        KB 基准: MCP<40° 严重竖直, MCP<60° 偏竖直。
        """
        if camera_angle == "front":
            return {
                "pip_flat_severe": 175,
                "pip_flat_moderate": 170,
                "pip_bent_severe": 40,
                "pip_bent_moderate": 65,
                "mcp_vertical_severe": 45,   # 正面：提升灵敏度，捕捉更多竖直手指
                "mcp_vertical_moderate": 65,
                "wrist_collapse_severe": 60,
                "wrist_collapse_moderate": 80,
            }
        else:
            return {
                "pip_flat_severe": 170,
                "pip_flat_moderate": 165,
                "pip_bent_severe": 45,
                "pip_bent_moderate": 70,
                "mcp_vertical_severe": 50,   # 侧面：提升灵敏度
                "mcp_vertical_moderate": 70,
                "wrist_collapse_severe": 70,
                "wrist_collapse_moderate": 90,
            }

    def _detect_left_hand_issues(self, angles: Dict,
                                   chord_angle_standards: Dict = None) -> List[str]:
        """左手（按弦手）检测: 手指弧度、指尖站立、拇指位置、手腕弧度。

        角度约定：测量值 = 偏离直线的弯角（0°=完全伸直，90°=直角弯曲）。
        KB 内角（180°=完全伸直）在检测前转为弯角。
        """
        issues = []
        th = self._get_adaptive_thresholds(self._camera_angle)
        use_kb = chord_angle_standards and len(chord_angle_standards) > 0
        kb_tolerance = 15

        # 汇总诊断
        finger_summary = ", ".join(
            f"{self.FINGER_CN.get(f,f)}:PIP={angles[f].get('pip_deg','?')} MCP={angles[f].get('mcp_deg','?')} fs={angles[f].get('foreshorten','?')}"
            for f in ["index","middle","ring","pinky"] if f in angles
        )
        logger.info(f"[diag-L] {finger_summary or '(no fingers)'} | "
                    f"wrist={angles.get('wrist',{}).get('angle_deg','?')}° | "
                    f"use_kb={use_kb} kb_fingers={list(chord_angle_standards.keys()) if use_kb else '[]'} | "
                    f"camera={self._camera_angle}")

        # — 手腕检测 —
        if "wrist" in angles:
            w = angles["wrist"]["angle_deg"]
            if w < th["wrist_collapse_severe"]:
                issues.append("⚠️ 左手手腕太往里扣了——手腕应保持自然弧度")
            elif w < th["wrist_collapse_moderate"]:
                issues.append("💡 左手手腕有点往里扣——手腕可以再自然一点")
            elif w > 170:
                issues.append("💡 左手手腕太直了——可以稍微放松，让手腕有点自然弧度")

        # — 每指检测 —
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            if f not in angles:
                continue
            cn = self.FINGER_CN.get(f, f)
            pip = angles[f]["pip_deg"]
            mcp = angles[f]["mcp_deg"]
            fs = angles[f].get("foreshorten", 1.0)
            zd = angles[f]["z_depth"]

            # KB 转换: 内角→弯角 (180°→0°)
            kb_finger = chord_angle_standards.get(cn, {}) if use_kb else {}
            kb_pip_raw = kb_finger.get("pip_ideal")
            kb_mcp_raw = kb_finger.get("mcp_ideal")
            kb_pip_bend = (180.0 - kb_pip_raw) if kb_pip_raw is not None else None
            kb_mcp_bend = kb_mcp_raw  # MCP 已同约定

            # 拇指
            if f == "thumb":
                if mcp < 60:
                    issues.append("⚠️ 左手拇指捏得太紧或者位置太低了——拇指自然搭在琴颈上方就好，不用捏")
                elif mcp > 160:
                    issues.append("💡 左手拇指太直了——拇指保持自然微弯，自然搭在琴颈上方")
                continue

            # 小指
            if f == "pinky":
                if pip > 170:
                    issues.append(f"⚠️ 左手{cn}翘得太高了——小指靠近指板，随时准备帮忙")
                elif pip > 155:
                    issues.append(f"💡 左手{cn}有点飞出去了——小指靠近指板一点")
                elif pip < 40 and pip > 15:
                    issues.append(f"💡 左手{cn}太紧张了——小指放松一点")
                if zd < -0.008:
                    issues.append(f"⚠️ 左手{cn}塌下去了——指尖立起来")
                elif zd < 0:
                    issues.append(f"💡 左手{cn}还不够立——用指尖，不要用指腹")
                continue

            # === index / middle / ring ===
            # 将内角 pip (180°=直, 小=弯) 转为弯角 (0°=直, 大=弯)，与 KB 的 kb_pip_bend 对齐
            pip_bend = 180.0 - pip

            # 严重透视缩短 (fs<0.3): 手指直指镜头，2D角度不可靠
            if fs < 0.3:
                if pip > 175:  # 内角极大→手指笔直像筷子
                    issues.append(f"⚠️ 左手{cn}太竖直了——像筷子一样直戳在弦上，手指应该像握着鸡蛋一样自然弯曲")
                logger.info(f"[diag] {cn}: pip={pip:.0f} mcp={mcp:.0f} fs={fs:.1f} (foreshortened, skipped)")
                continue

            # === KB 驱动检测 ===
            if kb_pip_bend is not None:
                diff = kb_pip_bend - pip_bend  # positive = straighter than ideal
                if diff > 28:
                    issues.append(f"⚠️ 左手{cn}太竖直了——像筷子一样直戳在弦上，手指应该像握着鸡蛋一样自然弯曲")
                elif diff > 15:
                    issues.append(f"💡 左手{cn}有点太直——放松指根，让手指像握着鸡蛋一样有点自然弧度")
                elif diff < -50:
                    issues.append(f"⚠️ 左手{cn}太弯了——手指太紧张像握拳，放松指根让弧度更自然")
                elif diff < -35:
                    issues.append(f"💡 左手{cn}弯得有点多——手指弧度稍微放松一点")
            else:
                # === 通用检测（无KB数据时） ===
                # 注意：手指竖直站立时 PIP 也会偏小，不能误判为「太弯」
                # 只有当 MCP 不竖直（mcp >= moderate 阈值）时，PIP 偏小才是真正的蜷指
                mcp_not_vertical = mcp >= th["mcp_vertical_moderate"]
                if pip > 170:
                    issues.append(f"⚠️ 左手{cn}太竖直了——像筷子一样直戳在弦上，手指应该像握着鸡蛋一样自然弯曲")
                elif pip > 155 and mcp > 140:
                    issues.append(f"💡 左手{cn}有点太直——手指放松，想象轻轻握着一个乒乓球")
                elif pip < 80 and mcp_not_vertical:
                    issues.append(f"⚠️ 左手{cn}太弯了——手指蜷得太紧像握拳，放松指根让手指自然延展")
                elif pip < 105 and mcp_not_vertical:
                    issues.append(f"💡 左手{cn}弯得有点多——手指弧度稍微放松一点")

            # === MCP 检测（内角：小=弯折、大=伸直，kb_mcp_raw 已是内角） ===
            # MCP过度弯曲 → 手指竖直戳弦（与KB「手指过于竖直」触发条件MCP<40°对齐）
            if kb_mcp_bend is not None:
                mcp_diff = mcp - kb_mcp_bend  # positive = straighter than ideal
                if mcp_diff > 35:
                    issues.append(f"💡 左手{cn}指根太直了——指根关节应该自然弯曲，像在琴颈上搭了个小拱桥")
                elif mcp_diff < -30:
                    issues.append(f"⚠️ 左手{cn}太竖直了——指根过度弯曲让手指像筷子直戳在弦上，放松指根让手指有点自然弧度")
                elif mcp_diff < -15:
                    issues.append(f"💡 左手{cn}有点偏竖直——指根弯得偏多，手指站得太直，试着让指根自然一些")
            else:
                if mcp < th["mcp_vertical_severe"]:
                    issues.append(f"⚠️ 左手{cn}太竖直了——指根过度弯曲让手指像筷子直戳在弦上，放松指根让手指有点自然弧度")
                elif mcp < th["mcp_vertical_moderate"]:
                    issues.append(f"💡 左手{cn}有点偏竖直——指根弯得偏多，手指站得太直，试着让指根自然一些")
                elif mcp > 165:
                    issues.append(f"💡 左手{cn}指根太直了——放松指根关节，让它微微弯曲")

            logger.info(f"[diag] {cn}: pip={pip:.0f} mcp={mcp:.0f} fs={fs:.1f}"
                        + (f" kb_pip_bend={kb_pip_bend:.0f} kb_mcp_bend={kb_mcp_bend:.0f}" if kb_pip_bend is not None else ""))

            # z_depth 检查（Phase 2c: 弱化，仅极端情况报告）
            if not use_kb:
                if zd < -0.015:
                    issues.append(f"⚠️ 左手{cn}指尖塌下去了——手型塌陷，手指立起来")
                elif zd < -0.012:
                    issues.append(f"💡 左手{cn}指尖没完全立起来——再立起来一点点")

        issues.sort(key=self._issue_severity, reverse=True)
        return issues[:6]

    def _detect_right_hand_issues(self, angles: Dict) -> List[str]:
        """右手（弹奏手）检测: 手腕灵活性、手指自然度、拇指位置、手部位置。

        右手负责拨弦/扫弦/闷音/泛音等，手指不需要像左手那样「立起来按弦」。
        右手的问题是: 手腕太僵硬/太弯、手指太直/蜷太紧、拇指捏太紧、
        手离音孔太远、整体紧张。

        生物力学参考（吉他弹奏场景）:
        - 手腕正常范围: 100-175°（扫弦时大幅度摆动正常）
        - 手指 PIP 正常范围: 100-160°（自然微弯，拨弦用指尖）
        - 拇指 MCP 正常范围: 60-150°（自然握拨片或搭弦）
        """
        issues = []
        th = self._get_adaptive_thresholds(self._camera_angle)

        # — 手腕检测（右手手腕灵活性是关键）—
        if "wrist" in angles:
            w = angles["wrist"]["angle_deg"]
            # 手腕太弯：如果在做PM闷音是正常的，否则太紧张
            if w < 50:
                issues.append("⚠️ 右手手腕弯得太厉害了——如果在用掌根做PM闷音就正常，否则手腕太紧张，像甩水一样自然放松")
            elif w < 75:
                issues.append("💡 右手手腕有点往里弯——如果是在闷音或靠弦就正常，平时弹分解和弦手腕放松一点更灵活")
            # 手腕接近180°（直）对右手弹奏是正常默认姿态，不报问题
            # 右手手腕灵活性应通过角度变化来评估，而非单帧绝对值

        # — 每指检测 —
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            if f not in angles:
                continue
            cn = self.FINGER_CN.get(f, f)
            pip = angles[f]["pip_deg"]
            mcp = angles[f]["mcp_deg"]

            # 拇指检测：握拨片/自然搭弦，重点是不过度紧张
            if f == "thumb":
                if mcp < 40:
                    issues.append("⚠️ 右手拇指捏得太紧了——握拨片或拨弦时拇指自然放松，像拿着一张纸的力度就好")
                elif mcp < 60:
                    issues.append("💡 右手拇指有点紧——试试放松一点，拇指微微弯曲就好，不用使劲捏")
                elif mcp > 170:
                    issues.append("💡 右手拇指太直了——保持自然微弯，像轻轻握住一支笔")
                continue

            # index / middle / ring / pinky
            # 右手手指主要用于拨弦，需要自然微弯
            # 蜷太紧 → 紧张，影响灵活度
            if pip < 50:
                issues.append(f"⚠️ 右手{cn}蜷得太紧了——拨弦手指放松一点，自然弯曲、像在键盘上打字那样")
            elif pip < 75:
                issues.append(f"💡 右手{cn}有点紧张——如果是AM指弹技巧就正常；如果不是，手指可以再放松一点")

            # 太直 → 可能是用指腹拨弦而不是指尖，或者整体紧张
            if pip > 170:
                issues.append(f"⚠️ 右手{cn}完全伸直了——拨弦时手指保持自然微弯，用指尖拨弦声音更清晰")
            elif pip > 155:
                issues.append(f"💡 右手{cn}有点太直了——拨弦时手指自然微弯，不要完全伸直")

            # 组合检测：PIP+MCP都很直 → 手指僵硬
            if pip > 150 and mcp > 150:
                if not any("太直" in i or "伸直" in i for i in issues):
                    issues.append(f"💡 右手{cn}看起来有点僵硬——弹琴时手指应该像在水中轻轻划动，保持自然的弧度")

            logger.info(f"[diag right] {cn}: pip={pip:.0f} mcp={mcp:.0f}")

        # 按 severity 排序
        issues.sort(key=self._issue_severity, reverse=True)
        return issues[:6]

    def _compute_vl_signals(self, all_hand_landmarks: List[Dict]) -> Dict:
        """VL 证人模式：从累积的手部 landmarks 中提取三个判断题信号。

        只输出 yes/no 判断，不做和弦识别：
        1. barre（横按）: 食指是否横跨多根弦？
        2. zone（品位区域）: 手在低把位(1-4品)/中把位(5-9品)/高把位(10+品)？
        3. floating（浮空指）: 是否有手指未按实（伸得太直）？

        Returns:
            {"barre": bool, "barre_conf": 0.0-1.0,
             "zone": "low"|"mid"|"high", "zone_conf": 0.0-1.0,
             "floating": ["index"|"middle"|"ring"|"pinky", ...], "floating_conf": 0.0-1.0,
             "frames_used": N}
        """
        FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
        MCP_IDX = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
        PIP_IDX = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
        TIP_IDX = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

        left_frames = [lm for lm in all_hand_landmarks if lm.get("hand") == "Left"]
        if len(left_frames) < 3:
            return {"barre": False, "barre_conf": 0, "zone": "low", "zone_conf": 0,
                    "floating": [], "floating_conf": 0, "frames_used": len(left_frames)}

        barre_votes = 0
        zone_x_vals = []
        floating_counts = {"index": 0, "middle": 0, "ring": 0, "pinky": 0}
        total_valid = 0

        for frame_data in left_frames:
            lm_list = frame_data.get("landmarks", [])
            if len(lm_list) < 21:
                continue
            total_valid += 1

            # ── Barre 检测：食指伸直横跨，其他手指正常弯曲 ──
            idx_mcp = np.array(lm_list[5])
            idx_pip = np.array(lm_list[6])
            idx_tip = np.array(lm_list[8])
            idx_flatness = self._finger_extension_angle(idx_mcp, idx_pip, idx_tip)
            # 食指较直（>150°）且其他手指弯曲 → barre 特征
            other_curled = 0
            for fn in ["middle", "ring", "pinky"]:
                mcp = np.array(lm_list[MCP_IDX[fn]])
                pip = np.array(lm_list[PIP_IDX[fn]])
                tip = np.array(lm_list[TIP_IDX[fn]])
                ext = self._finger_extension_angle(mcp, pip, tip)
                if ext < 130:  # 明显弯曲
                    other_curled += 1
            if idx_flatness > 145 and other_curled >= 2:
                barre_votes += 1

            # ── Zone 检测：手腕 x 坐标 → 品位 ──
            zone_x_vals.append(lm_list[0][0])  # wrist x

            # ── Floating 检测：手指太直 → 未按弦 ──
            for fn in ["index", "middle", "ring", "pinky"]:
                mcp = np.array(lm_list[MCP_IDX[fn]])
                pip = np.array(lm_list[PIP_IDX[fn]])
                tip = np.array(lm_list[TIP_IDX[fn]])
                ext = self._finger_extension_angle(mcp, pip, tip)
                if ext > 150:
                    floating_counts[fn] += 1

        if total_valid == 0:
            return {"barre": False, "barre_conf": 0, "zone": "low", "zone_conf": 0,
                    "floating": [], "floating_conf": 0, "frames_used": 0}

        barre_conf = barre_votes / total_valid
        barre = barre_conf >= 0.5

        # Zone: 根据手腕 x 中位数判断。x<0.35=低把位(镜面则>0.65)
        med_x = float(np.median(zone_x_vals)) if zone_x_vals else 0.5
        if self._mirror_detected:
            med_x = 1.0 - med_x
        if med_x < 0.33:
            zone, zone_conf = "low", min(1.0, (0.33 - med_x) / 0.15 + 0.5)
        elif med_x < 0.55:
            zone, zone_conf = "mid", 0.7
        else:
            zone, zone_conf = "high", min(1.0, (med_x - 0.55) / 0.15 + 0.5)

        # Floating: 超过半数的帧中出现即为浮空
        floating = [fn for fn, cnt in floating_counts.items()
                    if cnt > total_valid * 0.5]

        return {
            "barre": barre, "barre_conf": round(barre_conf, 2),
            "zone": zone, "zone_conf": round(zone_conf, 2),
            "floating": floating, "floating_conf": round(max(floating_counts.values()) / max(total_valid, 1), 2),
            "frames_used": total_valid,
        }

    @staticmethod
    def _finger_extension_angle(mcp, pip, tip) -> float:
        """计算手指伸直程度：MCP→PIP→TIP 三点的夹角，180°=完全伸直。"""
        v1 = pip[:2] - mcp[:2]
        v2 = tip[:2] - pip[:2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 180.0
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def _compute_right_hand_strumming(self, all_hand_landmarks: List[Dict]) -> Dict:
        """右手弹奏分析：拨弦角度 + 手腕运动速度 + 逐次扫弦方向/力度。

        从累积的右手 landmarks 中提取：
        1. pick_grip_angle: 拇指尖到食指尖的角度（握拨片姿态）
        2. wrist_velocity: 手腕帧间运动速度（扫弦时会有规律的摆动）
        3. strumming_motion_score: 手腕运动的规律性
        4. strum_events: 每次扫弦的方向+速度（AirStrum风格：右手腕帧间位移→方向+力度）

        Returns:
            {"grip_angle": float, "wrist_velocity_avg": float,
             "strumming_motion_score": 0.0-1.0, "frames_used": N,
             "strum_events": [{"time": float, "direction": "up"|"down",
                               "velocity": float, "duration": float}],
             "strum_count": int, "down_count": int, "up_count": int}
        """
        right_frames = [lm for lm in all_hand_landmarks if lm.get("hand") == "Right"]
        if len(right_frames) < 3:
            return {"grip_angle": 0, "wrist_velocity_avg": 0,
                    "strumming_motion_score": 0, "frames_used": len(right_frames),
                    "strum_events": [], "strum_count": 0,
                    "down_count": 0, "up_count": 0}

        grip_angles = []
        wrist_positions = []  # (time, x, y)

        for frame_data in right_frames:
            lm_list = frame_data.get("landmarks", [])
            if len(lm_list) < 21:
                continue
            t = frame_data.get("time", 0)

            # 拇指尖(4)到食指尖(8)的角度 → 握拨片姿态
            thumb_tip = np.array(lm_list[4][:2])
            index_tip = np.array(lm_list[8][:2])
            wrist = np.array(lm_list[0][:2])
            v_thumb = thumb_tip - wrist
            v_index = index_tip - wrist
            n_t = np.linalg.norm(v_thumb)
            n_i = np.linalg.norm(v_index)
            if n_t > 1e-9 and n_i > 1e-9:
                cos_a = np.clip(np.dot(v_thumb, v_index) / (n_t * n_i), -1.0, 1.0)
                grip_angles.append(float(np.degrees(np.arccos(cos_a))))

            wrist_positions.append((t, lm_list[0][0], lm_list[0][1]))

        # 手腕帧间速度（标量）
        velocities = []
        for i in range(1, len(wrist_positions)):
            t0, x0, y0 = wrist_positions[i - 1]
            t1, x1, y1 = wrist_positions[i]
            dt = t1 - t0
            if dt > 0.01:
                dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                velocities.append(dist / dt)

        avg_grip = float(np.mean(grip_angles)) if grip_angles else 0
        avg_vel = float(np.mean(velocities)) if velocities else 0

        # 扫弦运动评分：有规律的速度波动（CV适中=规律，太低=静止，太高=乱动）
        strum_score = 0.0
        if len(velocities) >= 4:
            vel_arr = np.array(velocities)
            vel_cv = float(np.std(vel_arr) / max(np.mean(vel_arr), 0.001))
            if 0.2 < vel_cv < 1.2 and avg_vel > 0.02:
                strum_score = min(1.0, vel_cv * 0.8)
            elif avg_vel > 0.05:
                strum_score = 0.5

        # ── 逐次扫弦检测：AirStrum 风格右手腕 y-velocity 方向判断 ──
        strum_events = self._detect_individual_strums(wrist_positions)

        return {
            "grip_angle": round(avg_grip, 1),
            "wrist_velocity_avg": round(avg_vel, 4),
            "strumming_motion_score": round(strum_score, 2),
            "frames_used": len(right_frames),
            "strum_events": strum_events,
            "strum_count": len(strum_events),
            "down_count": sum(1 for s in strum_events if s["direction"] == "down"),
            "up_count": sum(1 for s in strum_events if s["direction"] == "up"),
        }

    def _detect_individual_strums(self, wrist_positions: List[tuple]) -> List[Dict]:
        """从手腕 y 坐标时间序列中检测逐次扫弦事件。

        AirStrum 风格：右手腕帧间 y 位移 → 方向+速度。
        y 增加 = 手腕下移 = down strum（右撇子吉他手面对摄像头）
        y 减少 = 手腕上移 = up strum

        零交叉检测 + 滞后阈值，对抗 MediaPipe landmark 抖动。
        """
        if len(wrist_positions) < 5:
            return []

        # 提取 y 序列 + 简单移动平均平滑（窗口=3）
        ys = np.array([y for _, _, y in wrist_positions])
        ts = np.array([t for t, _, _ in wrist_positions])
        if len(ys) >= 3:
            kernel = np.ones(3) / 3
            ys_smooth = np.convolve(ys, kernel, mode='same')
            ys_smooth[:1] = ys[:1]
            ys_smooth[-1:] = ys[-1:]
        else:
            ys_smooth = ys

        # 帧间 y 速度（归一化坐标/秒）
        y_velocities = []
        vel_times = []
        for i in range(1, len(ys_smooth)):
            dt = ts[i] - ts[i - 1]
            if dt > 0.005:
                vy = (ys_smooth[i] - ys_smooth[i - 1]) / dt
                y_velocities.append(vy)
                vel_times.append(ts[i])

        if len(y_velocities) < 3:
            return []

        vy_arr = np.array(y_velocities)
        vt_arr = np.array(vel_times)

        # 全局统计确定阈值
        abs_vy = np.abs(vy_arr)
        median_abs = float(np.median(abs_vy)) if len(abs_vy) > 0 else 0.01
        noise_floor = max(median_abs * 0.5, 0.003)  # 噪声地板：中值的50%或0.003

        # 零交叉检测 → 分割扫弦事件
        # 符号：正=down(y增加)，负=up(y减少)
        events = []
        in_strum = False
        strum_start_idx = 0
        prev_sign = 0

        for i in range(len(vy_arr)):
            v = vy_arr[i]
            sign = 1 if v > noise_floor else (-1 if v < -noise_floor else 0)

            if not in_strum:
                if sign != 0:
                    in_strum = True
                    strum_start_idx = i
                    prev_sign = sign
            else:
                # 方向反转（过零）→ 结束当前扫弦，开始新的
                if sign != 0 and sign != prev_sign:
                    events.append((strum_start_idx, i - 1, prev_sign))
                    strum_start_idx = i
                    prev_sign = sign
                elif sign == 0:
                    # 速度降到噪声以下 → 扫弦结束
                    # 检查是否持续静止（≥2帧静止才算真的结束）
                    still_count = 0
                    for j in range(i, min(i + 3, len(vy_arr))):
                        if abs(vy_arr[j]) <= noise_floor:
                            still_count += 1
                        else:
                            break
                    if still_count >= 2:
                        events.append((strum_start_idx, i - 1, prev_sign))
                        in_strum = False
                        prev_sign = 0

        # 收尾：如果最后还在扫弦中
        if in_strum:
            events.append((strum_start_idx, len(vy_arr) - 1, prev_sign))

        # 将原始段转换为 strum_events
        strum_events = []
        for start_idx, end_idx, direction_sign in events:
            if end_idx <= start_idx:
                continue

            seg_vy = vy_arr[start_idx:end_idx + 1]
            seg_t = vt_arr[start_idx:end_idx + 1]

            # 累计 y 位移
            total_dy = float(np.sum(seg_vy * np.diff(np.concatenate([[vt_arr[start_idx - 1] if start_idx > 0 else vt_arr[0]], seg_t]))))

            # 过滤：位移太小不是真扫弦
            min_displacement = max(noise_floor * 0.08, 0.002)
            if abs(total_dy) < min_displacement:
                continue

            direction = "down" if direction_sign > 0 else "up"
            mean_velocity = float(np.mean(np.abs(seg_vy)))
            peak_velocity = float(np.max(np.abs(seg_vy)))
            duration = float(seg_t[-1] - seg_t[0])

            # 过滤：持续时间太短（< 40ms）通常是噪音或过渡帧
            if duration < 0.04:
                continue

            mid_time = float(seg_t[0] + duration / 2)

            strum_events.append({
                "time": round(mid_time, 3),
                "direction": direction,
                "velocity": round(mean_velocity, 4),
                "peak_velocity": round(peak_velocity, 4),
                "duration": round(duration, 3),
            })

        return strum_events

    def _score_frame(self, issues: List[str]) -> Dict:
        """对单帧的手型问题打分，用于 VL 智能帧选择。
        severe 问题 = 3分, moderate 问题 = 2分, mild 问题 = 1分。
        """
        score = 0
        sev_map = {"⚠️": 3, "💡": 1}
        top = []
        for iss in issues:
            prefix = iss[:2] if len(iss) >= 2 else ""
            s = sev_map.get(prefix, 1)
            # 区分 moderate: 有具体数值但未到 severe
            if prefix == "💡":
                # 偏平 PIP>150 = moderate(2), 纯建议 = mild(1)
                if "°" in iss or "z=" in iss:
                    s = 2
            score += s
            top.append({"issue": iss, "score": s})
        top.sort(key=lambda x: x["score"], reverse=True)
        return {"total_score": score, "top_issues": top}

    def extract_landmark_vector(self, image_path: str) -> Optional[List[float]]:
        """从单张图片提取手部关键点向量 (63维: 21点 × 3坐标, 归一化)。
        用于 RAG 知识库的向量相似度搜索。返回 None 表示未检测到手部。
        """
        if not self.model_loaded:
            return None
        detector = self._create_detector()
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image(image_format=self._mp_image_format.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, 0)
        if not results.hand_landmarks:
            return None
        hlm = results.hand_landmarks[0]
        vec = []
        for lm in hlm:
            vec.extend([lm.x, lm.y, lm.z])
        wrist = np.array([hlm[0].x, hlm[0].y, hlm[0].z])
        arr = np.array(vec, dtype=np.float64).reshape(21, 3)
        arr = arr - wrist
        mid_mcp = arr[9]
        scale = np.linalg.norm(mid_mcp)
        if scale > 0.01:
            arr = arr / scale
        return arr.flatten().tolist()

    def landmark_list_to_vector(self, lm_list_3d: List) -> Optional[List[float]]:
        """将已提取的单帧landmark列表转为63维归一化向量。

        Args:
            lm_list_3d: 21个关键点，每个为 [x, y, z] 或 (x, y, z)，图像坐标

        Returns:
            63维归一化向量，或 None（数据不足时）
        """
        if len(lm_list_3d) < 21:
            return None
        vec = []
        for lm in lm_list_3d[:21]:
            if len(lm) >= 3:
                vec.extend([float(lm[0]), float(lm[1]), float(lm[2])])
            elif len(lm) >= 2:
                vec.extend([float(lm[0]), float(lm[1]), 0.0])
            else:
                return None
        wrist = np.array([float(lm_list_3d[0][0]), float(lm_list_3d[0][1]),
                          float(lm_list_3d[0][2]) if len(lm_list_3d[0]) >= 3 else 0.0])
        arr = np.array(vec, dtype=np.float64).reshape(21, 3)
        arr = arr - wrist
        mid_mcp = arr[9]
        scale = np.linalg.norm(mid_mcp)
        if scale > 0.01:
            arr = arr / scale
        return arr.flatten().tolist()





