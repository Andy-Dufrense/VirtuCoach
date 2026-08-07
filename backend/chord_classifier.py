"""基于手部关键点向量的和弦分类器（AirStrum风格：MediaPipe landmark → 和弦约束）。

架构（2026-08-05 重构）：
- 视觉(landmark 几何特征) → 横按检测（食指伸直度, 84%准确率）
- 视觉(landmark k-NN) → 手型家族辅助信号
- 音频(chroma) → 和弦性质（大三/小三/属七/大七）
- 横按信号作为硬约束：barre → 候选限定 {F型,B型}, open → 候选限定 {C,A,D,E,G}
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# CAGED 手型家族：同家族内手指位置几乎相同，视觉效果难以区分
# 家族分类由视觉负责，家族内具体和弦由音频 chroma 区分
CHORD_TO_FAMILY = {
    # C 型手型：食指②弦1品 + 无名指⑤弦3品（或食指③弦1品）
    "chord-C": "C_SHAPE",
    "chord-C7": "C_SHAPE",
    "chord-Cmaj7": "C_SHAPE",
    "chord-C-major": "C_SHAPE",
    # A 型手型：食中无名指集中在④③②弦2品
    "chord-A": "A_SHAPE",
    "chord-Am": "A_SHAPE",
    "chord-A7": "A_SHAPE",
    "chord-Am7": "A_SHAPE",
    "chord-A-major": "A_SHAPE",
    # D 型手型：三角形——三指在③②①弦
    "chord-D": "D_SHAPE",
    "chord-Dm": "D_SHAPE",
    "chord-D7": "D_SHAPE",
    "chord-Dm7": "D_SHAPE",
    "chord-D-major": "D_SHAPE",
    # E 型手型：中指⑤弦2品 + 无名指④弦2品 + 食指③弦1品
    "chord-E": "E_SHAPE",
    "chord-Em": "E_SHAPE",
    "chord-E7": "E_SHAPE",
    "chord-E-major": "E_SHAPE",
    # G 型手型：中指⑥弦3品 + 食指⑤弦2品 + 无名指①弦3品
    "chord-G": "G_SHAPE",
    "chord-G7": "G_SHAPE",
    "chord-G-major": "G_SHAPE",
    # F 横按：E 型手型在 1 品横按
    "chord-F": "F_BARRE",
    "chord-fmaj7": "F_BARRE",
    "chord-F-major": "F_BARRE",
    # B 横按：A 型/Am 型手型在第 2 品横按
    "chord-B": "B_BARRE",
    "chord-B7": "B_BARRE",
    "chord-Bm": "B_BARRE",
    "chord-Bm7": "B_BARRE",
    "chord-B-major": "B_BARRE",
    # 特殊：D/F# 拇指扣或修改 D 型
    "chord-D-over-Fsharp": "D_FSHARP",
}

# 手型家族 → 包含的具体和弦（用于音频交叉验证时缩小候选范围）
FAMILY_MEMBERS = {
    "C_SHAPE": ["chord-C", "chord-C7", "chord-Cmaj7"],
    "A_SHAPE": ["chord-A", "chord-Am", "chord-A7", "chord-Am7"],
    "D_SHAPE": ["chord-D", "chord-Dm", "chord-D7", "chord-Dm7"],
    "E_SHAPE": ["chord-E", "chord-Em", "chord-E7"],
    "G_SHAPE": ["chord-G", "chord-G7"],
    "F_BARRE": ["chord-F", "chord-fmaj7"],
    "B_BARRE": ["chord-B", "chord-B7", "chord-Bm", "chord-Bm7"],
    "D_FSHARP": ["chord-D-over-Fsharp"],
}

FAMILY_NAMES = {
    "C_SHAPE": "C型手型",
    "A_SHAPE": "A型手型",
    "D_SHAPE": "D型手型",
    "E_SHAPE": "E型手型",
    "G_SHAPE": "G型手型",
    "F_BARRE": "F横按(E型1品)",
    "B_BARRE": "B横按(A型2品)",
    "D_FSHARP": "D/F#(特殊)",
}


class LandmarkChordClassifier:
    """从手部21点landmark预测手型家族（CAGED + 横按），而非具体和弦。

    视觉负责粗略分类（8家族），音频负责家族内精细区分（大三/小三/属七）。
    """

    def __init__(self, reference_db=None):
        self._reference_vectors: List[Tuple[np.ndarray, str, str]] = []
        self._family_names: Dict[str, str] = dict(FAMILY_NAMES)
        self._ready = False
        if reference_db is not None:
            self.load_references(reference_db)

    def load_references(self, reference_db) -> int:
        """从参考图数据库加载已标注和弦的landmark向量，映射到手型家族。"""
        all_refs = reference_db.list_all()
        loaded = 0
        family_counts: Dict[str, int] = {}

        for ref in all_refs:
            vec_json = ref.get("landmark_vector")
            if not vec_json:
                continue
            chord_tags = [t for t in (ref.get("tags") or []) if t.startswith("chord-")]
            if not chord_tags:
                continue
            chord_id = chord_tags[0]
            family = CHORD_TO_FAMILY.get(chord_id)
            if not family:
                continue

            try:
                vec = np.array(json.loads(vec_json) if isinstance(vec_json, str) else vec_json,
                               dtype=np.float64)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if len(vec) != 63:
                continue
            v_norm = np.linalg.norm(vec)
            if v_norm > 1e-10:
                vec = vec / v_norm

            self._reference_vectors.append((vec, family, FAMILY_NAMES.get(family, family)))
            family_counts[family] = family_counts.get(family, 0) + 1
            loaded += 1

        self._ready = loaded > 0
        logger.info(f"和弦分类器(家族模式) 加载了 {loaded} 个参考向量 "
                    f"→ {len(family_counts)} 个手型家族: "
                    f"{', '.join(f'{k}({v})' for k, v in sorted(family_counts.items()))}")
        return loaded

    def predict_family(self, landmark_vector: List[float], top_k: int = 3,
                       min_similarity: float = 0.70) -> List[Dict]:
        """预测手型家族（8选1或Top-K）。

        Args:
            landmark_vector: 63维手部关键点向量（需预先归一化）
            top_k: 返回Top-K候选家族
            min_similarity: 最低余弦相似度阈值

        Returns:
            [{"family": "C_SHAPE", "family_name": "C型手型",
              "candidate_chords": ["chord-C","chord-C7","chord-Cmaj7"],
              "confidence": 0.91}, ...]
        """
        if not self._ready:
            return []

        query = np.array(landmark_vector, dtype=np.float64)
        q_norm = np.linalg.norm(query)
        if q_norm > 1e-10:
            query = query / q_norm

        # 按家族聚合相似度（取每个家族内最高相似度的参考图）
        family_best: Dict[str, Tuple[float, str]] = {}
        for ref_vec, family, family_name in self._reference_vectors:
            sim = float(np.dot(query, ref_vec))
            if sim >= min_similarity:
                if family not in family_best or sim > family_best[family][0]:
                    family_best[family] = (sim, family_name)

        scored = sorted(family_best.items(), key=lambda x: x[1][0], reverse=True)

        results = []
        for family, (sim, fname) in scored[:top_k]:
            results.append({
                "family": family,
                "family_name": fname,
                "candidate_chords": FAMILY_MEMBERS.get(family, []),
                "confidence": round(sim, 4),
            })
        return results

    def predict_top_family(self, landmark_vector: List[float],
                           min_similarity: float = 0.70) -> Optional[Dict]:
        """返回最可能的手型家族。"""
        results = self.predict_family(landmark_vector, top_k=1, min_similarity=min_similarity)
        return results[0] if results else None

    def detect_barre(self, landmark_vector: List[float]) -> Dict:
        """检测手型是否为横按（食指伸直覆盖多弦）。

        基于食指 PIP 夹角：MCP→PIP→TIP 三点夹角。
        - 0° = 完全伸直（典型横按）
        - 90°+ = 弯曲（开放和弦按弦）
        - 阈值 < 22° → 判定为横按

        在25张参考图上验证：84%准确率（LOO logistic regression 76%）。

        Returns:
            {"is_barre": bool, "barre_score": 0.0-1.0,
             "index_angle": float(°), "detail": str}
        """
        if len(landmark_vector) < 63:
            return {"is_barre": False, "barre_score": 0.0,
                    "index_angle": 0, "detail": "数据不足"}

        arr = np.array(landmark_vector, dtype=np.float64).reshape(21, 3)
        mcp = arr[5]   # index MCP
        pip = arr[6]   # index PIP
        tip = arr[8]   # index TIP
        v1 = pip - mcp
        v2 = tip - pip
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return {"is_barre": False, "barre_score": 0.0,
                    "index_angle": 0, "detail": "关键点无效"}

        angle = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

        # 阈值: < 22° = 横按, 22-40° = 模糊区, > 40° = 开放
        BARRE_THRESHOLD = 22.0
        SOFT_MAX = 50.0

        if angle < BARRE_THRESHOLD:
            is_barre = True
            score = 1.0 - (angle / BARRE_THRESHOLD) * 0.3
        elif angle < BARRE_THRESHOLD * 1.8:
            is_barre = False
            score = 1.0 - (angle - BARRE_THRESHOLD) / (BARRE_THRESHOLD * 0.8)
            score = max(0.3, score)
        else:
            is_barre = False
            score = max(0.0, 1.0 - angle / SOFT_MAX)

        return {
            "is_barre": is_barre,
            "barre_score": round(min(1.0, score), 3),
            "index_angle": round(angle, 1),
            "detail": (
                f"食指伸直({angle:.0f}°)→横按" if is_barre
                else f"食指弯曲({angle:.0f}°)→开放"
            ),
        }

    def constraint_chord_candidates(self, landmark_vector: List[float],
                                    audio_chord_id: str) -> Dict:
        """用视觉横按检测约束音频和弦候选范围。

        横按检测84%准确率 → 作为硬约束：
        - 检测到横按 → 候选限定 {F, Fmaj7, B, B7, Bm, Bm7}
        - 未检测到横按 → 候选排除横按和弦
        - 与音频结果矛盾时标记冲突

        Returns:
            {"constraint": "barre"|"open"|"uncertain",
             "audio_chord": str,
             "consistent": bool,
             "detail": str}
        """
        barre = self.detect_barre(landmark_vector)
        is_barre = barre["is_barre"]
        score = barre["barre_score"]

        BARRE_CHORDS = {"chord-F", "chord-fmaj7", "chord-B", "chord-B7", "chord-Bm", "chord-Bm7"}
        audio_is_barre = audio_chord_id in BARRE_CHORDS

        if score < 0.5:
            return {"constraint": "uncertain", "audio_chord": audio_chord_id,
                    "consistent": True, "barre": barre,
                    "detail": f"视觉横按置信度不足({score:.2f})，不约束"}

        if is_barre and audio_is_barre:
            return {"constraint": "barre", "audio_chord": audio_chord_id,
                    "consistent": True, "barre": barre,
                    "detail": f"视觉横按确认→{audio_chord_id}在横按候选内"}
        elif is_barre and not audio_is_barre:
            return {"constraint": "barre", "audio_chord": audio_chord_id,
                    "consistent": False, "barre": barre,
                    "detail": f"视觉横按({score:.0%}) vs 音频{audio_chord_id}(开放)→矛盾！"}
        elif not is_barre and audio_is_barre:
            return {"constraint": "open", "audio_chord": audio_chord_id,
                    "consistent": False, "barre": barre,
                    "detail": f"视觉开放({score:.0%}) vs 音频{audio_chord_id}(横按)→矛盾！"}
        else:
            return {"constraint": "open", "audio_chord": audio_chord_id,
                    "consistent": True, "barre": barre,
                    "detail": f"视觉开放确认→{audio_chord_id}在开放候选内"}

    def cross_validate_with_audio(self, landmark_vector: List[float],
                                  audio_chord_id: str,
                                  min_similarity: float = 0.70) -> Optional[Dict]:
        """用视觉手型家族验证音频和弦检测结果。

        Returns:
            {"match": True/False,
             "visual_family": "C_SHAPE",
             "audio_chord": "chord-C7",
             "audio_family": "C_SHAPE",
             "family_consistent": True,  # 视觉家族包含音频和弦
             "confidence": 0.91,
             "detail": "视觉C型手型确认 → C7在家族内"}
            None: 视觉未检测到足够相似的手型
        """
        family_pred = self.predict_top_family(landmark_vector, min_similarity=min_similarity)
        if family_pred is None:
            return None

        visual_family = family_pred["family"]
        audio_family = CHORD_TO_FAMILY.get(audio_chord_id, "UNKNOWN")

        family_consistent = audio_chord_id in family_pred["candidate_chords"]

        return {
            "match": family_consistent,
            "visual_family": visual_family,
            "visual_family_name": family_pred["family_name"],
            "audio_chord": audio_chord_id,
            "audio_family": audio_family,
            "family_consistent": family_consistent,
            "confidence": family_pred["confidence"],
            "detail": (
                f"视觉{family_pred['family_name']}确认"
                if family_consistent
                else f"视觉{family_pred['family_name']} vs 音频{audio_chord_id}({audio_family})矛盾"
            ),
        }

    # 保持向后兼容的 predict 方法
    def predict(self, landmark_vector: List[float], top_k: int = 3,
                min_similarity: float = 0.6) -> List[Dict]:
        """向后兼容：返回手型家族预测。"""
        return self.predict_family(landmark_vector, top_k=top_k,
                                   min_similarity=min_similarity)

    def predict_top1(self, landmark_vector: List[float],
                     min_similarity: float = 0.6) -> Optional[Dict]:
        """向后兼容：返回最佳家族预测。"""
        return self.predict_top_family(landmark_vector, min_similarity=min_similarity)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def num_classes(self) -> int:
        return len(set(f for _, f, _ in self._reference_vectors))

    def family_coverage(self) -> Dict[str, int]:
        """每个手型家族的参考图数量。"""
        counts: Dict[str, int] = {}
        for _, family, _ in self._reference_vectors:
            counts[family] = counts.get(family, 0) + 1
        return counts
