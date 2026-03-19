import cv2
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = "sample.mp4"
TAG_TEMPLATE_PATH = "tag_template.png"
OUTPUT_EXCEL = "staff_coordinates.xlsx"
OUTPUT_VIDEO = "staff_output.mp4"

PERSON_CONF = 0.35
DETECT_THRESHOLD = 0.42
REACQUIRE_THRESHOLD = 0.58
TRACKER_MAX_MISSES = 12
RESET_MISSES = 12


def make_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("CSRT not found, using KCF tracker instead.")
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerKCF_create"):
        print("CSRT not found, using KCF tracker instead.")
        return cv2.TrackerKCF_create()
    return None


def reinit_tracker(frame, box_xyxy):
    tracker = make_tracker()
    if tracker is None:
        return None
    x1, y1, x2, y2 = box_xyxy
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return tracker


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (int(x), int(y), int(x + w), int(y + h))


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def area_of(box):
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))


def distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def load_tag_template(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load tag template: {path}")
    return img


def preprocess(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 60, 160)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, edges, binary


def get_tag_search_regions(person_crop):
    """
    Search both upper-left and upper-right torso.
    """
    h, w = person_crop.shape[:2]

    y1 = int(h * 0.10)
    y2 = int(h * 0.52)

    left_x1 = int(w * 0.08)
    left_x2 = int(w * 0.62)

    right_x1 = int(w * 0.38)
    right_x2 = int(w * 0.92)

    left_region = person_crop[y1:y2, left_x1:left_x2]
    right_region = person_crop[y1:y2, right_x1:right_x2]

    return [
        ("left", left_region),
        ("right", right_region),
    ]


def multi_scale_tag_score(search_region, tag_template):
    if search_region is None or search_region.size == 0:
        return -1.0

    region_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    region_gray = cv2.resize(
        region_gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC
    )

    rg, re, rb = preprocess(region_gray)
    best_score = -1.0

    for scale in [0.35, 0.50, 0.65, 0.80, 1.0, 1.2, 1.5, 1.8]:
        temp = cv2.resize(
            tag_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        tg, te, tb = preprocess(temp)
        th, tw = tg.shape[:2]
        rh, rw = rg.shape[:2]

        if th >= rh or tw >= rw:
            continue

        res_gray = cv2.matchTemplate(rg, tg, cv2.TM_CCOEFF_NORMED)
        res_edge = cv2.matchTemplate(re, te, cv2.TM_CCOEFF_NORMED)
        res_bin = cv2.matchTemplate(rb, tb, cv2.TM_CCOEFF_NORMED)

        _, gray_score, _, _ = cv2.minMaxLoc(res_gray)
        _, edge_score, _, _ = cv2.minMaxLoc(res_edge)
        _, bin_score, _, _ = cv2.minMaxLoc(res_bin)

        combined = 0.35 * gray_score + 0.40 * edge_score + 0.25 * bin_score
        if combined > best_score:
            best_score = combined

    return best_score


def person_tag_score(person_crop, tag_template):
    best_score = -1.0
    best_side = None

    for side_name, region in get_tag_search_regions(person_crop):
        score = multi_scale_tag_score(region, tag_template)
        if score > best_score:
            best_score = score
            best_side = side_name

    return best_score, best_side


def choose_candidate(candidates, last_box):
    """
    Conservative choice:
    - if no previous box, choose only a strong detection
    - if already tracking someone, only allow nearby candidates
    - NEVER jump to a far-away strongest candidate
    """
    strong = [c for c in candidates if c["tag_score"] >= DETECT_THRESHOLD]
    if not strong:
        return None

    if last_box is None:
        best = max(strong, key=lambda c: c["tag_score"])
        if best["tag_score"] >= REACQUIRE_THRESHOLD:
            return best
        return None

    last_center = center_of(last_box)
    last_area = area_of(last_box)
    last_w = last_box[2] - last_box[0]
    last_h = last_box[3] - last_box[1]

    gate_dist = max(65, 0.85 * max(last_w, last_h))

    nearby = []
    for c in strong:
        d = distance(c["center"], last_center)
        size_ratio = c["area"] / last_area

        if d > gate_dist:
            continue
        if size_ratio < 0.50 or size_ratio > 2.0:
            continue

        motion_score = max(0.0, 1.0 - d / gate_dist)
        combined = 0.55 * c["tag_score"] + 0.45 * motion_score
        nearby.append((combined, c))

    if nearby:
        nearby.sort(key=lambda x: x[0], reverse=True)
        return nearby[0][1]

    # IMPORTANT:
    # no nearby match = do not switch to another person
    return None


def main():
    model = YOLO("yolov8n.pt")
    tag_template = load_tag_template(TAG_TEMPLATE_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows = []
    frame_number = 0

    last_box = None
    last_score = -1.0
    tracker = None
    misses = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp_sec = frame_number / fps

        tracker_box = None
        if tracker is not None:
            ok, tbox = tracker.update(frame)
            if ok:
                tracker_box = xywh_to_xyxy(tbox)

        detections = model(frame, verbose=False)[0]
        candidates = []

        for box in detections.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id != 0 or conf < PERSON_CONF:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            tag_score, best_side = person_tag_score(person_crop, tag_template)

            candidates.append({
                "box": (x1, y1, x2, y2),
                "center": center_of((x1, y1, x2, y2)),
                "area": area_of((x1, y1, x2, y2)),
                "tag_score": tag_score,
                "best_side": best_side,
            })

        chosen = choose_candidate(candidates, last_box)
        chosen_box = None
        chosen_score = -1.0
        chosen_side = None

        if chosen is not None:
            chosen_box = chosen["box"]
            chosen_score = chosen["tag_score"]
            chosen_side = chosen.get("best_side", None)
            last_box = chosen_box
            last_score = chosen_score
            tracker = reinit_tracker(frame, chosen_box)
            misses = 0

        elif tracker_box is not None and misses < TRACKER_MAX_MISSES:
            # keep following the previous person briefly
            chosen_box = tracker_box
            chosen_score = max(0.0, last_score - 0.02)
            chosen_side = "tracker"
            last_box = chosen_box
            misses += 1

        else:
            # uncertain: do NOT switch to blue shirt
            chosen_box = None
            chosen_score = -1.0
            chosen_side = None
            misses += 1

            if misses >= RESET_MISSES:
                last_box = None
                last_score = -1.0
                tracker = None

        staff_present = "No"
        center_x = ""
        center_y = ""
        bbox_x1 = ""
        bbox_y1 = ""
        bbox_x2 = ""
        bbox_y2 = ""

        if chosen_box is not None:
            x1, y1, x2, y2 = chosen_box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            staff_present = "Yes"
            center_x = cx
            center_y = cy
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = x1, y1, x2, y2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            label = f"STAFF {chosen_score:.2f}"
            if chosen_side:
                label += f" {chosen_side}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Searching staff...",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if candidates:
            best_debug = max(candidates, key=lambda c: c["tag_score"])
            bx1, by1, bx2, by2 = best_debug["box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
            cv2.putText(
                frame,
                f"BEST {best_debug['tag_score']:.2f} {best_debug['best_side']}",
                (bx1, min(height - 10, by2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        rows.append({
            "frame_number": frame_number,
            "timestamp_sec": round(timestamp_sec, 3),
            "staff_present": staff_present,
            "x": center_x,
            "y": center_y,
            "bbox_x1": bbox_x1,
            "bbox_y1": bbox_y1,
            "bbox_x2": bbox_x2,
            "bbox_y2": bbox_y2,
            "match_score": round(chosen_score, 4) if chosen_box is not None else "",
        })

        out.write(frame)

        preview = cv2.resize(frame, None, fx=0.7, fy=0.7)
        cv2.imshow("Staff Detection", preview)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df = df[df["staff_present"] == "Yes"][[
        "frame_number", "timestamp_sec", "x", "y",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
    ]]
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"Done. Excel saved to: {OUTPUT_EXCEL}")
    print(f"Annotated video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()