# 필요한 전문가들 불러오기
from ultralytics import YOLO
import ctypes
import cv2
import numpy as np
from collections import defaultdict
import sys
import time

sys.path.append('./sort')
from sort import Sort

# === 메인 프로그램 시작 ===
def main():
    # --- 1. 초기 설정 ---
    model = YOLO('yolov8n-pose.pt') # YOLO 모델 준비
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3) # ID 추적 전문가 준비

    base_data = {}
    final_scores = {}
    jump_counters = defaultdict(int)
    last_head_ys = defaultdict(lambda: None)
    jump_states = defaultdict(int)

    floor_timers = defaultdict(lambda: None)
    floor_y_history = defaultdict(list)
    CALIBRATION_TIME = 2.0
    STABILITY_THRESH = 20

    jump_threshold_ratio = 0.07

    # ★★★ 신뢰도 기준점 추가 ★★★
    JOINT_CONF_THRESH = 0.5 # 50% 이상 신뢰도일 때만 '보인다'고 판단

    # --- 2. 카메라 및 화면 설정 ---
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print("모니터 해상도:", screen_width, screen_height)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"요청 해상도: 1920x1080, 카메라 실제 해상도: {frame_width} x {frame_height}")

    cv2.namedWindow('Jump Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Jump Counter - Multi Person Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # === 3. 메인 루프 (프로그램 실행) ===
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 화면 그리기 ---
        threshold_text = f"Jump Threshold: {jump_threshold_ratio * 100:.1f}% (Up/Down Arrow)"
        cv2.putText(frame, threshold_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- 사람 감지 및 ID 추적 ---
        results = model(frame, conf=0.6, verbose=False) 

        # ★★★ 감지된 사람 정보 정리 (신뢰도 'conf' 추가) ★★★
        dets = []
        keypoints_list = [] # (xy_data, conf_data) 튜플로 저장
        
        for r in results:
            if r.keypoints is not None:
                # keypoints 객체에서 xy와 conf를 가져옴
                xy_data = getattr(r.keypoints, "xy", [])
                conf_data = getattr(r.keypoints, "conf", [])

                if len(xy_data) != len(conf_data):
                    continue # 데이터 짝이 안 맞으면 스킵

                for i in range(len(xy_data)):
                    person_kp_tensor = xy_data[i]
                    person_conf_tensor = conf_data[i]
                    
                    person_kp = person_kp_tensor.cpu().numpy()
                    person_conf = person_conf_tensor.cpu().numpy()

                    # 좌표 데이터로 바운딩 박스 계산
                    # (신뢰도가 낮은 좌표도 박스 계산에는 포함될 수 있음)
                    valid_kps = person_kp[person_kp[:, 1] > 10] # (0,0)이 아닌 좌표만
                    if len(valid_kps) == 0:
                        continue # 유효한 관절이 하나도 없으면 스킵

                    min_x = np.min(valid_kps[:,0])
                    max_x = np.max(valid_kps[:,0])
                    min_y = np.min(valid_kps[:,1])
                    max_y = np.max(valid_kps[:,1])
                        
                    dets.append([min_x, min_y, max_x, max_y, 1.0])
                    keypoints_list.append((person_kp, person_conf)) # 좌표와 신뢰도를 함께 저장
        
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
        # ★★★ 여기까지 수정 ★★★

        tracks = tracker.update(dets)

        # --- 키보드 입력 처리 ---
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break
        elif key == 2490368: 
            jump_threshold_ratio += 0.005
            print(f"Jump threshold increased to: {jump_threshold_ratio * 100:.1f}%")
        elif key == 2621440:
            jump_threshold_ratio = max(0.005, jump_threshold_ratio - 0.005)
            print(f"Jump threshold decreased to: {jump_threshold_ratio * 100:.1f}%")

        # --- ID별 점프 로직 처리 ---
        active_track_ids = set()
        matched = set()

        for t in tracks:
            x1, y1, x2, y2, track_id = t
            active_track_ids.add(track_id)

            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            best_idx = -1
            best_dist = 1e9
            
            # ★★★ ID-관절 매칭 시, (xy, conf) 튜플 사용 ★★★
            for idx, (person_kp_loop, person_conf_loop) in enumerate(keypoints_list):
                head_kp = person_kp_loop[0] # 0번 관절 (코)
                center = np.array([head_kp[0], head_kp[1]])
                dist = np.linalg.norm(bbox_center - center)
                if dist < best_dist and idx not in matched:
                    best_dist = dist
                    best_idx = idx
            if best_idx == -1:
                continue
            matched.add(best_idx)

            # --- 좌표 및 상태 변수 계산 ---
            person_xy, person_conf = keypoints_list[best_idx] # (xy, conf) 분리
            
            head = person_xy[0]
            l_ankle = person_xy[15]
            r_ankle = person_xy[16]
            head_y = head[1]
            head_x = head[0]
            foot_y = max(l_ankle[1], r_ankle[1])

            min_x, min_y, max_x, max_y = int(x1), int(y1), int(x2), int(y2)
            current_bbox_height = max_y - min_y
            current_bbox_width = max_x - min_x

            # (1) ★★★ 가시성 판단 (신뢰도 기반) ★★★
            head_conf = person_conf[0]
            l_ankle_conf = person_conf[15]
            r_ankle_conf = person_conf[16]
            
            is_head_visible = head_conf > JOINT_CONF_THRESH
            is_foot_visible = (l_ankle_conf > JOINT_CONF_THRESH or r_ankle_conf > JOINT_CONF_THRESH)

            # (2) ★★★ 캘리브레이션용 전신 가시성 (신뢰도 기반) ★★★
            major_joints_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16] # 코, 양어깨, 양골반, 양무릎, 양발목
            
            is_full_body_visible = True
            for idx in major_joints_indices:
                confidence = person_conf[idx]
                if confidence < JOINT_CONF_THRESH:
                    is_full_body_visible = False
                    # print(f"ID {track_id} 캘리브레이션 불가: {idx}번 관절 신뢰도 낮음 ({confidence:.2f})")
                    break
            
            lh = last_head_ys[track_id]
            diff = lh - head_y if (lh is not None and is_head_visible) else 0

            # 5. "기준 키"가 아직 측정되지 않았는지 확인 (캘리브레이션 단계)
            if track_id not in base_data:
                
                # ★★★ 시작 조건 (is_full_body_visible) ★★★
                if is_full_body_visible:
                    current_time = time.time()
                    if floor_timers[track_id] is None:
                        floor_timers[track_id] = current_time
                        floor_y_history[track_id] = [foot_y]
                    else:
                        floor_y_history[track_id].append(foot_y)
                        elapsed = current_time - floor_timers[track_id]
                        
                        if elapsed > CALIBRATION_TIME:
                            history = floor_y_history[track_id]
                            y_movement = np.max(history) - np.min(history)
                            
                            if y_movement < STABILITY_THRESH:
                                base_height = foot_y - head_y 
                                if base_height > 100:
                                    base_data[track_id] = (base_height, current_bbox_height, current_bbox_width)
                                    jump_counters[track_id] = 0
                                    jump_states[track_id] = 0
                                    print(f"ID {track_id} 자동 캘리브레이션 완료: H:{base_height}, H_box:{current_bbox_height}, W_box:{current_bbox_width}")
                            else:
                                print(f"ID {track_id} 캘리브레이션 실패: Y좌표 흔들림 {y_movement}px > {STABILITY_THRESH}px")
                            
                            floor_timers[track_id] = None
                            floor_y_history[track_id] = []
                else:
                    # (전신이 안보이면) 타이머 리셋
                    floor_timers[track_id] = None
                    floor_y_history[track_id] = []

            # 6. "기준 키"가 이미 있음 (점프 카운트 단계)
            # (리셋 로직은 'is_foot_visible' 사용 - 신뢰도 기반으로 자동 변경됨)
            else:
                base_height, base_bbox_height, base_bbox_width = base_data[track_id]

                is_too_close = False
                if current_bbox_height > (base_bbox_height * 1.5): is_too_close = True
                if current_bbox_width > (base_bbox_width * 1.5): is_too_close = True

                if is_too_close:
                    final_scores[track_id] = jump_counters[track_id] 
                    del base_data[track_id] 
                    jump_states[track_id] = 0
                    print(f"ID {track_id} 너무 가까움 (박스 커짐). 기준 키 리셋.")
                elif not is_foot_visible and jump_states[track_id] == 0: 
                    final_scores[track_id] = jump_counters[track_id] 
                    del base_data[track_id] 
                    jump_states[track_id] = 0
                    print(f"ID {track_id} 발 안보임 (땅). 기준 키 리셋.")
                elif is_head_visible:
                    JUMP_THRESH_PIXELS = base_height * jump_threshold_ratio 
                    if jump_states[track_id] == 0 and diff > JUMP_THRESH_PIXELS:
                        jump_counters[track_id] += 1
                        jump_states[track_id] = 1
                    if jump_states[track_id] == 1 and diff < 10: 
                        jump_states[track_id] = 0

            if is_head_visible:
                last_head_ys[track_id] = head_y

            if track_id in base_data and is_head_visible:
                cx, cy = int(head_x), int(head_y)
                cv2.putText(frame, f'ID:{track_id} {jump_counters[track_id]}', (cx-40, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 128, 255), 2)

        # === ID별 순회 루프 끝 ===

        # --- 사라진 ID 처리 ---
        calibrated_ids = set(base_data.keys())
        lost_ids = calibrated_ids - active_track_ids
        for lost_track_id in lost_ids:
            print(f"ID {lost_track_id} 화면 이탈. 점수 저장 및 리셋.")
            final_scores[lost_track_id] = jump_counters[lost_track_id]
            del base_data[lost_track_id]
            if lost_track_id in jump_states: del jump_states[lost_track_id]
            if lost_track_id in last_head_ys: del last_head_ys[lost_track_id]
            if lost_track_id in floor_timers: del floor_timers[lost_track_id]
            if lost_track_id in floor_y_history: del floor_y_history[lost_track_id]

        # --- 최종 점수판 그리기 ---
        y_pos = frame_height - 100 
        x_pos = frame_width - 250
        cv2.putText(frame, "== Final Scores ==", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        sorted_scores = sorted(final_scores.items())
        for id_num, count in sorted_scores:
            y_pos += 30 
            text = f"ID {id_num} : {count}"
            cv2.putText(frame, text, (x_pos + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- 최종 화면 표시 ---
        cv2.imshow('Jump Counter - Multi Person Tracking', frame)

    # === 메인 루프 끝 ===

    # --- 종료 처리 ---
    cap.release()
    cv2.destroyAllWindows()

# === 프로그램 진입점 ===
if __name__ == "__main__":
    main()