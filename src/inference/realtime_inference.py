import cv2
import time
import collections
import numpy as np
import webbrowser
import subprocess
from src.camera.camera import CameraManager
from src.inference.gesture_translator import ISLGestureTranslator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("RealTimePipeline")

def launch_whatsapp():
    """Opens WhatsApp via Web and native desktop URI protocol."""
    try:
        logger.info("Triggering WhatsApp launch...")
        webbrowser.open("https://web.whatsapp.com")
    except Exception as e:
        logger.error(f"Failed to launch WhatsApp Web, falling back to desktop protocol: {e}")
        try:
            subprocess.Popen(["cmd", "/c", "start", "whatsapp:"], shell=True)
        except Exception as e2:
            logger.error(f"Failed desktop protocol: {e2}")

def run_realtime_application(mode: str = "STATIC", config_path: str = "config.yaml"):
    """
    Unified Real-Time ISL Recognition & Action Trigger System.
    - Tracks hand landmarks using MediaPipe.
    - Classifies ISL signs (A-Z, Thumbs Up/Down, Peace/V, W, etc.).
    - Triggers Assistive Desktop Actions:
        * Hold Sign 'W' for 1.5s -> Automatically Opens WhatsApp!
    """
    cfg = load_config(config_path)
    cam_cfg = cfg.get("camera", {})
    inf_cfg = cfg.get("inference", {})

    conf_threshold = inf_cfg.get("confidence_threshold", 0.40)
    mirror_mode = True

    # Initialize Camera Hardware
    cam = CameraManager(
        camera_index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        target_fps=cam_cfg.get("fps", 30)
    )

    if not cam.start():
        logger.error("Failed to start camera hardware. Exiting real-time application.")
        return

    # Initialize Gesture & ISL Translator Engine
    translator = ISLGestureTranslator()

    frame_counter = 0
    prev_time = time.time()
    fps = 0.0

    # Phrase history for dynamic sentence formation
    phrase_tokens = []
    last_added_token = ""
    last_added_time = 0

    # Action Trigger Tracking for WhatsApp
    whatsapp_start_time = None
    last_action_time = 0
    action_message = ""
    action_message_time = 0
    HOLD_REQUIRED_SEC = 1.0
    ACTION_COOLDOWN_SEC = 5.0

    logger.info(f"Starting Unified ISL Translator | Mode: {mode.upper()} | Mirror: {mirror_mode}")

    try:
        while True:
            ret, raw_frame = cam.read_frame()
            if not ret or raw_frame is None:
                logger.error("Empty frame received from camera feed.")
                break

            # Mirror mode (flip horizontally so movements match user perspective)
            if mirror_mode:
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame.copy()

            frame_counter += 1
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Run MediaPipe hand tracking & sign translation
            out_frame, current_label, current_confidence, top3_candidates, hand_bbox, distance_hint = translator.process_frame(frame)

            h, w, _ = out_frame.shape

            # -------------------------------------------------------------
            # ASSISTIVE ACTION TRIGGER: HOLD SIGN 'W' -> OPEN WHATSAPP
            # -------------------------------------------------------------
            is_w_sign = ("WhatsApp" in current_label) or ("Sign 'W'" in current_label)
            hold_progress = 0.0

            if is_w_sign and current_confidence >= 0.70:
                if whatsapp_start_time is None:
                    whatsapp_start_time = curr_time
                hold_duration = curr_time - whatsapp_start_time
                hold_progress = min(1.0, hold_duration / HOLD_REQUIRED_SEC)

                if hold_progress >= 1.0 and (curr_time - last_action_time) > ACTION_COOLDOWN_SEC:
                    launch_whatsapp()
                    last_action_time = curr_time
                    action_message = "ACTION EXECUTED: OPENING WHATSAPP!"
                    action_message_time = curr_time
                    whatsapp_start_time = None
                    hold_progress = 0.0
            else:
                whatsapp_start_time = None
                hold_progress = 0.0

            # Token accumulation for dynamic phrase mode
            if mode.upper() == "DYNAMIC":
                if current_confidence >= 0.70 and current_label not in ("Show Hand to Camera", "Analyzing Hand..."):
                    if current_label != last_added_token and (curr_time - last_added_time) > 1.2:
                        token_short = current_label.replace("Sign '", "").replace("'", "").split(" ")[0]
                        phrase_tokens.append(token_short)
                        last_added_token = current_label
                        last_added_time = curr_time
                        if len(phrase_tokens) > 6:
                            phrase_tokens.pop(0)

            # Determine status color
            if current_confidence >= 0.70:
                conf_color = (0, 255, 0)      # Bright Green
            elif current_confidence >= conf_threshold:
                conf_color = (0, 230, 255)    # Amber/Yellow
            else:
                conf_color = (120, 180, 255)  # Light Blue / Gray

            # -------------------------------------------------------------
            # Top Dashboard Header Bar (Height = 115 px)
            # -------------------------------------------------------------
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)

            # Line 1: System Info & Distance Coaching
            if distance_hint == "TOO_CLOSE":
                dist_str = "TOO CLOSE (Move hand back slightly)"
                dist_color = (0, 140, 255)
            elif distance_hint == "TOO_FAR":
                dist_str = "TOO FAR (Move hand closer 1-2 ft)"
                dist_color = (0, 200, 255)
            elif distance_hint == "OPTIMAL":
                dist_str = "OPTIMAL (3D Skeleton Tracked)"
                dist_color = (0, 255, 100)
            else:
                dist_str = "SEARCHING (Keep hand 1-2 ft away)"
                dist_color = (180, 180, 180)

            cv2.putText(out_frame, f"Mode: {mode.upper()}  |  FPS: {fps:.1f}  |  Position: {dist_str}", 
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, dist_color, 1)

            # Line 2: Prominent Recognized Sign Banner
            cv2.putText(out_frame, "Recognized Sign :", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(out_frame, f"[ {current_label} ]", (210, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)

            # Line 2 (Right): Confidence Percentage
            conf_percent = current_confidence * 100.0
            cv2.putText(out_frame, f"{conf_percent:.1f}%", (w - 110, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, conf_color, 2)

            # Draw Confidence Bar Underneath
            bar_x = w - 180
            bar_y = 72
            bar_w = 160
            bar_h = 7
            cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * min(1.0, current_confidence))
            cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)

            # Line 3: Dynamic Sentence or Top 3 Candidates
            if mode.upper() == "DYNAMIC" and phrase_tokens:
                sentence_str = "Sign Sequence:  " + "  ->  ".join(phrase_tokens)
                cv2.putText(out_frame, sentence_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
            elif top3_candidates:
                top3_str = "CNN Top Candidates:  " + "   |   ".join(
                    [f"#{i+1}: [{c[0]}] ({c[1]*100:.0f}%)" for i, c in enumerate(top3_candidates[:3])]
                )
                cv2.putText(out_frame, top3_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)
            else:
                cv2.putText(out_frame, "Shortcuts: Sign 'W' -> Open WhatsApp  |  Thumbs Up  |  V / Peace  |  A-Z", 
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

            # -------------------------------------------------------------
            # ACTION TRIGGER OVERLAYS (WHATSAPP HOLD PROGRESS & TRIGGER BANNER)
            # -------------------------------------------------------------
            if hold_progress > 0.0:
                # Draw animated progress bar in center of screen
                prog_w = 340
                prog_h = 42
                px = int((w - prog_w) / 2)
                py = 135

                cv2.rectangle(out_frame, (px - 5, py - 5), (px + prog_w + 5, py + prog_h + 5), (20, 20, 20), -1)
                cv2.rectangle(out_frame, (px - 5, py - 5), (px + prog_w + 5, py + prog_h + 5), (0, 255, 100), 2)

                # Progress fill
                fill_p = int(prog_w * hold_progress)
                cv2.rectangle(out_frame, (px, py + 22), (px + fill_p, py + prog_h - 4), (0, 255, 0), -1)

                pct = int(hold_progress * 100)
                cv2.putText(out_frame, f"OPENING WHATSAPP... HOLD SIGN ({pct}%)", (px + 12, py + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

            # If action was recently triggered, flash success banner for 3.5 seconds
            if (curr_time - action_message_time) < 3.5:
                banner_w = 440
                banner_h = 48
                bx = int((w - banner_w) / 2)
                by = 130
                cv2.rectangle(out_frame, (bx - 5, by - 5), (bx + banner_w + 5, by + banner_h + 5), (0, 180, 0), -1)
                cv2.rectangle(out_frame, (bx - 5, by - 5), (bx + banner_w + 5, by + banner_h + 5), (255, 255, 255), 2)
                cv2.putText(out_frame, action_message, (bx + 15, by + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # -------------------------------------------------------------
            # Bottom-Right Picture-in-Picture (PiP) Inset (Cropped Hand)
            # -------------------------------------------------------------
            if hand_bbox is not None:
                xmin, ymin, xmax, ymax = hand_bbox
                cropped_hand = frame[ymin:ymax, xmin:xmax]
                if cropped_hand is not None and cropped_hand.size > 0:
                    pip_size = 110
                    pip_img = cv2.resize(cropped_hand, (pip_size, pip_size))
                    px1 = w - pip_size - 15
                    py1 = h - pip_size - 38
                    px2 = px1 + pip_size
                    py2 = py1 + pip_size

                    cv2.rectangle(out_frame, (px1 - 2, py1 - 16), (px2 + 2, py2 + 2), (30, 30, 30), -1)
                    cv2.rectangle(out_frame, (px1 - 2, py1 - 16), (px2 + 2, py2 + 2), (0, 255, 0), 1)
                    cv2.putText(out_frame, "Hand Crop (AI)", (px1, py1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    out_frame[py1:py2, px1:px2] = pip_img

            # -------------------------------------------------------------
            # Footer Instructions Bar (Height = 28 px)
            # -------------------------------------------------------------
            cv2.rectangle(out_frame, (0, h - 28), (w, h), (15, 15, 15), -1)
            cv2.putText(out_frame, "Hold Sign 'W' for 1.0s -> Launches WhatsApp  |  [Q] Exit  |  [M] Mirror  |  [C] Clear", 
                        (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

            cv2.imshow("ISL Real-Time Recognition System", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested exit ('q' key pressed).")
                break
            elif key == ord('m'):
                mirror_mode = not mirror_mode
                logger.info(f"Mirror mode toggled: {mirror_mode}")
            elif key == ord('c'):
                phrase_tokens.clear()
                translator.history.clear()

    finally:
        cam.release()
        logger.info("Real-time application terminated safely.")
