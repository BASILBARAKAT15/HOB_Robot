HOB Robot V8 — Automatic CAD door goals and shared entrances (desktop simulation)

Windows: extract the ZIP, open PowerShell in the extracted folder containing
hob_navigation.py, then run:
  python -m pip install -r requirements.txt
  python hob_navigation.py

All precomputed goals are loaded automatically from door_positions.json. No
manual door selection is needed. Click a corridor to place the simulated robot
and set HOME, choose a room, and click Add task. Submit the simulated OCR code
on arrival; the next queued task starts, or the robot returns to HOME.
Show detected doors displays red opening centers and green stopping points.

Detection report: 92 CAD room labels (91 distinct codes), 92 supported goals:
78 door blocks, 4 inferred paired wall openings, and 7 shared exterior door
assignments inferred from interior door geometry and room-number families.
The 3 unresolved labels (2270, 2271, 2680) remain blocked. No manual door
selection is required for the 92 supported goals. The requested room stays
in the task history; OCR checks the exterior room code for a shared entrance.
For example, request 2032 and submit 2030 at arrival. This confirms only the
suite entrance, not arrival inside 2032. If the real sign differs, retry or
stop; do not assume a match. Duplicate 2780 labels are separate locations.
All coordinates and corridor graph must be reviewed before physical use.
Door stop offset is 900 CAD units (likely mm, not calibrated).
This is not validated for motor control.

To rerun extraction on the original DXF for this same map, install:
  python -m pip install -r requirements-detection.txt
  python auto_detect_doors.py "C:\path\engineering final as build.dxf"
This regenerates door_positions.json, the detection report and preview.
The original DXF is not bundled. The extraction uses the existing floor2 map
and corridor graph; it is not a general CAD floor plan importer.

Camera OCR (optional): python -m pip install -r requirements-camera.txt
Install Tesseract separately, select Camera OCR and use a webcam index or a
uStreamer snapshot URL. Physical camera, robot and floor clearance untested.

Tests: python -m unittest -v test_navigation
