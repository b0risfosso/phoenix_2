Phone Tilt Joystick 3D Cursor Recorder
======================================

This version uses the phone as a tilt joystick for a 3D cursor.

Run:
  cd /Users/b/wwdc/phone_tilt_joystick_cursor_recorder
  python3 server.py

Open on computer:
  http://localhost:8765/dashboard.html

Open on phone using the printed LAN URL, for example:
  http://192.168.x.x:8765/phone.html

Phone setup:
  1. Tap Test server.
  2. Tap Connect to computer.
  3. Tap Run sensor diagnostic.
  4. Tap Enable sensors.
  5. Tap Start recording if you want saved JSON/CSV output.

Dashboard setup:
  1. Hold the phone in the neutral position.
  2. Click Set neutral tilt.
  3. Tilt left/right to move X.
  4. Tilt forward/back to move Y.
  5. Use Z mode to choose twist/lift/combined/locked Z movement.

Notes:
  - This mode maps orientation to desired velocity, not acceleration.
  - Returning the phone to neutral slows/stops the cursor.
  - Use Dead zone to ignore small hand tremors.
  - Use Speed to increase/decrease cursor travel.
  - Use Smoothing for steadier motion.
  - If sensors stay at zero on Android Chrome, serve the folder through HTTPS.
