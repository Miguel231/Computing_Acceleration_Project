# Computing_Acceleration_Project: Smart EdgeAI Security
A high-performance, edge-based security system for the Raspberry Pi 5. It distinguishes family members from intruders in real-time without ever sending data to the cloud.

## How does it work?
Our system is able to recognize family members by converting faces into 512-dimensional vectors (embeddings) and measures the distance between a live face and a small "Family Gallery" of images.

- **Target Hardware**: Raspberry Pi 5 
- **Detection**: YuNet (A fast, lightweight CNN detector).
- **Recognition**: GhostFaceNet++ (SoTA efficiency/accuracy for Edge)

## Installation

## Logic Flow
The system operates on a **Threshold-based Verification** system:

1.  **Extract:** On startup, the system generates embeddings for all images in the `/family` directory.
2.  **Compare:** For every face detected in the live stream, the system calculates the **Euclidean Distance** ($d$) against the stored embeddings.
3.  **Action:**
    * If $d < 0.40$: **Access Granted** (Family recognized).
    * If $d \geq 0.40$: **Intruder Alert** (Triggers external alarm via Webhook).

## Security and Privacy
- **Zero Cloud**: All processing happens locally on the hardware. No video streams or face data leave the device.
- **No Retraining:** The system is "Zero-Shot". You don't need to rebuild the model to add or remove family members.
- **Ephemereal Data**: Face embeddings are stored in RAM during runtime and are not stored permanently unless specified.

## Known Constraints
- **Lightning:** Our model uses Euclidean distance. Bad lightning can increase the distance and potentially significantly underperform.
- **Angles:** GhostFaceNet++ is robust, but profiles (side views) are generally less accurate than frontal views.
