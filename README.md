# AI-Based Anti-Poaching Detection System

This project is a high-stakes application of computer vision, designed to provide a real-time, automated system for detecting poaching activities from video footage. Leveraging a custom-trained YOLOv8 object detection model, the system can identify potential threats such as poachers, weapons, and animal traps.

When a threat is detected, the system automatically assigns a threat score, geo-tags the location of the incident on a dynamic map, and saves the annotated video frame as visual evidence. This tool is designed to empower park rangers and conservation authorities by providing them with immediate, actionable intelligence to protect endangered wildlife.

## 🛰️ Key Features

-   **Real-Time Threat Detection**: Analyzes video streams frame-by-frame to identify poachers, weapons (`gun`, `knife`, `machete`), and animal traps.
-   **Threat Scoring System**: Each detected object is assigned a score, and a cumulative "frame threat score" is calculated to prioritize alerts and reduce false positives.
-   **Geospatial Alerting**: Simulates a GPS patrol path and pinpoints the exact coordinates of high-threat events on an interactive map.
-   **Dynamic Threat Mapping**: Generates a `folium` map with satellite imagery, plotting the patrol route and marking threat locations with color-coded icons based on severity.
-   **Visual Evidence Generation**: Automatically saves the annotated frames where threats are detected, providing clear visual proof for review and action.
-   **Optimized for Deployment**: The final model is exported to the ONNX (Open Neural Network Exchange) format for high-performance inference on various hardware.

## ⚙️ How It Works: The Technical Pipeline

The project is divided into two core components: the AI model training and the real-time detection system.

### 1. The AI Model: Training a Poaching Specialist

The heart of the system is a YOLOv8m (medium) model, fine-tuned on a specialized dataset to recognize objects relevant to poaching.

1.  **Dataset Preparation**: The model was trained on a weapon detection dataset. A critical pre-processing step involved remapping all weapon-related classes (e.g., `gun`, `knife`) into a single, unified `Weapon_Tool` class (ID: 1). This consolidation creates a more robust and generalized weapon detector.
2.  **Data Splitting**: The dataset was programmatically split into training (70%), validation (20%), and testing (10%) sets to ensure proper model evaluation and prevent overfitting.
3.  **Model Fine-Tuning**: We started with a pre-trained YOLOv8m model and fine-tuned it on our custom dataset for 30 epochs. This process, known as transfer learning, allows the model to leverage its existing knowledge of general objects and adapt it to our specific use case.
4.  **Export to ONNX**: After training, the best-performing model weights (`best.pt`) are converted to the ONNX format. This creates a lightweight, framework-agnostic model optimized for fast inference, making it ideal for deployment in real-world applications.

### 2. The Detection & Alerting System (Inference)

The main application notebook orchestrates the end-to-end process of analyzing a video and generating alerts.

1.  **Video Input**: The user uploads a video file (e.g., from a drone, a patrol vehicle, or a static camera).
2.  **Frame-by-Frame Analysis**: The system reads the video and processes each frame individually.
3.  **GPS Path Simulation**: To demonstrate the geospatial capabilities, a realistic GPS patrol path is simulated for the duration of the video. In a real-world scenario, this would be replaced with actual GPS data from the drone or vehicle.
4.  **YOLOv8 Inference**: Each frame is passed to our custom-trained ONNX model. The model returns bounding boxes, class labels, and confidence scores for any detected objects.
5.  **Threat Assessment**: If objects are detected with a confidence above a set threshold (e.g., 40%), the system calculates a `frame_threat_score` based on predefined values (e.g., `poacher`: 10, `gun`: 8).
6.  **Geo-Tagging High-Threat Events**: If the frame's total threat score exceeds a critical threshold (e.g., 8), the event is logged as a high-threat alert. The system records the simulated GPS coordinates, the detected objects, the threat score, and the path to the annotated frame.
7.  **Visualization**:
    *   **Threat Map**: A `folium` map is generated, showing the patrol path and all geo-tagged alerts. Markers are color-coded (orange for high threat, red for critical) and contain popup information about the event.
    *   **Visual Evidence**: The annotated frames corresponding to the highest-threat alerts are displayed, providing immediate visual context for the detected incident.

## 🛠️ Technology Stack

-   **Object Detection**: Ultralytics YOLOv8
-   **Geospatial Visualization**: Folium
-   **Core Computer Vision**: OpenCV
-   **Deep Learning Framework**: PyTorch (for training)
-   **Model Deployment Format**: ONNX
-   **Data Manipulation**: NumPy
-   **Development Environment**: Google Colab, Google Drive

## 🚀 Setup and Usage

This project is designed to be run in a Google Colab environment.

### Prerequisites
- A Google account with access to Google Drive.
- A video file containing potential poaching-related scenarios for testing.

### Step 1: Train the Model (Optional)
A pre-trained `Final_AntiPoaching_Model_V1.onnx` is provided. However, if you wish to train the model yourself:
1.  Clone this repository.
2.  Upload the weapon detection dataset to your Google Drive.
3.  Open `anti_poaching_model.ipynb` in Google Colab.
4.  Update the `DRIVE_DATA_ROOT` path to point to your dataset location.
5.  Update the `DRIVE_SAVE_PATH` to specify where you want to save the final ONNX model.
6.  Run all cells to start the training and export process.

### Step 2: Run the Detection System
1.  Ensure the `Final_AntiPoaching_Model_V1.onnx` file is in your Google Drive.
2.  Open the `anti_poaching_detection.ipynb` notebook in Google Colab.
3.  **Update the `MODEL_PATH` variable** to the exact location of your `.onnx` model file in Google Drive.
4.  Run the cells sequentially.
5.  When prompted, upload the video file you want to analyze.
6.  The system will process the video and output the dynamic threat map and visual evidence frames directly in the notebook.

