import cv2
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
DATA_DIR = "student_faces"
LOG_FILE = "attendance_log.csv"
HAAR_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Initialize OpenCV components
face_cascade = cv2.CascadeClassifier(HAAR_FILE)
# LBPH is lightweight and works well with small datasets
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Ensure directories and log file exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Roll_ID", "Date", "Time"])

def register_student():
    """Captures 30 face samples for a new student."""
    name = input("Enter Student Name: ")
    roll_id = input("Enter Roll Number (Numbers only): ")
    
    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Capturing samples for {name}. Please look at the camera...")

    while count < 30:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            # Filename format: Name_RollID_Count.jpg
            file_path = f"{DATA_DIR}/{name}_{roll_id}_{count}.jpg"
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}/30", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Registering Student", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Registration complete for {name}.")

def train_system():
    """Trains the recognizer model with saved images."""
    faces, ids = [], []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            path = os.path.join(DATA_DIR, file)
            # Extract Roll ID from filename: Name_ID_Count.jpg
            roll_id = int(file.split('_')[1])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(roll_id)
    
    if not faces:
        return False
    
    # FIXED: Convert IDs to a numpy array of integers
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    return True

def take_attendance():
    """Recognizes faces and logs attendance to CSV."""
    if not train_system():
        print("Error: No registered students found. Please register first.")
        return

    # Create a mapping of IDs to Names from filenames
    id_to_name = {int(f.split('_')[1]): f.split('_')[0] for f in os.listdir(DATA_DIR)}
    
    cap = cv2.VideoCapture(0)
    already_marked = set() # Avoid multiple logs in one session

    print("Attendance System Active. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roll_id, confidence = recognizer.predict(roi_gray)

            # Confidence < 75 is generally a good match for LBPH
            if confidence < 75:
                name = id_to_name.get(roll_id, "Unknown")
                color = (0, 255, 0) # Green for match
                
                # Log attendance if not already marked
                if roll_id not in already_marked:
                    now = datetime.now()
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, roll_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    already_marked.add(roll_id)
                    print(f"Attendance Recorded: {name}")
            else:
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("FaceSync Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

def generate_reports():
    """Generates Day-wise and Week-wise Excel reports."""
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        print("No attendance data found to report.")
        return

    df = pd.read_csv(LOG_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    today = datetime.now().date()
    one_week_ago = today - timedelta(days=7)

    # 1. Daily Report (Today)
    daily_report = df[df['Date'].dt.date == today]
    daily_filename = f"Attendance_Daily_{today}.xlsx"
    daily_report.to_excel(daily_filename, index=False)
    
    # 2. Weekly Report (Last 7 Days)
    weekly_report = df[df['Date'].dt.date >= one_week_ago]
    weekly_filename = f"Attendance_Weekly_{today}.xlsx"
    weekly_report.to_excel(weekly_filename, index=False)
    
    print(f"Reports generated:\n- {daily_filename}\n- {weekly_filename}")

if __name__ == "__main__":
    while True:
        print("\n=== FaceSync Menu ===")
        print("1. Register Student")
        print("2. Take Attendance")
        print("3. Generate Excel Reports")
        print("4. Exit")
        choice = input("Enter choice: ")

        if choice == '1': register_student()
        elif choice == '2': take_attendance()
        elif choice == '3': generate_reports()
        elif choice == '4': break
        else: print("Invalid choice.")