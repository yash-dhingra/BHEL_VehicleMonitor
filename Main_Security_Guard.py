import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
import csv
from datetime import datetime

def append_to_csv(file_path, data):
    try:
        with open(file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data)
        print("Data appended successfully.")
    except Exception as e:
        print(f"Error: {e}")

def mark_exit(csv_file_path, number_plate):
    try:
        print("Run")
        # Read the CSV file and store its content
        with open(csv_file_path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
        print(rows)
        # Find the row with the specified number plate
        for row in rows:
            if row[0] == number_plate and row[1] == '0':
                print("Match Success")
                # Mark "Exited" as 1 and insert the exit time
                row[1] = '1'
                row[4] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write the updated content back to the CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(rows)

        print(f"Exit marked for {number_plate}.")
    except Exception as e:
        print(f"Error: {e}")

def predict_torc(path):
    from keras.models import load_model  # TensorFlow is required for Keras to work
    from PIL import Image, ImageOps  # Install pillow instead of PIL
    import numpy as np

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("model/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("model/labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score,class_name)
    return class_name[2:]


class EntryWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Entry Details")

        # Image placeholders (replace with actual images)
        self.image1 = Image.open("Temp1.png")
        self.resized = self.image1.resize((720, 480))
        self.photo1 = ImageTk.PhotoImage(self.resized)
        self.image1_label = tk.Label(self.window,image=self.photo1,width=720, height=480)
        self.image1_label.pack()

        self.image2 = Image.open("Temp2.png")
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.image2_label = tk.Label(self.window, text="Number Plate: ", image=self.photo2,height=200)
        self.image2_label.pack()

        # add label for number plate 
        self.number_plate_label = tk.Label(self.window, text="Number Plate")
        self.number_plate_label.pack()


        # Text area for number plate
        self.number_plate_entry = tk.Entry(self.window, width=30)
        self.number_plate_entry.pack()

        # Radio buttons for vehicle type
        self.vehicle_type_var = tk.StringVar(value="Cars")
        self.car_radio = tk.Radiobutton(self.window, text="Cars", variable=self.vehicle_type_var, value="Cars")
        self.car_radio.pack()

        self.truck_radio = tk.Radiobutton(self.window, text="Trucks", variable=self.vehicle_type_var, value="Trucks")
        self.truck_radio.pack()
        value_pred=str(predict_torc("Temp1.png"))
        if(value_pred[0]=="C"):
            self.vehicle_type_var.set("Cars")
        else:
            self.vehicle_type_var.set("Trucks")

        # self.vehicle_type_var.set(value_pred)

        # Approve button
        self.approve_button = tk.Button(self.window, text="Approve", command=self.approve_button_click)
        self.approve_button.pack()

    def approve_button_click(self):
        number_plate = self.number_plate_entry.get()
        vehicle_type = self.vehicle_type_var.get()
        csv_file_path = 'entry_exit.csv'
        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_to_append = [number_plate, 0,vehicle_type ,entry_time, '-']
        append_to_csv(csv_file_path, data_to_append)
        print(f"Entry Approved: Number Plate - {number_plate}, Vehicle Type - {vehicle_type}")
        self.window.destroy()

class ExitWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Exit Details")

        # Image placeholders (replace with actual images)
        
        self.image1 = Image.open("Temp1.png")
        self.resized = self.image1.resize((720, 480))
        self.photo1 = ImageTk.PhotoImage(self.resized)
        self.image1_label = tk.Label(self.window,image=self.photo1,width=720, height=480)
        self.image1_label.pack()

        self.image2 = Image.open("Temp2.png")
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.image2_label = tk.Label(self.window, text="Number Plate: ", image=self.photo2,height=200)
        self.image2_label.pack()

        # add label for number plate 
        self.number_plate_label = tk.Label(self.window, text="Number Plate")
        self.number_plate_label.pack()


        # Text area for number plate
        self.number_plate_entry = tk.Entry(self.window, width=30)
        self.number_plate_entry.pack()

        # Radio buttons for vehicle type
        self.vehicle_type_var = tk.StringVar(value="Cars")
        self.car_radio = tk.Radiobutton(self.window, text="Cars", variable=self.vehicle_type_var, value="Cars")
        self.car_radio.pack()

        self.truck_radio = tk.Radiobutton(self.window, text="Trucks", variable=self.vehicle_type_var, value="Trucks")
        self.truck_radio.pack()
        value_pred=str(predict_torc("Temp1.png"))
        if(value_pred[0]=="C"):
            self.vehicle_type_var.set("Cars")
        else:
            self.vehicle_type_var.set("Trucks")

        # self.vehicle_type_var.set(value_pred)

        # Approve button
        self.approve_button = tk.Button(self.window, text="Approve", command=self.approve_button_click)
        self.approve_button.pack()

    def approve_button_click(self):
        number_plate = self.number_plate_entry.get()
        vehicle_type = self.vehicle_type_var.get()
        csv_file_path = 'entry_exit.csv'
        mark_exit(csv_file_path, number_plate)
        print(f"Exit Approved: Number Plate - {number_plate}, Vehicle Type - {vehicle_type}")
        self.window.destroy()

class WebcamApp:
    # def __init__(self, window, window_title):
    #     self.window = window
    #     self.window.title(window_title)

    #     # Initialize webcam
    #     self.cap = cv2.VideoCapture(2)
    #     _, self.frame = self.cap.read()

    #     # Create canvas to display webcam feed
    #     # self.canvas = tk.Canvas(window, width=self.frame.shape[1], height=self.frame.shape[0])
    #     self.canvas = tk.Canvas(window, width=720, height=480)
    #     self.canvas.pack()

    #     # Create Entry button
    #     self.entry_button = tk.Button(window, text="Entry", command=self.open_entry_window)
    #     self.entry_button.pack(side=tk.TOP, pady=10)

    #     # Create Exit button
    #     self.exit_button = tk.Button(window, text="Exit", command=self.open_exit_window)
    #     self.exit_button.pack(side=tk.BOTTOM, pady=10)

    #     # Update webcam feed
    #     self.update()

    #     # Set up main loop
    #     self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    #     self.window.mainloop()

    # def update(self):
    #     _, self.frame = self.cap.read()
    #     self.photo = self.convert_to_photo(self.frame)
    #     self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    #     self.window.after(10, self.update)

    # def convert_to_photo(self, frame):
    #     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     pil_image = Image.fromarray(rgb_image)
    #     photo = ImageTk.PhotoImage(image=pil_image)
    #     return photo

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize webcam
        self.cap = cv2.VideoCapture(2)
        _, self.frame = self.cap.read()

        # Set the desired dimensions for the canvas
        canvas_width = 720
        canvas_height = 480

        # Create canvas to display webcam feed
        self.canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        # Create Entry button
        self.entry_button = tk.Button(window, text="Entry", command=self.open_entry_window)
        self.entry_button.pack(side=tk.TOP, pady=10)

        # Create Exit button
        self.exit_button = tk.Button(window, text="Exit", command=self.open_exit_window)
        self.exit_button.pack(side=tk.BOTTOM, pady=10)

        # Update webcam feed
        self.update()

        # Set up main loop
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def update(self):
        _, self.frame = self.cap.read()

        # Resize the frame to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        resized_frame = cv2.resize(self.frame, (canvas_width, canvas_height))

        self.photo = self.convert_to_photo(resized_frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.window.after(10, self.update)

    def convert_to_photo(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        return photo
    
    def open_entry_window(self):
        # ///////////////// Capture Photo /////////////////
        cam = cv2.VideoCapture(2) 
        time.sleep(1)
        result, image = cam.read() 
        if result: 
            cv2.imwrite("Temp1.png", image) 
        else: 
            print("No image detected. Please! try again") 
        # ///////////////// End Capture Photo /////////////////
            
        # ///////////////// Detect Number Plate /////////////////
        harcascade = "model/haarcascade_russian_plate_number.xml"

        # Replace 'your_image.jpg' with the path to your image file
        image_path = "Temp1.png"

        img = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if img is None:
            print(f"Error: Unable to read the image at {image_path}")
            exit()

        # Resize the image if needed
        # img = cv2.resize(img, (640, 480))

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        min_area = 500
        count = 0

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y+h, x:x+w]
                # cv2.imshow("ROI", img_roi)
                cv2.imwrite("Temp2.png", img_roi) 

        # cv2.imshow("Result", img)

        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ///////////////// End Detect Number Plate ///////////////// 
        entry_window = EntryWindow(self.window)
    def open_exit_window(self):
        cam = cv2.VideoCapture(2) 
        time.sleep(1)
        result, image = cam.read() 
        if result: 
            cv2.imwrite("Temp1.png", image) 
        else: 
            print("No image detected. Please! try again") 
        # ///////////////// End Capture Photo /////////////////
            
        # ///////////////// Detect Number Plate /////////////////
        harcascade = "/Users/yashdhingra/Documents/Python/BHEL Proj/Modules/AI_NumberPlate/Car-Number-Plates-Detection-main/model/haarcascade_russian_plate_number.xml"

        # Replace 'your_image.jpg' with the path to your image file
        image_path = "Temp1.png"

        img = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if img is None:
            print(f"Error: Unable to read the image at {image_path}")
            exit()

        # Resize the image if needed
        # img = cv2.resize(img, (640, 480))

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        min_area = 500
        count = 0

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y+h, x:x+w]
                # cv2.imshow("ROI", img_roi)
                cv2.imwrite("Temp2.png", img_roi) 

        # cv2.imshow("Result", img)

        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ///////////////// End Detect Number Plate ///////////////// 
        exit_window = ExitWindow(self.window)

    def on_close(self):
        self.cap.release()
        self.window.destroy()

# Create the main window
root = tk.Tk()
app = WebcamApp(root, "Webcam App")
