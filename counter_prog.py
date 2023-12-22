import tkinter as tk
import csv 


class VehicleCounterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Vehicle Counter")
        self.master.geometry("700x700")  # Set window size

        self.cars_count = 0
        self.trucks_count = 0

        self.cars_label = tk.Label(master, text="Cars: 0")
        self.cars_label.pack(side="top", anchor="n", pady=(50, 0))  # Center vertically

        self.trucks_label = tk.Label(master, text="Trucks: 0")
        self.trucks_label.pack(side="top", anchor="n", pady=(10, 0))  # Center vertically

        # Schedule the update_counts function to run every 1000 milliseconds (1 second)
        self.master.after(1000, self.update_counts)

    def update_counts(self):
        # In a real-world scenario, you might fetch the counts from an external source.
        # For simplicity, I'm just incrementing the counts here.
        self.cars_count += 1
        self.trucks_count += 1
        csv_file_path = 'entry_exit.csv'

        try:
            with open(csv_file_path, mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                # next(csv_reader)  # Skip the header row

                # Initialize counters
                car_count = 0
                truck_count = 0
                
                # Iterate through the rows and count cars and trucks
                for row in csv_reader:
                    vehicle_type = row[2].lower()
                    # print(row)  # Assuming vehicle type is in the third column
                    if "cars" in vehicle_type and row[1] == '0':
                        car_count += 1
                    elif "trucks" in vehicle_type and row[1] == '0':
                        truck_count += 1

            print(f"Number of cars: {car_count}")
            print(f"Number of trucks: {truck_count}")

        except Exception as e:
            print(f"Error: {e}")
            
        self.cars_count = car_count
        self.trucks_count = truck_count

        # Update the labels with the new counts
        self.cars_label.config(text=f"Cars: {self.cars_count}")
        self.trucks_label.config(text=f"Trucks: {self.trucks_count}")

        # Schedule the update_counts function to run again after 1000 milliseconds (1 second)
        self.master.after(1000, self.update_counts)

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.mainloop()
