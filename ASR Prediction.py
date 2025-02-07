import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Load the saved XGB model
XGB_Reg = joblib.load('ASR_XGB_model.sav')

# Function to predict y for given inputs
def predict_moments(inputs):
    times = np.arange(0, 800, 50)
    predictions = []
    for time in times:
        input_data = np.array([[*inputs, time]])  # Time as the last input
        prediction = XGB_Reg.predict(input_data)
        predictions.append(prediction[0])
    return times, predictions

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time vs ASR")
        
        # Input labels and entry fields
        self.input_labels = [
            "Silica Content (%)", "Reactive Aggregate Fraction (%)", 
            "Avg. Size of Reactive Aggregate (mm)", "Water to Cement Ratio", 
            "Alkali Content (Kg/m³)", "Temperature Type", "Temperature (°C)", 
            "Relative Humidity (%)", "Section Area (mm²)", "Specimen Length (mm)"
        ]
        self.input_entries = []
        for i, label in enumerate(self.input_labels):
            ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=10)
            entry = ttk.Entry(root)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.input_entries.append(entry)
        
        # Prediction button
        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.input_labels), columnspan=2, padx=5, pady=5)

    def get_input_values(self):
        inputs = [float(entry.get()) for entry in self.input_entries]
        return inputs
    
    def predict(self):
        inputs = self.get_input_values()
        times, predictions = predict_moments(inputs)
        self.plot_curve(times, predictions)
    
    def plot_curve(self, times, predictions):
        plt.figure()
        plt.plot(times, predictions, color='red')  # Set line color to red
        plt.xlabel("Time (days)", fontname='Arial', fontsize=12)  # Set x-axis label font
        plt.ylabel("ASR", fontname='Arial', fontsize=12)  # Set y-axis label font
        plt.grid(True)
        plt.show()

# Create Tkinter window
root = tk.Tk()
app = PredictionApp(root)
root.mainloop()