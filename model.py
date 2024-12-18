import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import tkinter as tk
from tkinter import messagebox


# Load and preprocess the dataset
data = pd.read_csv('mental_health_data.csv')  # Change 'mental_health_data.csv' to your actual file name

# Check the DataFrame structure
print(data.head())
print(data.columns)  # Print column names for debugging

# Preprocess categorical columns using Label Encoding
label_encoders = {}
categorical_columns = ['Gender', 'Sleep Schedule', 'Exercise Frequency', 
                       'Family History of Mental Illness', 'Substance Use', 'Mood']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string to handle NaN
    label_encoders[column] = le

# Select features and target variable
features = ['Age', 'Gender', 'Daily Screen Time (hours)', 
            'Do you have Anxiety?', 'Do you have Panic attack?', 
            'Sleep Schedule', 'Exercise Frequency', 
            'Family History of Mental Illness', 
            'Substance Use', 'Stress Levels', 'Work-Life Balance']
X = data[features]
y = data['Mood']  # Assuming 'Mood' is the target variable, change if needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
dump(model, 'mental_health_model.joblib')


# Function to predict mental health based on user input
def predict_mental_health(user_data):
    input_data = np.array(user_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]


# Create a simple questionnaire interface using Tkinter
def on_predict_button_click():
    try:
        user_data = [
            float(entry_age.get()),
            int(entry_gender.get()),  # Ensure you validate this input before using it
            float(entry_screen_time.get()),
            int(entry_anxiety.get()),
            int(entry_panic_attack.get()),
            int(entry_sleep_schedule.get()),
            int(entry_exercise_frequency.get()),
            int(entry_family_history.get()),
            int(entry_substance_use.get()),
            int(entry_stress_levels.get()),
            int(entry_work_life_balance.get())
        ]
        
        prediction = predict_mental_health(user_data)
        result_text = "You are : " + ("Slightly Depressed" if prediction == 1 else "Prone To Be Depressed")
        messagebox.showinfo('Prediction Result', result_text)

    except Exception as e:
        messagebox.showerror('Error', str(e))



window = tk.Tk()
window.title('Mental Health Predictor')
window.geometry('400x600')  
window.resizable(False, False) 

# Create a canvas and add a vertical scrollbar
canvas = tk.Canvas(window)
scrollbar = tk.Scrollbar(window, orient='vertical', command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold the form
form_frame = tk.Frame(canvas)

# Add the form frame to the canvas
canvas.create_window((0, 0), window=form_frame, anchor='nw')

# Place the canvas and scrollbar in the window
canvas.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

# Function to create and pack labels and entry fields
def create_labeled_input(label_text, input_type, options=None):
    label = tk.Label(form_frame, text=label_text)
    label.pack(pady=(10, 5))
    
    if input_type == 'entry':
        entry = tk.Entry(form_frame)
        entry.pack(pady=(0, 10))
        return entry
    elif input_type == 'dropdown':
        entry = tk.OptionMenu(form_frame, *options)
        entry.pack(pady=(0, 10))
        return entry


# Create form entries
entry_age = create_labeled_input('Enter your age:', 'entry')
entry_gender = create_labeled_input('Choose Your Gender (0 for Female, 1 for Male, 2 for Other):', 'entry')
entry_screen_time = create_labeled_input('Enter Daily Screen Time (in hours):', 'entry')
entry_anxiety = create_labeled_input('Do you have anxiety? (1 for Yes, 0 for No):', 'entry')
entry_panic_attack = create_labeled_input('Do you have panic attacks? (1 for Yes, 0 for No):', 'entry')
entry_sleep_schedule = create_labeled_input('How is your sleep schedule? (1 for Regular, 0 for Poor):', 'entry')
entry_exercise_frequency = create_labeled_input('Exercise Frequency (1 for Regular, 0 for Rarely):', 'entry')
entry_family_history = create_labeled_input('Family History of Mental Illness (1 for Yes, 0 for No):', 'entry')
entry_substance_use = create_labeled_input('Substance Use (1 for Yes, 0 for No):', 'entry')
entry_stress_levels = create_labeled_input('Stress Levels (1 for High, 0 for Low):', 'entry')
entry_work_life_balance = create_labeled_input('Work-Life Balance (1 for Good, 0 for Poor):', 'entry')


# Create the predict button
predict_button = tk.Button(form_frame, text='Predict', command=on_predict_button_click)
predict_button.pack(pady=20)

# Update the scroll region to include all widgets
form_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Start the Tkinter event loop
window.mainloop()
