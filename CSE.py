import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import LinearRegression
import numpy as np

# Historical data
year = np.array([2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
categories = {
    "SM": [2194, 1965, 1686, 2039, 2660],
    "EZ": [2843, 2218, 2000, 2847, 3381],
    "MU": [2227, 2151, 1822, 2526, 3086],
    "LA": [3315, 4956, 2994, 5180, 4754],
    "DV": [5628, 5731, 6287, 7884, 5674],
    "VK": [3320, 2426, 2521, 2452, 4247],
    "BH": [2904, 2916, 2516, 2343, 3184],
    "BX": [4871, 8703, 8305, 7804, 8565],
    "KN": [42082, 18934, 17350, 15809, 16224],
    "KU": [22809, 7446, 7000, 6065, 10352],
    "SC": [26000, 25542, 20356, 19265, 20516],
    "ST": [13651, 41394, 34619, 54300, 14003],
}

# Predict cutoff for 2025 using linear regression
predicted_2025 = {}
for cat, ranks in categories.items():
    model = LinearRegression()
    model.fit(year, ranks)
    pred = model.predict(np.array([[2025]]))[0]
    predicted_2025[cat.upper()] = int(round(pred))

# Create GUI
def check_eligibility():
    try:
        user_rank = int(entry_rank.get())
        user_cat = entry_category.get().strip().upper()
        
        if user_cat not in predicted_2025:
            messagebox.showerror("Error", "Invalid category code!")
            return
        
        cutoff = predicted_2025[user_cat]
        
        if user_rank <= cutoff:
            result = f"✅ Seat Possible!\nPredicted Cutoff: {cutoff}"
        else:
            result = f"❌ Seat Not Likely\nPredicted Cutoff: {cutoff}"
            
        messagebox.showinfo("Result", result)
    
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid rank.")

# Tkinter window
root = tk.Tk()
root.title("KEAM 2025 MACE CS Admission Checker")
root.geometry("400x300")

tk.Label(root, text="Enter your KEAM 2025 Rank:", font=("Arial", 12)).pack(pady=10)
entry_rank = tk.Entry(root, font=("Arial", 12))
entry_rank.pack(pady=5)

tk.Label(root, text="Enter your Category Code (e.g., SM, EZ, MU):", font=("Arial", 12)).pack(pady=10)
entry_category = tk.Entry(root, font=("Arial", 12))
entry_category.pack(pady=5)

tk.Button(root, text="Check Admission Possibility", font=("Arial", 12), command=check_eligibility).pack(pady=20)

root.mainloop()
