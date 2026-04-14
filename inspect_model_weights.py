import pickle
import os

def load_and_inspect(file_path):
    # Check if the file actually exists first
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("--- Pickle File Loaded Successfully ---")
        print(f"Type of object: {type(data)}")
    
        # Inspection logic based on common data types
        if isinstance(data, dict):
            print(f"Keys found: {list(data.keys())}")
            for key, value in data.items():
                print(f"  - {key}: {type(value)} (Size/Length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
        
        elif isinstance(data, list):
            print(f"Length of list: {len(data)}")
            if len(data) > 0:
                print(f"Type of first element: {type(data[0])}")
        
        # Print a preview of the data
        print("\n--- Data Preview ---")
        print(data)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    path = "./experiments/params_100.pkl"
    load_and_inspect(path)