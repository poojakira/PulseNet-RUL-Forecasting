import os
import glob
import joblib
import skops.io as sio

files = glob.glob('models/**/*.joblib', recursive=True) + glob.glob('**/*.joblib', recursive=True)
for file in set(files):
    print(f"Converting {file} to skops...")
    try:
        obj = joblib.load(file)
        new_file = file.replace('.joblib', '.skops')
        sio.dump(obj, new_file)
        os.remove(file)
    except Exception as e:
        print(f"Failed to convert {file}: {e}")
