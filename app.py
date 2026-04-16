from pathlib import Path
import runpy


# Allows `streamlit run app.py` to launch the real app in app/app.py.
runpy.run_path(str(Path(__file__).parent / "app" / "app.py"), run_name="__main__")
