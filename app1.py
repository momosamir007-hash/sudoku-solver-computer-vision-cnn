import streamlit as st
import numpy as np
import os
from PIL import Image

st.set_page_config(
    page_title="Sudoku Solver AI",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Sudoku Solver AI")
st.markdown("Upload a picture of a Sudoku puzzle, and watch our AI extract and solve it step by step!")

# Sidebar for options
st.sidebar.header("Configuration")
model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
selected_model_file = st.sidebar.selectbox("Select Pre-trained Model", model_files)
model_path = os.path.join("models", selected_model_file)

# File uploader
uploaded_file = st.file_uploader("Choose a Sudoku image...", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_ml_modules():
    """Import heavy ML operations only when needed to avoid blocking Streamlit UI startup."""
    import cv2
    import torch
    from src.model.model import ConvNet
    from src.model.solver import Sudoku as solve_sudoku_algorithm
    from src.preprocess.build_features import process_sudoku_image
    from src.scripts.pipeline import predict_grid, overlay_digits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cv2, torch, ConvNet, solve_sudoku_algorithm, process_sudoku_image, predict_grid, overlay_digits, device

def load_sudoku_model(path, ConvNet, torch, device):
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

if uploaded_file is not None:
    # Read the uploaded image using PIL first for fast UI rendering
    pil_image = Image.open(uploaded_file)
    
    st.subheader("1. Original Image")
    st.image(pil_image, width="stretch")

    # State management for digit predictions
    if "grid_predictions" not in st.session_state:
        st.session_state.grid_predictions = None
        st.session_state.warped = None
        st.session_state.coords = None

    if st.button("Extract Grid"):
        with st.spinner("Loading ML models for the first time might take a moment. Processing image..."):
            try:
                # Load heavy dependencies only on demand
                cv2_module, torch_module, ConvNet_class, solve_sudoku_logic, process_image_func, predict_func, overlay_func, dev = load_ml_modules()
                
                # Convert PIL image back to OpenCV format
                open_cv_image = np.array(pil_image) 
                
                # Check if image is grayscale (2D) or RGB (3D)
                if len(open_cv_image.shape) == 2:
                    # It's grayscale; convert it to 3-channel BGR as expected by the pipeline
                    image_cv2 = cv2_module.cvtColor(open_cv_image, cv2_module.COLOR_GRAY2BGR)
                elif len(open_cv_image.shape) == 3:
                    if open_cv_image.shape[2] == 4:
                        # RGBA to BGR
                        image_cv2 = cv2_module.cvtColor(open_cv_image, cv2_module.COLOR_RGBA2BGR)
                    else:
                        # Convert RGB to BGR 
                        image_cv2 = open_cv_image[:, :, ::-1].copy() 
                else:
                    st.error("Unsupported image format.")
                    st.stop() 

                # 1. Extract Grid and Digits
                cells, coords, warped = process_image_func(image_cv2)
                
                if cells is None or coords is None or warped is None:
                    st.error("Failed to detect a valid Sudoku grid in the image. Please try a clearer picture.")
                else:
                    # 2. Load Model & Predict
                    model = load_sudoku_model(model_path, ConvNet_class, torch_module, dev)
                    grid_predictions = predict_func(model, cells)
                    
                    st.session_state.grid_predictions = grid_predictions
                    st.session_state.warped = warped
                    st.session_state.coords = coords
            except Exception as e:
                import traceback
                st.error(f"An error occurred during extraction: {e}")
                st.write(traceback.format_exc())

    if st.session_state.grid_predictions is not None:
        st.subheader("2. Extracted Grid")
        st.markdown("The AI has interpreted the grid as follows. **Please double-check and correct any misread digits before solving:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.warped, channels="BGR", width="stretch", caption="Detected Sudoku Grid")
            
        with col2:
            import pandas as pd
            df = pd.DataFrame(st.session_state.grid_predictions, columns=[str(i) for i in range(1, 10)])
            # Display editable table
            edited_df = st.data_editor(df, use_container_width=True, hide_index=True)
            
        if st.button("Solve Corrected Puzzle"):
            with st.spinner("Solving..."):
                cv2_module, _, _, solve_sudoku_logic, _, _, overlay_func, _ = load_ml_modules()
                # Convert edited dataframe back to grid list
                grid_solution = edited_df.values.tolist()
                
                st.subheader("3. Solved Puzzle")
                # solve_sudoku_algorithm modifies the list in place
                if solve_sudoku_logic(grid_solution, 0, 0):
                    solved_image = overlay_func(st.session_state.warped, grid_solution, st.session_state.coords, color=(0, 200, 0))
                    # Convert to RGB just to be absolutely sure Streamlit renders it correctly
                    solved_image_rgb = cv2_module.cvtColor(solved_image, cv2_module.COLOR_BGR2RGB)
                    st.image(solved_image_rgb, width="stretch")
                    st.success("Puzzle solved successfully!")
                else:
                    st.error("There is no valid solution for the numbers you provided. Please check for duplicate or incorrect digits.")
                    
        if st.button("Clear / Reset Selection"):
            st.session_state.grid_predictions = None
            st.rerun()
