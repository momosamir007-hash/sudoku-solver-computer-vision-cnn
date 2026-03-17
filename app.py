import streamlit as st
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import google.generativeai as genai

st.set_page_config(
    page_title="Sudoku Solver AI",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Sudoku Solver AI")
st.markdown("قم برفع صورة للغز سودوكو، وسيقوم الذكاء الاصطناعي باستخراجها. سيتم التحقق من الأرقام المشكوك فيها تلقائياً باستخدام **طبقة تحقق ثانية (Gemini Vision)**.")

# --- جلب مفتاح API من Secrets ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("⚠️ لم يتم العثور على مفتاح Gemini API في إعدادات الأمان (Secrets). يرجى إضافته لتعمل طبقة التحقق الثانية.")
    gemini_api_key = None

# --- القائمة الجانبية والإعدادات ---
st.sidebar.header("Configuration")
model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
selected_model_file = st.sidebar.selectbox("Select Pre-trained Model", model_files)
model_path = os.path.join("models", selected_model_file)

# --- تحميل النماذج الأساسية ---
@st.cache_resource
def load_ml_modules():
    """تحميل المكتبات الثقيلة والنماذج لتجنب بطء واجهة المستخدم."""
    from src.model.model import ConvNet
    from src.model.solver import Sudoku as solve_sudoku_algorithm
    from src.preprocess.build_features import process_sudoku_image
    from src.scripts.pipeline import overlay_digits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ConvNet, solve_sudoku_algorithm, process_sudoku_image, overlay_digits, device

def load_sudoku_model(path, ConvNet, device):
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- دوال التوقع والتحقق ---
def predict_with_confidence(model, cells, device, confidence_threshold=0.70):
    """توقع الأرقام محلياً مع حساب نسبة الثقة لكل خلية."""
    grid = []
    low_confidence_cells = [] 
    
    with torch.no_grad():
        for i, cell in enumerate(cells):
            row, col = i // 9, i % 9
            
            # حل مشكلة القنوات (تحويل 1 Channel إلى 3 Channels)
            tensor_cell = torch.FloatTensor(cell).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            
            output = model(tensor_cell)
            probabilities = F.softmax(output, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)
            
            digit = predicted_class.item()
            confidence = max_prob.item()
            
            grid.append(digit)
            
            # إذا كانت الثقة أقل من الحد، سجلها للمراجعة
            if confidence < confidence_threshold:
                low_confidence_cells.append({
                    'index': i, 'row': row, 'col': col, 
                    'digit': digit, 'confidence': confidence, 'image': cell
                })
                
    # تحويل إلى مصفوفة 9x9
    grid_9x9 = [grid[i:i+9] for i in range(0, 81, 9)]
    return grid_9x9, low_confidence_cells

def validate_cell_with_gemini(cell_image, api_key):
    """تنظيف وتكبير الخلية ثم إرسالها إلى Gemini للتحقق منها بدقة عالية."""
    # 1. تنظيف الصورة: إزالة 15% من الحواف للتخلص من خطوط الشبكة السوداء
    h, w = cell_image.shape[:2]
    crop_h, crop_w = int(h * 0.15), int(w * 0.15)
    clean_cell = cell_image[crop_h:h-crop_h, crop_w:w-crop_w]
    
    # 2. تكبير الصورة ليقرأها VLM بوضوح
    enlarged_cell = cv2.resize(clean_cell, (150, 150), interpolation=cv2.INTER_LANCZOS4)
    pil_img = Image.fromarray(enlarged_cell)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
    prompt = "This is a single cell from a Sudoku grid. What single digit (1-9) is in this image? If the cell is completely empty, blank, or you cannot read any clear digit, return 0. Output ONLY the single number and nothing else."
    
    try:
        response = model.generate_content([prompt, pil_img])
        digit = int(response.text.strip())
        if 0 <= digit <= 9:
            return digit
    except Exception as e:
        return -1 
    return -1

# --- التطبيق الرئيسي ---
uploaded_file = st.file_uploader("اختر صورة سودوكو...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.subheader("1. الصورة الأصلية")
    st.image(pil_image, width="stretch")

    if "grid_predictions" not in st.session_state:
        st.session_state.grid_predictions = None
        st.session_state.warped = None
        st.session_state.coords = None
        st.session_state.ai_logs = []

    if st.button("استخراج الشبكة والتحقق"):
        with st.spinner("جاري استخراج الشبكة وتحليل الأرقام..."):
            try:
                ConvNet_class, solve_sudoku_logic, process_image_func, overlay_func, dev = load_ml_modules()
                
                open_cv_image = np.array(pil_image) 
                if len(open_cv_image.shape) == 2:
                    image_cv2 = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
                elif len(open_cv_image.shape) == 3:
                    if open_cv_image.shape[2] == 4:
                        image_cv2 = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
                    else:
                        image_cv2 = open_cv_image[:, :, ::-1].copy() 

                # 1. المعالجة واستخراج الخلايا
                cells, coords, warped = process_image_func(image_cv2)
                
                if cells is None:
                    st.error("لم يتم العثور على شبكة سودوكو واضحة.")
                else:
                    # 2. التوقع المحلي + حساب الثقة
                    model = load_sudoku_model(model_path, ConvNet_class, dev)
                    grid_predictions, low_conf_cells = predict_with_confidence(model, cells, dev, confidence_threshold=0.70)
                    
                    st.session_state.warped = warped
                    st.session_state.coords = coords
                    st.session_state.ai_logs = []

                    # 3. التحقق باستخدام Gemini
                    if gemini_api_key and low_conf_cells:
                        st.info(f"تم العثور على {len(low_conf_cells)} خلايا مشكوك فيها. جاري التحقق التلقائي...")
                        for cell_data in low_conf_cells:
                            gemini_digit = validate_cell_with_gemini(cell_data['image'], gemini_api_key)
                            r, c = cell_data['row'], cell_data['col']
                            local_digit = cell_data['digit']
                            
                            if gemini_digit != -1 and gemini_digit != local_digit:
                                grid_predictions[r][c] = gemini_digit
                                st.session_state.ai_logs.append(f"🔄 تصحيح (الصف {r+1}، العمود {c+1}): من {local_digit} إلى {gemini_digit}.")
                            else:
                                st.session_state.ai_logs.append(f"✅ تأكيد (الصف {r+1}، العمود {c+1}): الرقم {local_digit}.")
                    elif not gemini_api_key and low_conf_cells:
                        st.warning(f"يوجد {len(low_conf_cells)} خلايا مشكوك فيها. يرجى مراجعتها يدوياً لعدم توفر مفتاح Gemini.")

                    st.session_state.grid_predictions = grid_predictions

            except Exception as e:
                import traceback
                st.error(f"حدث خطأ أثناء الاستخراج: {e}")
                st.write(traceback.format_exc())

    # --- عرض النتائج والحل ---
    if st.session_state.grid_predictions is not None:
        st.subheader("2. الشبكة المستخرجة")
        st.markdown("يرجى إلقاء نظرة سريعة. إذا كانت الأرقام صحيحة، اضغط على حل.")
        
        if st.session_state.ai_logs:
            with st.expander("تفاصيل تحقق الذكاء الاصطناعي السحابي"):
                for log in st.session_state.ai_logs:
                    st.write(log)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.warped, channels="BGR", width="stretch", caption="الشبكة المستخرجة")
            
        with col2:
            df = pd.DataFrame(st.session_state.grid_predictions, columns=[str(i) for i in range(1, 10)])
            edited_df = st.data_editor(df, use_container_width=True, hide_index=True)
            
        if st.button("حل اللغز"):
            with st.spinner("جاري الحل..."):
                _, solve_sudoku_logic, _, overlay_func, _ = load_ml_modules()
                grid_solution = edited_df.values.tolist()
                
                st.subheader("3. اللغز المحلول")
                if solve_sudoku_logic(grid_solution, 0, 0):
                    solved_image = overlay_func(st.session_state.warped, grid_solution, st.session_state.coords, color=(0, 200, 0))
                    solved_image_rgb = cv2.cvtColor(solved_image, cv2.COLOR_BGR2RGB)
                    st.image(solved_image_rgb, width="stretch")
                    st.success("تم حل اللغز بنجاح!")
                else:
                    st.error("لا يوجد حل صحيح للأرقام الحالية. يرجى التحقق من وجود أرقام مكررة أو خاطئة.")
                    
        if st.button("إعادة تعيين / رفع صورة جديدة"):
            st.session_state.grid_predictions = None
            st.rerun()
