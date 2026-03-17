import streamlit as st
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import google.generativeai as genai

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Sudoku Solver AI",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Sudoku Solver AI")
st.markdown(
    "قم برفع صورة للغز سودوكو، وسيقوم الذكاء الاصطناعي باستخراجها. "
    "سيتم التحقق من الأرقام المشكوك فيها تلقائياً باستخدام "
    "**طبقة تحقق ثانية (Gemini Vision 2.5)**."
)

# --- جلب مفتاح API من Secrets ---
gemini_api_key = None
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)  # إعداد المفتاح مرة واحدة
except KeyError:
    st.sidebar.error(
        "⚠️ لم يتم العثور على مفتاح Gemini API. "
        "يرجى إضافته في إعدادات Secrets لتعمل طبقة التحقق الثانية."
    )

# --- القائمة الجانبية ---
st.sidebar.header("Configuration")
models_dir = "models"
if not os.path.exists(models_dir):
    st.sidebar.error("⚠️ مجلد 'models' غير موجود!")
    st.stop()

model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
if not model_files:
    st.sidebar.error("⚠️ لا توجد نماذج في المجلد!")
    st.stop()

selected_model_file = st.sidebar.selectbox(
    "Select Pre-trained Model", model_files
)
model_path = os.path.join(models_dir, selected_model_file)

if gemini_api_key:
    st.sidebar.success("✅ مفتاح Gemini متصل وجاهز للتحقق.")

# --- تحميل النماذج الأساسية ---
@st.cache_resource
def load_ml_modules():
    from src.model.model import ConvNet
    from src.model.solver import Sudoku as solve_sudoku_algorithm
    from src.preprocess.build_features import process_sudoku_image
    from src.scripts.pipeline import overlay_digits

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return (
        ConvNet,
        solve_sudoku_algorithm,
        process_sudoku_image,
        overlay_digits,
        device,
    )

@st.cache_resource
def load_sudoku_model(path, _ConvNet, device):
    """مخزن مؤقتاً - لا يُعاد تحميله كل مرة"""
    model = _ConvNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- دوال التوقع والتحقق ---
def predict_with_confidence(model, cells, device, confidence_threshold=0.70):
    grid = []
    low_confidence_cells = []
    with torch.no_grad():
        for i, cell in enumerate(cells):
            row, col = i // 9, i % 9
            h, w = cell.shape[:2]
            
            # ✅ تم تقليل القص إلى 10% لتجنب مسح الأرقام الكبيرة
            crop_h = int(h * 0.10)
            crop_w = int(w * 0.10)  
            
            cropped_cell = cell[
                crop_h : h - crop_h, 
                crop_w : w - crop_w
            ]
            
            clean_cell = cv2.resize(
                cropped_cell, (w, h), interpolation=cv2.INTER_AREA
            )
            
            # --- ميزة التصحيح (Debugging) ---
            # حفظ إحدى الخلايا لنعرضها للمستخدم ليرى ما يراه النموذج
            if i == 40: # الخلية رقم 40 تقع في منتصف الشبكة تقريباً
                st.session_state.debug_cell = clean_cell
            # ---------------------------------
            
            # ✅ إزالة القسمة على 255.0 (تمرير الصورة كما هي)
            tensor_cell = (
                torch.FloatTensor(clean_cell)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 3, 1, 1)
                .to(device)
            )
            
            output = model(tensor_cell)
            probabilities = F.softmax(output, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)
            digit = predicted_class.item()
            confidence = max_prob.item()
            
            grid.append(digit)
            if confidence < confidence_threshold:
                low_confidence_cells.append(
                    {
                        "index": i,
                        "row": row,
                        "col": col,
                        "digit": digit,
                        "confidence": confidence,
                        "image": clean_cell,
                    }
                )
                
    grid_9x9 = [grid[i : i + 9] for i in range(0, 81, 9)]
    return grid_9x9, low_confidence_cells


def validate_cell_with_gemini(cell_image):
    """إرسال الخلية إلى Gemini 2.5 للتحقق"""
    enlarged_cell = cv2.resize(
        cell_image, (150, 150), interpolation=cv2.INTER_LANCZOS4
    )
    pil_img = Image.fromarray(enlarged_cell)
    
    # ✅ تحديث النموذج إلى gemini-2.5-flash ليعمل بدون أخطاء 404
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "This is a single cell from a Sudoku grid. "
        "What single digit (1-9) is in this image? "
        "If the cell is completely empty, blank, or "
        "you cannot read any clear digit, return 0. "
        "Output ONLY the single number and nothing else."
    )
    try:
        response = model.generate_content([prompt, pil_img])
        digit = int(response.text.strip())
        if 0 <= digit <= 9:
            return digit
    except ValueError:
        return -1
    except Exception as e:
        st.warning(f"خطأ Gemini: {str(e)[:80]}")
        return -1
    return -1


# --- Session State الأولية ---
if "grid_predictions" not in st.session_state:
    st.session_state.grid_predictions = None
    st.session_state.warped = None
    st.session_state.coords = None
    st.session_state.ai_logs = []
    st.session_state.uploaded_filename = None
    st.session_state.debug_cell = None

# --- التطبيق الرئيسي ---
uploaded_file = st.file_uploader(
    "اختر صورة سودوكو...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # إعادة تعيين عند رفع صورة جديدة
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.grid_predictions = None
        st.session_state.warped = None
        st.session_state.coords = None
        st.session_state.ai_logs = []
        st.session_state.debug_cell = None

    pil_image = Image.open(uploaded_file)
    st.subheader("1. الصورة الأصلية")
    st.image(pil_image, use_container_width=True)

    if st.button("استخراج الشبكة والتحقق"):
        with st.spinner("جاري المعالجة..."):
            try:
                (
                    ConvNet_class,
                    solve_sudoku_logic,
                    process_image_func,
                    overlay_func,
                    dev,
                ) = load_ml_modules()

                open_cv_image = np.array(pil_image)
                if len(open_cv_image.shape) == 2:
                    image_cv2 = cv2.cvtColor(
                        open_cv_image, cv2.COLOR_GRAY2BGR
                    )
                elif open_cv_image.shape[2] == 4:
                    image_cv2 = cv2.cvtColor(
                        open_cv_image, cv2.COLOR_RGBA2BGR
                    )
                else:
                    image_cv2 = open_cv_image[:, :, ::-1].copy()

                cells, coords, warped = process_image_func(image_cv2)

                if cells is None:
                    st.error("لم يتم العثور على شبكة سودوكو واضحة.")
                else:
                    model = load_sudoku_model(
                        model_path, ConvNet_class, dev
                    )
                    
                    grid_predictions, low_conf_cells = predict_with_confidence(
                        model, cells, dev
                    )
                    
                    st.session_state.warped = warped
                    st.session_state.coords = coords
                    st.session_state.ai_logs = []

                    # التحقق باستخدام Gemini مع شريط تقدم
                    if gemini_api_key and low_conf_cells:
                        st.info(
                            f"تم العثور على {len(low_conf_cells)} "
                            f"خلايا تحتاج تأكيد من Gemini..."
                        )
                        progress = st.progress(0)
                        
                        for idx, cell_data in enumerate(low_conf_cells):
                            gemini_digit = validate_cell_with_gemini(
                                cell_data["image"]
                            )
                            r, c = cell_data["row"], cell_data["col"]
                            local_digit = cell_data["digit"]

                            if gemini_digit != -1 and gemini_digit != local_digit:
                                grid_predictions[r][c] = gemini_digit
                                st.session_state.ai_logs.append(
                                    f"🔄 تصحيح ({r+1},{c+1}): "
                                    f"توقع محلي {local_digit} ← تصحيح Gemini {gemini_digit}"
                                )
                            else:
                                st.session_state.ai_logs.append(
                                    f"✅ تأكيد ({r+1},{c+1}): "
                                    f"الرقم {local_digit}"
                                )
                                
                            progress.progress((idx + 1) / len(low_conf_cells))
                            
                        progress.empty()

                    elif not gemini_api_key and low_conf_cells:
                        st.warning(
                            f"يوجد {len(low_conf_cells)} خلايا "
                            f"مشكوك فيها. راجعها يدوياً لعدم توفر مفتاح Gemini."
                        )

                    st.session_state.grid_predictions = grid_predictions

            except Exception as e:
                import traceback
                st.error(f"خطأ أثناء المعالجة: {e}")
                st.code(traceback.format_exc())

# --- عرض النتائج ---
if st.session_state.grid_predictions is not None:
    st.subheader("2. الشبكة المستخرجة")
    
    # ✅ عرض صورة الخلية التجريبية للتشخيص
    if st.session_state.debug_cell is not None:
        st.markdown("**🔍 نظرة النموذج لخلايا السودوكو (للتأكد من الألوان والقص):**")
        st.image(st.session_state.debug_cell, width=150, caption="خلية اختبارية (مُعالجة)")
    
    if st.session_state.ai_logs:
        with st.expander("تفاصيل تحقق الذكاء الاصطناعي (Gemini Logs)"):
            for log in st.session_state.ai_logs:
                st.write(log)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            st.session_state.warped,
            channels="BGR",
            use_container_width=True,
            caption="صورة الشبكة المقصوصة",
        )
        
    with col2:
        df = pd.DataFrame(
            st.session_state.grid_predictions,
            columns=[str(i) for i in range(1, 10)],
        )
        edited_df = st.data_editor(
            df, use_container_width=True, hide_index=True
        )

    if st.button("حل اللغز"):
        with st.spinner("جاري الحل..."):
            _, solve_sudoku_logic, _, overlay_func, _ = load_ml_modules()
            
            grid_solution = edited_df.values.astype(int).tolist()

            st.subheader("3. اللغز المحلول")
            if solve_sudoku_logic(grid_solution, 0, 0):
                solved_image = overlay_func(
                    st.session_state.warped,
                    grid_solution,
                    st.session_state.coords,
                    color=(0, 200, 0),
                )
                solved_image_rgb = cv2.cvtColor(
                    solved_image, cv2.COLOR_BGR2RGB
                )
                st.image(
                    solved_image_rgb, use_container_width=True,
                )
                st.success("تم حل اللغز بنجاح! 🎉")
            else:
                st.error("لا يوجد حل رياضي لهذه الأرقام. يرجى التحقق من عدم وجود أرقام مكررة أو خاطئة في الجدول.")

    if st.button("إعادة تعيين / صورة جديدة"):
        st.session_state.grid_predictions = None
        st.session_state.uploaded_filename = None
        st.session_state.debug_cell = None
        st.rerun()
