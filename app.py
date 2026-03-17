import streamlit as st
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import subprocess
import sys

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Sudoku Solver AI",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Sudoku Solver AI")
st.markdown(
    "قم برفع صورة للغز سودوكو، وسيقوم نموذج الذكاء الاصطناعي المحلي (CNN) باستخراجها. "
    "يمكنك مراجعة الأرقام وتعديلها يدوياً في الجدول قبل حل اللغز."
)

# --- القائمة الجانبية ---
st.sidebar.header("Configuration")
models_dir = "models"

# التحقق من وجود مجلد النماذج
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

st.sidebar.info("ℹ️ طبقة التحقق السحابية معطلة حالياً. يتم استخدام النموذج المحلي فقط.")

# --- قسم تدريب النموذج في القائمة الجانبية ---
st.sidebar.divider()
st.sidebar.header("🛠️ أدوات المطور (تدريب)")
st.sidebar.info("يمكنك تدريب نموذج جديد بأحدث البيانات مباشرة من هنا.")

if st.sidebar.button("🚀 بدء تدريب نموذج جديد"):
    st.sidebar.warning("⚠️ جاري التدريب... يرجى عدم إغلاق الصفحة. قد يستغرق الأمر عدة دقائق.")
    
    # مكان فارغ لعرض سجل التدريب (Logs) في الوقت الفعلي
    log_placeholder = st.sidebar.empty()
    output_log = ""
    
    try:
        # تجهيز أمر التشغيل للسكربت
        cmd = [sys.executable, "train.py", "--model", "convnet", "--dataset", "sudoku"]
        
        # تشغيل السكربت في الخلفية وقراءة المخرجات
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # قراءة السجل سطراً بسطر وعرضه في الواجهة
        for line in process.stdout:
            output_log += line
            # عرض آخر 15 سطر لتجنب بطء الواجهة
            display_text = "\n".join(output_log.splitlines()[-15:])
            log_placeholder.code(display_text, language="bash")
            
        process.wait() # الانتظار حتى تنتهي العملية
        
        if process.returncode == 0:
            st.sidebar.success("✅ انتهى التدريب بنجاح! تم حفظ النموذج الجديد.")
            st.sidebar.info("🔄 يرجى إعادة تحميل الصفحة (Refresh) لتحديث قائمة النماذج.")
        else:
            st.sidebar.error("❌ حدث خطأ أثناء التدريب. راجع السجل أعلاه.")
            
    except Exception as e:
        st.sidebar.error(f"حدث خطأ غير متوقع: {e}")


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

# --- دوال التوقع ---
def predict_with_confidence(model, cells, device, confidence_threshold=0.70):
    grid = []
    low_confidence_cells = []
    with torch.no_grad():
        for i, cell in enumerate(cells):
            row, col = i // 9, i % 9
            h, w = cell.shape[:2]
            
            # تقليل القص إلى 10% لتجنب مسح الأرقام الكبيرة
            crop_h = int(h * 0.10)
            crop_w = int(w * 0.10)  
            
            cropped_cell = cell[
                crop_h : h - crop_h, 
                crop_w : w - crop_w
            ]
            
            clean_cell = cv2.resize(
                cropped_cell, (w, h), interpolation=cv2.INTER_AREA
            )
            
            # حفظ إحدى الخلايا للـ Debugging
            if i == 40:
                st.session_state.debug_cell = clean_cell
            
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
                    }
                )
                
    grid_9x9 = [grid[i : i + 9] for i in range(0, 81, 9)]
    return grid_9x9, low_confidence_cells

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

    if st.button("استخراج الشبكة"):
        with st.spinner("جاري المعالجة باستخدام النموذج المحلي..."):
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

                    # تنبيه المستخدم بالخلايا ذات الثقة المنخفضة
                    if low_conf_cells:
                        st.warning(
                            f"⚠️ تم العثور على {len(low_conf_cells)} خلايا مشكوك فيها. "
                            f"يرجى مراجعة الجدول أدناه وتصحيح الأخطاء يدوياً قبل الحل."
                        )
                        for cell_data in low_conf_cells:
                            r, c = cell_data["row"], cell_data["col"]
                            st.session_state.ai_logs.append(
                                f"🔍 مراجعة يدوية مطلوبة للخلية ({r+1},{c+1}): "
                                f"الرقم المتوقع {cell_data['digit']} (نسبة الثقة: {cell_data['confidence']:.2f})"
                            )

                    st.session_state.grid_predictions = grid_predictions

            except Exception as e:
                import traceback
                st.error(f"خطأ أثناء المعالجة: {e}")
                st.code(traceback.format_exc())

# --- عرض النتائج ---
if st.session_state.grid_predictions is not None:
    st.subheader("2. الشبكة المستخرجة (قابلة للتعديل)")
    
    if st.session_state.debug_cell is not None:
        st.markdown("**🔍 نظرة النموذج لخلايا السودوكو (للتأكد من الألوان والقص):**")
        st.image(st.session_state.debug_cell, width=150, caption="خلية اختبارية (مُعالجة)")
    
    if st.session_state.ai_logs:
        with st.expander("تفاصيل الخلايا المشكوك فيها (تحتاج مراجعتك)"):
            for log in st.session_state.ai_logs:
                st.write(log)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            st.session_state.warped,
            channels="BGR",
            use_container_width=True,
            caption="صورة الشبكة المقصوصة (راجعها مع الجدول)",
        )
        
    with col2:
        df = pd.DataFrame(
            st.session_state.grid_predictions,
            columns=[str(i) for i in range(1, 10)],
        )
        # الجدول التفاعلي للسماح للمستخدم بتصحيح الأرقام
        edited_df = st.data_editor(
            df, use_container_width=True, hide_index=True
        )

    if st.button("حل اللغز"):
        with st.spinner("جاري الحل..."):
            _, solve_sudoku_logic, _, overlay_func, _ = load_ml_modules()
            
            # أخذ القيم بعد التعديل اليدوي
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
                st.error("لا يوجد حل رياضي لهذه الأرقام. يرجى التحقق من عدم وجود أرقام مكررة أو خاطئة في الجدول قبل الضغط على حل.")

    if st.button("إعادة تعيين / صورة جديدة"):
        st.session_state.grid_predictions = None
        st.session_state.uploaded_filename = None
        st.session_state.debug_cell = None
        st.rerun()
