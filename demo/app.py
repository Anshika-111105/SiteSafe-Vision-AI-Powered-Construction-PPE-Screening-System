import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import gradio as gr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import get_predictor
from app.risk import evaluate_risk_and_recommendation

DISCLAIMER_TEXT = (
    "⚠️ **LEGAL & REGULATORY DISCLAIMER**: This tool is an AI-based screening aid and is not a substitute "
    "for professional workplace safety inspection, certified safety officer assessment, or regulatory compliance verification."
)


def predict_ppe_image(input_image: Image.Image, show_gradcam: bool = True):
    if input_image is None:
        return None, "No Image Provided", "0.0%", "N/A", "Please upload an image to begin screening.", {}, None

    predictor = get_predictor()
    if not predictor.is_loaded:
        return None, "ERROR", "0.0%", "HIGH", "Model checkpoint unavailable. Ensure production model is trained.", {}, None

    try:
        if show_gradcam:
            pred_res, gradcam_overlay = predictor.predict_with_gradcam(input_image)
        else:
            pred_res = predictor.predict(input_image)
            gradcam_overlay = input_image

        prediction = pred_res["prediction"]
        confidence = pred_res["confidence"]
        probabilities = pred_res["probabilities"]
        latency = pred_res["inference_latency_ms"]

        risk_level, recommendation = evaluate_risk_and_recommendation(prediction, confidence)

        conf_str = f"{confidence * 100:.1f}%"
        rec_formatted = f"**{recommendation}**\n\n*(Model: {pred_res['model_name']} | Latency: {latency:.1f}ms)*"

        return (
            gradcam_overlay,
            prediction,
            conf_str,
            risk_level,
            rec_formatted,
            probabilities,
        )
    except Exception as e:
        return None, "ERROR", "0.0%", "HIGH", f"Prediction Error: {str(e)}", {}, None


# Gather sample examples from interim crops
examples_dir = PROJECT_ROOT / "data" / "interim" / "crops"
example_paths = []
if examples_dir.exists():
    example_paths = [[str(p), True] for p in sorted(list(examples_dir.glob("*.jpg")))[:6]]

custom_css = """
.gradio-container {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}
.risk-high { background-color: #fee2e2; color: #991b1b; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
.risk-med { background-color: #fef3c7; color: #92400e; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
.risk-low { background-color: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
"""

with gr.Blocks(title="SiteSafe Vision | PPE Screening System", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦺 SiteSafe Vision")
    gr.Markdown("### AI-Powered Construction PPE Compliance Screening System")
    gr.Markdown(f"> {DISCLAIMER_TEXT}")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Worker Crop Image Input")
            gradcam_toggle = gr.Checkbox(value=True, label="Generate Grad-CAM Explainability Heatmap")
            submit_btn = gr.Button("🔍 Run PPE Screening", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Visual Analysis (Grad-CAM Overlay)")
            with gr.Row():
                pred_label = gr.Textbox(label="Compliance Prediction", interactive=False)
                conf_label = gr.Textbox(label="Confidence", interactive=False)
                risk_label = gr.Textbox(label="Risk Level", interactive=False)

            rec_box = gr.Markdown(label="Safety Recommendation")
            prob_chart = gr.Label(label="Class Probability Distribution", num_top_classes=3)

    if example_paths:
        gr.Examples(
            examples=example_paths,
            inputs=[input_img, gradcam_toggle],
            outputs=[output_img, pred_label, conf_label, risk_label, rec_box, prob_chart],
            fn=predict_ppe_image,
            cache_examples=False,
            label="Representative Screening Examples",
        )

    submit_btn.click(
        fn=predict_ppe_image,
        inputs=[input_img, gradcam_toggle],
        outputs=[output_img, pred_label, conf_label, risk_label, rec_box, prob_chart],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
