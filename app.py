import os
import base64
from openai import OpenAI
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io

def enhance_image(image_bytes: bytes,
                  sharpness_factor=2.0,
                  brightness_factor=1.2,
                  contrast_factor=1.2,
                  use_unsharp=True,
                  unsharp_radius=2,
                  unsharp_percent=150,
                  unsharp_threshold=3,
                  auto_contrast=True):
    """
    Enhances an image's clarity using default parameters.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if auto_contrast:
        image = ImageOps.autocontrast(image)
    if use_unsharp:
        image = image.filter(ImageFilter.UnsharpMask(radius=unsharp_radius,
                                                       percent=unsharp_percent,
                                                       threshold=unsharp_threshold))
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return image

def split_image_vertical(image: Image.Image) -> list:
    """
    Splits the image vertically into two halves.
    Returns a list containing the left and right halves.
    """
    width, height = image.size
    left_half = image.crop((0, 0, width // 2, height))
    right_half = image.crop((width // 2, 0, width, height))
    return [left_half, right_half]

def pil_image_to_bytes(image: Image.Image, format="JPEG"):
    """
    Converts a Pillow Image to bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()

def analyze_image_with_gpt4_vision_custom(image_bytes: bytes, custom_prompt: str, api_key: str) -> str:
    """
    Sends an image with a custom prompt to GPT-4 Vision for analysis.
    """
    try:
        client = OpenAI(api_key=api_key)
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error analyzing image: {str(e)}")

def summarize_responses(summary_prompt: str, api_key: str) -> str:
    """
    Uses GPT-4 to summarize combined responses.
    """
    try:
        client = OpenAI(api_key=api_key)
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error summarizing responses: {str(e)}")

def main():
    st.set_page_config(
        page_title="Marketing Mix Model Analysis with GPT-4 Vision",
        page_icon="📊",
        layout="wide"
    )
    st.markdown(
        """
        <style>
            .block-container { max-width: 1400px; padding: 2rem; }
        </style>
        """,
        unsafe_allow_html=True
    )
    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("📊 Marketing Mix Model Analysis")
    st.markdown("""
    <div class="info-box">
        <p>Upload your marketing mix model output images (charts, metrics, etc.) for detailed analysis.
        The AI will provide insights on model performance, key drivers, and optimization recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed prompt for full-image analysis
    detailed_prompt = (
        "You are an experienced marketing data scientist. The user has uploaded an image containing output from a Robyn marketing mix model. "
        "Please read the image carefully and pay extra attention to all numerical details. Verify every number; if there is any uncertainty or ambiguity, clearly note your assumptions. "
        "Depending on the image title, please tailor your analysis as follows:\n\n"
        "--------------------------------------------------\n"
        "If the image is titled \"One-pager for Model\", interpret the key insights from the charts provided and address the following:\n\n"
        "Overall Model Performance:\n"
        "- Evaluate key metrics such as Adjusted R², NRMSE, and DECOMP.RSSD.\n"
        "- Identify signs of model robustness or potential overfitting/underfitting from the Fitted vs. Residual plot.\n\n"
        "Channel Contributions and Efficiency:\n"
        "- Analyze the Response Decomposition Waterfall by Predictor to highlight the channels driving positive or negative incremental responses.\n"
        "- Assess the Share of Total Spend, Effect & ROAS chart to determine spending efficiency.\n"
        "- Compare channel-level ROI/ROAS in the In-cluster Bootstrapped ROAS plot and highlight any channels with low returns.\n\n"
        "Time Dynamics and Carryover Effects:\n"
        "- Use the Weibull PDF Adstock plot to interpret the flexible rate over time for each channel.\n"
        "- Examine the Immediate vs. Carryover Response Percentage plot to understand the distribution of immediate vs. long-term effects.\n\n"
        "Spend and Response Relationships:\n"
        "- Interpret the Response Curves and Mean Spends by Channel to identify diminishing returns and spending thresholds.\n"
        "- Discuss how well spend and response align for each channel.\n\n"
        "Actionable Recommendations:\n"
        "- Identify opportunities to reallocate budget toward high-ROAS channels and reduce spending on underperforming ones.\n"
        "- Suggest improvements in spend allocation based on observed diminishing returns.\n"
        "- Highlight any adjustments to model assumptions that could improve future efforts.\n\n"
        "Summary and Next Steps:\n"
        "- Provide a concise summary of the model’s performance, strengths, and limitations.\n"
        "- Suggest actionable next steps for optimizing the media mix strategy.\n"
        "- If any channel’s ROI or ROAS is significantly below expectations, recommend delaying further investment until additional data is available or improvements are identified.\n\n"
        "--------------------------------------------------\n"
        "If the image is titled \"Budget Allocation Onepager for Model\", please address these points:\n\n"
        "Budget Allocation per Paid Media:\n"
        "- Provide a table that shows the Budget Allocation per Paid Media. This table should list all channel names along with the corresponding metrics for Initial, Bounded, and Bounded x3.\n\n"
        "1. Critical Budget Allocation Insights & Business Priorities\n"
        "   - Evaluate the current distribution of the marketing budget across channels based on the Roybn Model.\n"
        "   - Identify channels with high or low incremental response and ROAS to determine if they are under- or over-invested.\n"
        "   - For budget adjustments (e.g., scenarios of 5% and 10% reallocation), clearly outline which channels meet the performance criteria for additional investment. Provide specific dollar figures (e.g., \"$10,000 for Channel A\") to illustrate recommended investment amounts and maximize ROI.\n\n"
        "2. Detailed Channel Performance & ROI Analysis\n"
        "   - Provide a side-by-side comparison of channel-specific ROAS, incremental response, and bootstrapped confidence intervals.\n"
        "   - Assess whether the current spend is proportionate to the observed incremental impact.\n"
        "   - If certain channels show unfavorable ROI/ROAS, recommend holding off on further investment until more data is available or targeted strategies are implemented. Include example figures to quantify the proposed hold or delay.\n\n"
        "3. Additional Budget Investment Strategy\n"
        "   - Determine which channel(s) offer the best potential for improved ROI with additional budget allocation.\n"
        "   - Use data-driven insights to rank channels and highlight the most critical opportunities for growth.\n"
        "   - Explicitly provide recommended dollar amounts (e.g., \"Allocate an extra $10,000 to Channel A and $8,500 to Channel B\") based on the model’s sensitivity analysis and performance metrics.\n\n"
        "4. Final Model Evaluation & Strategic Recommendations\n"
        "   - Evaluate whether the Roybn Model meets industry standards and business needs, summarizing strengths and weaknesses.\n"
        "   - Provide actionable recommendations to refine both the model and the media mix strategy.\n"
        "   - Ensure all numerical details and assumptions are clearly validated.\n\n"
        "5. Structured Budget Allocation Table\n"
        "   - Present a table that outlines the recommended spend per channel and the expected ROI.\n"
        "   - If no channel meets the threshold for a strong ROI—even with extra budget—state that further investment is not advisable and recommend alternative actions (e.g., collecting more data or adjusting campaign strategies).\n\n"
        "Table Format:\n"
        "| Channel       | Recommended Spend ($) | Expected ROI (%) |\n"
        "|--------------|----------------------|------------------|\n"
        "| Channel 1    |       $10,000        |       15%       |\n"
        "| Channel 2    |       $8,500         |       12%       |\n"
        "| Channel 3    |       $12,000        |       18%       |\n\n"
        "Note: The \"Budget Allocation per Media\" table in the image is arranged horizontally (channels are rows and metrics are columns). Please ignore the Paid Media channel."
    )


    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    enhanced_image = None

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Model Output Image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Select an image containing your marketing mix model results"
        )
        if uploaded_file is not None:
            original_image_bytes = uploaded_file.read()
            enhanced_image = enhance_image(original_image_bytes)
            st.image(enhanced_image, use_container_width=True, caption="Enhanced Image")

        # Checkbox for splitting vertically
        split_vertically = st.checkbox("Split Vertically", value=False)

    with col2:
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Your API key will not be stored and needs to be entered each time you use the application"
        )
        if not api_key and uploaded_file is not None:
            st.warning("Please enter your OpenAI API key to analyze the image.")
            st.markdown("""
            <div class="info-box">
                <p>To get your OpenAI API key:</p>
                <ol>
                    <li>Go to <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API Keys</a></li>
                    <li>Sign in or create an account</li>
                    <li>Create a new API key</li>
                    <li>Copy and paste the key here</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        analyze_button = st.button(
            "📊 Analyze Model",
            help="Click to start the analysis",
            disabled=(uploaded_file is None or not api_key)
        )

    if uploaded_file is not None and analyze_button and api_key:
        with st.spinner("Analyzing marketing mix model..."):
            try:
                final_output = ""
                if split_vertically:
                    halves = split_image_vertical(enhanced_image)
                    st.markdown("### Preview of Split Images (Vertical)")
                    st.markdown("**Left Half:**")
                    st.image(halves[0], use_container_width=True)
                    st.markdown("**Right Half:**")
                    st.image(halves[1], use_container_width=True)

                    # Custom prompts for vertical splitting
                    left_prompt = (
                        "Please analyze the following image focusing on: model performance, response decomposition waterfall by predictor plot, "
                        "share of total spend, effect & ROAS in the modeling window plot, flexible rate over time plot, and response curves and mean spends by channel plot."
                    )
                    right_prompt = (
                        "Please analyze the following image focusing on: actual vs predicted response plot, in-cluster bootstrapped ROAS (95% CI & mean) plot, "
                        "immediate vs carryover response percentage plot, and fitted vs residual plot."
                    )

                    left_bytes = pil_image_to_bytes(halves[0])
                    right_bytes = pil_image_to_bytes(halves[1])
                    left_response = analyze_image_with_gpt4_vision_custom(left_bytes, left_prompt, api_key)
                    right_response = analyze_image_with_gpt4_vision_custom(right_bytes, right_prompt, api_key)

                    combined_prompt = (
                        "Based on the following analyses:\n\n"
                        "Left/Half Analysis:\n" + left_response + "\n\n"
                        "Right/Half Analysis:\n" + right_response + "\n\n"
                        '''Please integrate the analyses from both responses and interpret the following:

Overall Model Performance:
- Evaluate key performance metrics such as Adjusted R², NRMSE, and DECOMP.RSSD to assess model accuracy and reliability.
- Examine the Fitted vs. Residual plot to identify any signs of overfitting or underfitting.
- Determine whether the model's predictive power is sufficient for strategic decisions or if further refinement is needed.

Channel Contributions and Carryover Effects:
- Assess the Share of Total Spend, Effect & ROAS chart to evaluate spending efficiency across channels.
- Use the Weibull PDF Adstock plot to analyze how advertising effectiveness changes over time.
- Examine the Immediate vs. Carryover Response Percentage plot to understand the distribution of immediate and long-term effects.

Actionable Recommendations:
- Identify opportunities to reallocate budget toward high-ROAS channels and reduce spending on underperforming ones.
- Provide recommendations for strategic spend adjustments based on diminishing returns and response curves.
- Suggest model refinements to improve future forecasts.

Summary and Next Steps:
- Provide a concise summary of the model’s performance, strengths, and limitations.
- Offer actionable next steps, such as collecting additional data or testing alternative predictors.
- Clearly state whether the model’s insights support current marketing investment decisions or if further improvements are required.
'''
                    )
                    combined_summary = summarize_responses(combined_prompt, api_key)
                    final_output = combined_summary
                    st.markdown("### Model Interpretations Summary")
                    st.markdown(f'<div class="analysis-result">{combined_summary}</div>', unsafe_allow_html=True)
                else:
                    # No splitting: analyze the full image using the detailed prompt
                    st.markdown("### Uploaded Image")
                    st.image(enhanced_image, use_container_width=True, caption="Uploaded Image")
                    enhanced_image_bytes = pil_image_to_bytes(enhanced_image)
                    analysis_result = analyze_image_with_gpt4_vision_custom(enhanced_image_bytes, detailed_prompt, api_key)
                    final_output = analysis_result
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)

                st.download_button(
                    label="Download Analysis Output",
                    data=final_output,
                    file_name="analysis_output.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.markdown("""
                    <div class="error-help">
                        Please check:
                        - Your OpenAI API key is valid
                        - The image format is supported
                        - Your internet connection is stable
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            <p>Powered by GPT-4 Vision API • Upload limit: 10MB per image</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
