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

    # Convert image to RGB if it's not already (to avoid RGBA issues)
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

def split_image(image: Image.Image) -> list:
    """
    Splits the image vertically into two halves.
    Returns a list containing the left half and right half.
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
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
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
            model="gpt-4-turbo",
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
            .block-container {
                max-width: 1400px;
                padding: 2rem;
            }
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
        # Checkbox for splitting the image into halves
        split_image_option = st.checkbox("Split image into halves before analysis", value=False)

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
                if split_image_option:
                    # Split image vertically (left/right halves)
                    halves = split_image(enhanced_image)
                    st.markdown("### Preview of Split Images")
                    st.markdown("**Left Half:**")
                    st.image(halves[0], use_column_width=True)
                    st.markdown("**Right Half:**")
                    st.image(halves[1], use_column_width=True)

                    # Define different prompts for each half
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
                        "Left Half Analysis:\n" + left_response + "\n\n"
                        "Right Half Analysis:\n" + right_response + "\n\n"
                        '''Please integrate the analyses from both responses and interpret the following:
                        
                Overall Model Performance:
Evaluate key performance metrics such as Adjusted R², NRMSE, and DECOMP.RSSD to assess model accuracy and reliability.
Examine the Fitted vs. Residual plot to identify any signs of overfitting or underfitting, ensuring the model’s robustness.
Determine whether the model's predictive power is sufficient to inform business decisions or if further refinement is required.
Clearly state if the model is strong enough to proceed with marketing investments or if additional improvements (e.g., more data, tuning hyperparameters, refining assumptions) are needed before making strategic decisions.

                Channel Contributions and Carryover Effects:
Assess the Share of Total Spend, Effect & ROAS chart to evaluate spending efficiency across different channels.
Use the Weibull PDF Adstock plot to analyze how advertising effectiveness changes over time and its implications for future media planning.
Examine the Immediate vs. Carryover Response Percentage plot to understand the split between short-term and long-term effects across different channels.

                Actionable Recommendations:
Budget Optimization: Identify opportunities to reallocate budget toward high-ROAS channels and reduce spending on underperforming channels.
Strategic Spend Adjustments: Provide recommendations on spend allocation strategy based on observed diminishing returns and response curves.
Model Refinement: Highlight potential adjustments to model assumptions or inputs that could improve future modeling efforts.

                Summary and Next Steps:
Provide a concise evaluation of the model’s performance, strengths, and limitations.
Offer actionable next steps to optimize the media mix strategy, such as collecting additional data, testing alternative predictors, or validating key assumptions.
If any channel’s ROI or ROAS is significantly below expectations, recommend delaying further investment until more data is available or until specific improvements are identified.
Clearly conclude whether the model’s insights are reliable enough to proceed with marketing investment decisions or whether further refinements are required before taking action.
'''
                    )
                    
                    combined_summary = summarize_responses(combined_prompt, api_key)

                    st.markdown("### Model Interpretations Summary")
                    st.markdown(f'<div class="analysis-result">{combined_summary}</div>', unsafe_allow_html=True)
                else:
                    # Non-split analysis: use the general default prompt
                    default_prompt = """You are an experienced marketing data scientist. The user has uploaded an image containing output from a Robyn marketing mix model. Please read the image carefully and pay extra attention to all numerical details. Verify every number; if there is any uncertainty or ambiguity, clearly note your assumptions. Depending on the image title, please tailor your analysis as follows:

                    --------------------------------------------------
                    If the image is titled "One-pager for Model", Interpret the key insights from the charts provided and address the following:

                Overall Model Performance:
            
            Evaluate key metrics such as Adjusted R², NRMSE, and DECOMP.RSSD.
            Identify signs of model robustness or potential overfitting/underfitting from the Fitted vs. Residual plot.
            Channel Contributions and Efficiency:
            
            Analyze the Response Decomposition Waterfall by Predictor to highlight the channels driving positive or negative incremental responses.
            Assess the Share of Total Spend, Effect & ROAS chart to determine the efficiency of spending across channels.
            Compare channel-level ROI/ROAS in the In-cluster Bootstrapped ROAS plot and highlight any channels with low returns.
            
            Time Dynamics and Carryover Effects:
            Use the Weibull PDF Adstock plot to interpret the flexible rate over time for each channel and its implications for media planning.
            Examine the Immediate vs. Carryover Response Percentage plot to understand the distribution of immediate vs. long-term effects across channels.
            
            Spend and Response Relationships:
            Interpret the Response Curves and Mean Spends by Channel to identify diminishing returns and thresholds for media spend efficiency.
            Discuss how well spend and response align for each channel.

            Actionable Recommendations:
            Identify opportunities to reallocate budget toward high-ROAS channels and reduce spending on underperforming channels.
            Suggest improvements in spend allocation strategy based on the observed diminishing returns and response curves.
            Highlight any potential adjustments to model assumptions or inputs that could improve future modeling efforts.
            
            Summary and Next Steps:
            Provide a concise summary of the model’s performance, strengths, and limitations.
            Suggest actionable next steps for optimizing the media mix strategy, such as additional data collection, testing alternative predictors, or validating assumptions.
            If any channel’s ROI or ROAS is significantly below expectations, recommend delaying further investment until additional data is available or until actionable improvements are identified.

                    --------------------------------------------------
                    If the image is titled "Budget Allocation Onepager for Model", please address these points:

                    1. Budget Allocation Insights
Analyze how the marketing budget is currently allocated across channels.
Identify any channels that appear under- or over-invested relative to their performance metrics (e.g., incremental response, ROAS).

                    2. Channel Performance and ROI
Compare channel-specific ROAS and performance, including any bootstrapped confidence intervals.
Assess whether the spend allocation aligns with the incremental impact observed across channels.
Important: If the ROI or ROAS metrics are unfavorable, recommend delaying further investment until additional data is available or until actionable strategies are developed.
Ensure all numerical details are thoroughly verified for clarity and accuracy.

                    3. Final Evaluation & Actionable Insights
Provide an assessment of whether this is a 'good' model overall based on industry standards.
Offer specific, actionable recommendations on refining both the model and the media mix strategy.
Ensure detailed attention to the accuracy of all numerical values to avoid misinterpretation.

                    4. Budget Allocation Table
At the end of your analysis, present a structured budget allocation table indicating optimal spend per channel and expected ROI.
If the model suggests that no channel is likely to generate a strong return on investment, explicitly state that further spending is not advisable.
In such cases, recommend alternative actions such as collecting more data, refining the model, or adjusting campaign strategies before making investment decisions.
Format the budget recommendation as follows:
Channel	Recommended Spend ($)	Expected ROI (%)
Channel 1	$XX,XXX	XX%
Channel 2	$XX,XXX	XX%
...	...	...
"""
                    enhanced_image_bytes = pil_image_to_bytes(enhanced_image)
                    analysis_result = analyze_image_with_gpt4_vision_custom(enhanced_image_bytes, default_prompt, api_key)
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)
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
