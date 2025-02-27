import os
import base64
from openai import OpenAI
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

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
            max_tokens=2000
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
            temperature=0,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error summarizing responses: {str(e)}")

def add_table_to_doc(doc: Document, table_lines: list):
    """
    Parses a list of strings (each representing a table row in markdown format)
    and adds a formatted table to the DOCX document.
    """
    # Parse the rows by splitting on '|'
    rows = []
    for line in table_lines:
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(parts)

    # Check if the second row is a divider (only dashes)
    if len(rows) >= 2:
        is_divider = all(not cell or all(ch in "- " for ch in cell) for cell in rows[1])
        if is_divider:
            header = rows[0]
            data_rows = rows[2:]
        else:
            header = None
            data_rows = rows
    else:
        header = None
        data_rows = rows

    num_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=num_cols)
    table.style = "LightShading-Accent1"  # Change style as desired

    if header:
        hdr_cells = table.add_row().cells
        for i, cell_text in enumerate(header):
            hdr_cells[i].text = cell_text
    for row_data in data_rows:
        row_cells = table.add_row().cells
        for i in range(num_cols):
            cell_text = row_data[i] if i < len(row_data) else ""
            row_cells[i].text = cell_text

def add_formatted_text_to_doc(doc: Document, text: str):
    """
    Processes the final_output string line by line.
    Lines that start with '###' are added as headings.
    Lines that look like table rows (starting and ending with '|') are collected and added as a table.
    All other lines are added as normal paragraphs.
    """
    lines = text.splitlines()
    table_buffer = []
    in_table = False

    for line in lines:
        # If line is blank, flush any table buffer and add a blank paragraph.
        if not line.strip():
            if in_table and table_buffer:
                add_table_to_doc(doc, table_buffer)
                table_buffer = []
                in_table = False
            doc.add_paragraph("")
            continue

        # If the line is a markdown-style heading (starting with '#')
        if line.startswith("###"):
            if in_table and table_buffer:
                add_table_to_doc(doc, table_buffer)
                table_buffer = []
                in_table = False
            # Count the number of '#' to determine heading level.
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            doc.add_heading(heading_text, level=level)
        # If the line appears to be a table row
        elif line.startswith("|") and line.endswith("|"):
            table_buffer.append(line)
            in_table = True
        else:
            if in_table and table_buffer:
                add_table_to_doc(doc, table_buffer)
                table_buffer = []
                in_table = False
            # Add as a normal paragraph. You can customize font size, boldness, etc.
            para = doc.add_paragraph(line)
            run = para.runs[0]
            run.font.size = Pt(11)

    # Flush any remaining table buffer
    if in_table and table_buffer:
        add_table_to_doc(doc, table_buffer)

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
    detailed_prompt = '''
    You are an experienced marketing data scientist. The user has uploaded an image containing output from a Robyn marketing mix model.
    Please read the image carefully and pay extra attention to all numerical details. Verify every number; if there is any uncertainty or ambiguity, clearly note your assumptions.
    Depending on the image title, please tailor your analysis as follows:

    --------------------------------------------------
    If the image is titled "One-pager for Model", interpret the key insights from the charts provided and address the following:

    Overall Model Performance:
    - Evaluate key metrics such as Adjusted R², NRMSE, and DECOMP.RSSD.
    - Identify signs of model robustness or potential overfitting/underfitting from the Fitted vs. Residual plot.

    Channel Contributions and Efficiency:
    - Analyze the Response Decomposition Waterfall by Predictor to highlight the channels driving positive or negative incremental responses.
    - Assess the Share of Total Spend, Effect & ROAS chart to determine spending efficiency.
    - Compare channel-level ROI/ROAS in the In-cluster Bootstrapped ROAS plot and highlight any channels with low returns.

    Time Dynamics and Carryover Effects:
    - Use the Weibull PDF Adstock plot to interpret the flexible rate over time for each channel.
    - Examine the Immediate vs. Carryover Response Percentage plot to understand the distribution of immediate vs. long-term effects.

    Spend and Response Relationships:
    - Interpret the Response Curves and Mean Spends by Channel to identify diminishing returns and spending thresholds.
    - Discuss how well spend and response align for each channel.

    Actionable Recommendations:
    - Identify opportunities to reallocate budget toward high-ROAS channels and reduce spending on underperforming ones.
    - Suggest improvements in spend allocation based on observed diminishing returns.
    - Highlight any adjustments to model assumptions that could improve future efforts.

    Summary and Next Steps:
    - Provide a concise summary of the model’s performance, strengths, and limitations.
    - Suggest actionable next steps for optimizing the media mix strategy.
    - If any channel’s ROI or ROAS is significantly below expectations, recommend delaying further investment until additional data is available or improvements are identified.

    --------------------------------------------------
    If the image is titled "Budget Allocation Onepager for Model", please address these points:


    Budget Allocation per Paid Media:
    Extract the table data for 'Budget Allocation per Paid Media Variable per Month'. The table contains three sections:
    1. **Initial** (Left Table)
    2. **Bounded** (Middle Table)
    3. **Bounded x3** (Right Table)

    The first column represents the **Paid Media** names. Extract the **abs.mean spend ($)**, **mean spend%**, **mean response%**, **mean ROAS**, and **mROAS** values and format the table as follows:

    Please extract the numerical values from the image and format them into a structured table, ensuring that the paid media names match those shown in the image. The table should be formatted as follows:  

    | Paid Media      | Scenario    | abs.mean spend | mean spend% | mean response% | mean ROAS | mROAS |
    |-----------------|-------------|----------------|-------------|----------------|-----------|-------|
    | [Media Name 1]  | Initial     | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |
    | [Media Name 2]  | Initial     | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |
    | [Media Name 1]  | Bounded     | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |
    | [Media Name 2]  | Bounded     | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |
    | [Media Name 1]  | Bounded x3  | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |
    | [Media Name 2]  | Bounded x3  | [Value]        | [Value]     | [Value]        | [Value]   | [Value]  |

    ### Instructions:
    - Extract each row **horizontally from left to right**.
    - Maintain the **Paid Media names exactly** as they appear in the image.
    - Ensure the extracted values are structured in the table above.

    If any values are unclear, estimate based on the image and note any assumptions made.

    1. Critical Budget Allocation Insights & Business Priorities
       - Evaluate the current distribution of the marketing budget across channels based on the Roybn Model.
       - Identify channels with high or low incremental response and ROAS to determine if they are under- or over-invested.
       - For budget adjustments (e.g., scenarios of 5% and 10% reallocation), clearly outline which channels meet the performance criteria for additional investment. Provide specific dollar figures (e.g., "$10,000 for Channel A") to illustrate recommended investment amounts and maximize ROI.

    2. Detailed Channel Performance & ROI Analysis
       - Provide a side-by-side comparison of channel-specific ROAS, incremental response, and bootstrapped confidence intervals.
       - Assess whether the current spend is proportionate to the observed incremental impact.
       - If certain channels show unfavorable ROI/ROAS, recommend holding off on further investment until more data is available or targeted strategies are implemented. Include example figures to quantify the proposed hold or delay.

    3. Additional Budget Investment Strategy
       - Determine which channel(s) offer the best potential for improved ROI with additional budget allocation.
       - Use data-driven insights to rank channels and highlight the most critical opportunities for growth.
       - Explicitly provide recommended dollar amounts (e.g., "Allocate an extra $10,000 to Channel A and $8,500 to Channel B") based on the model’s sensitivity analysis and performance metrics.

    4. Final Model Evaluation & Strategic Recommendations
       - Evaluate whether the Roybn Model meets industry standards and business needs, summarizing strengths and weaknesses.
       - Provide actionable recommendations to refine both the model and the media mix strategy.
       - Ensure all numerical details and assumptions are clearly validated.

    5. Structured Budget Allocation Table
       - Present a table that outlines the recommended spend per channel and the expected ROI.
       - If no channel meets the threshold for a strong ROI—even with extra budget—state that further investment is not advisable and recommend alternative actions (e.g., collecting more data or adjusting campaign strategies).

    Table Format:
    | Channel       | Recommended Spend ($) | Expected ROI (%) |
    |---------------|-----------------------|------------------|
    | Channel 1     | $5K                   | 15%              |
    | Channel 2     | $3.5K                 | 12%              |
    | Channel 3     | $0.8K                 | 18%              |

    Note: The "Budget Allocation per Media" table in the image is arranged horizontally (channels are rows and metrics are columns). Please ignore the Paid Media channel.
    '''

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
        # New checkbox for cropping to 7/12 (top portion)
        crop_to_7_12 = st.checkbox("Crop to 7/12", value=False,
                                   help="If selected, only the top 7/12 portion of the image will be analyzed.")

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
                if crop_to_7_12:
                    # Crop the image to the top 7/12 fraction (by height)
                    width, height = enhanced_image.size
                    new_height = int(height * (7/12))
                    cropped_image = enhanced_image.crop((0, 0, width, new_height))
                    st.markdown("### Cropped Image (Top 7/12)")
                    st.image(cropped_image, use_container_width=True)
                    cropped_image_bytes = pil_image_to_bytes(cropped_image)
                    analysis_result = analyze_image_with_gpt4_vision_custom(cropped_image_bytes, detailed_prompt, api_key)
                    final_output = analysis_result
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)
                elif split_vertically:
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
                    # No splitting or cropping: perform additional clarity/resolution enhancement
                    upscale_factor = 2  # Adjust this factor as needed
                    width, height = enhanced_image.size
                    enhanced_image = enhanced_image.resize((width * upscale_factor, height * upscale_factor), resample=Image.LANCZOS)

                    st.markdown("### Uploaded Image (Enhanced Resolution)")
                    st.image(enhanced_image, use_container_width=True, caption="Uploaded Image")
                    enhanced_image_bytes = pil_image_to_bytes(enhanced_image)
                    analysis_result = analyze_image_with_gpt4_vision_custom(enhanced_image_bytes, detailed_prompt, api_key)
                    final_output = analysis_result
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)

                # Button to download analysis output as a text file
                st.download_button(
                    label="Download Analysis Output (TXT)",
                    data=final_output,
                    file_name="analysis_output.txt",
                    mime="text/plain"
                )

                # --- New: Create a formatted DOCX file for download ---
                doc = Document()
                doc.add_heading("Analysis Output", level=1)
                add_formatted_text_to_doc(doc, final_output)
                docx_buffer = io.BytesIO()
                doc.save(docx_buffer)
                docx_data = docx_buffer.getvalue()

                st.download_button(
                    label="Download Analysis Output (DOCX)",
                    data=docx_data,
                    file_name="analysis_output.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
