import os
import base64
from openai import OpenAI
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import numpy as np
import pytesseract
import tempfile
import matplotlib.pyplot as plt
import json
from matplotlib.figure import Figure

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
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if the image has an alpha channel to avoid issues with autocontrast
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Apply enhancements in a specific order for best results
    # 1. First adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    # 2. Then adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    # 3. Apply auto contrast if requested
    if auto_contrast:
        try:
            image = ImageOps.autocontrast(image, cutoff=0.5)
        except Exception as e:
            print(f"Autocontrast failed: {e}, skipping this step")
    
    # 4. Sharpen the image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)
    
    # 5. Apply unsharp mask for additional sharpening if requested
    if use_unsharp:
        image = image.filter(ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold
        ))
    
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

def detect_image_type(image_bytes: bytes, api_key: str) -> str:
    """
    Detects the type of marketing mix model image.
    Returns 'one-pager' or 'budget-allocation' or 'unknown'.
    """
    try:
        # OCR based detection - look for key phrases
        image = Image.open(io.BytesIO(image_bytes))
        
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(image)
        
        # Check for keywords that identify the image type
        if "One-pager for Model" in text or "Model Performance:" in text:
            return "one-pager"
        elif "Budget Allocation" in text or "Budget Allocation Onepager" in text:
            return "budget-allocation"
        
        # If OCR doesn't find it, try the OpenAI Vision API for image detection
        prompt = "What type of marketing mix model image is this? Is it a 'One-pager for Model' showing model performance metrics or a 'Budget Allocation Onepager'? Respond with only 'one-pager' or 'budget-allocation'."
        
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.lower().strip()
        
        if "one-pager" in result:
            return "one-pager"
        elif "budget" in result or "allocation" in result:
            return "budget-allocation"
        
        return "unknown"
        
    except Exception as e:
        print(f"Error in image type detection: {str(e)}")
        return "unknown"

def pil_image_to_bytes(image: Image.Image, format="JPEG"):
    """
    Converts a Pillow Image to bytes.
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format=format)
    return img_byte_array.getvalue()

def analyze_image_with_gpt4_vision_custom(image_bytes: bytes, custom_prompt: str, api_key: str) -> str:
    """
    Sends an image with a custom prompt to GPT-4 Vision for analysis.
    """
    try:
        client = OpenAI(api_key=api_key)
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
        raise Exception(f"Error analyzing image with custom prompt: {str(e)}")

def summarize_responses(summary_prompt: str, api_key: str) -> str:
    """
    Uses GPT-4 to summarize combined responses.
    """
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")

def add_table_to_doc(doc: Document, table_lines: list):
    """
    Parses a list of strings (each representing a table row in markdown format)
    and adds a formatted table to the DOCX document.
    """
    if not table_lines:
        return
    
    # Parse the header row to determine column count
    header = table_lines[0]
    # Remove leading/trailing | and split by |
    columns = [col.strip() for col in header.strip('|').split('|')]
    col_count = len(columns)
    
    # Create table with appropriate number of rows and columns
    table = doc.add_table(rows=len(table_lines)-1, cols=col_count)
    table.style = 'Table Grid'
    
    # Process each row (skip the separator row)
    data_rows = [row for i, row in enumerate(table_lines) if i != 1]
    
    for i, row_text in enumerate(data_rows):
        # Remove leading/trailing | and split by |
        cells = [cell.strip() for cell in row_text.strip('|').split('|')]
        
        # Add data to table cells
        for j, cell_text in enumerate(cells):
            if j < col_count:  # Ensure we don't exceed column count
                cell = table.cell(i, j)
                cell.text = cell_text
                
                # Style header row
                if i == 0:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
                            run.font.size = Pt(11)
    
    # Add some space after the table
    doc.add_paragraph()

def add_formatted_text_to_doc(doc: Document, text: str):
    """
    Processes the final_output string line by line.
    Lines that start with '###' are added as headings.
    Lines that look like table rows (starting and ending with '|') are collected and added as a table.
    All other lines are added as normal paragraphs.
    """
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Track table lines
    table_lines = []
    collecting_table = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            # If we were collecting a table, add it now
            if collecting_table and table_lines:
                add_table_to_doc(doc, table_lines)
                table_lines = []
                collecting_table = False
            continue
        
        # Check if it's a table row
        if line.startswith('|') and line.endswith('|'):
            collecting_table = True
            table_lines.append(line)
            continue
        else:
            # If we were collecting a table, add it now
            if collecting_table and table_lines:
                add_table_to_doc(doc, table_lines)
                table_lines = []
                collecting_table = False
        
        # Check if it's a heading (starts with '#')
        if line.startswith('### '):
            p = doc.add_heading(line[4:], level=3)
            p.style.font.size = Pt(14)
            p.style.font.bold = True
            continue
        elif line.startswith('## '):
            p = doc.add_heading(line[3:], level=2)
            p.style.font.size = Pt(16)
            p.style.font.bold = True
            continue
        elif line.startswith('# '):
            p = doc.add_heading(line[2:], level=1)
            p.style.font.size = Pt(18)
            p.style.font.bold = True
            continue
        
        # Add as normal paragraph
        p = doc.add_paragraph(line)
    
    # If there's a table at the end, add it
    if table_lines:
        add_table_to_doc(doc, table_lines)

def chatbot_with_image(image_bytes: bytes, user_question: str, api_key: str) -> str:
    """
    Allows users to ask specific questions about the marketing mix model image
    """
    try:
        client = OpenAI(api_key=api_key)
        
        if not api_key:
            raise ValueError("OpenAI API key is required.")
            
        prompt = f"""The user has uploaded a marketing mix model analysis image and asks the following question:

"{user_question}"

Please analyze the marketing mix model image and provide a detailed answer to the user's specific question. 
Focus on providing actionable insights, specific metrics, and clear explanations that directly address what the user asked about.
If the question is about ROI, ROAS, channel performance, budget allocation, or model metrics, extract specific numbers and provide context.
If the image doesn't contain information needed to answer the question completely, state what's missing and provide the best possible answer based on what's visible.
"""
        
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error processing chat question: {str(e)}")

def analyze_one_pager_with_gpt4_vision(image_bytes: bytes, api_key: str) -> str:
    """
    Analyzes a One-pager for Model image with GPT-4 Vision.
    This is optimized for analyzing one half of the one-pager image.
    """
    try:
        # First, split the image vertically since one-pagers are dense with information
        image = Image.open(io.BytesIO(image_bytes))
        halves = split_image_vertical(image)
        
        # Create prompts for each half
        left_prompt = """You are an expert marketing data scientist analyzing a marketing mix model one-pager. 
        This is the LEFT HALF of a 'One-pager for Model' document.
        Analyze the charts and metrics shown in this half. Pay special attention to:
        
        1. Overall model metrics like Adjusted R², NRMSE, DECOMP.RSSD
        2. The Fitted vs. Residual plot
        3. Any response decomposition charts
        4. ROI/ROAS metrics for different channels
        
        Extract specific numbers, percentages, and metrics. Be precise in your analysis.
        Organize your analysis by metric/chart but don't list every single data point - focus on the important insights.
        """
        
        right_prompt = """You are an expert marketing data scientist analyzing a marketing mix model one-pager. 
        This is the RIGHT HALF of a 'One-pager for Model' document.
        Analyze the charts and metrics shown in this half. Pay special attention to:
        
        1. Adstock and carryover metrics (like Weibull PDF plots)
        2. Response curve analysis
        3. Any channel-specific metrics like ROAS/ROI not covered in the left half
        4. Any other charts or tables visible
        
        Extract specific numbers, percentages, and metrics. Be precise in your analysis.
        Organize your analysis by metric/chart but don't list every single data point - focus on the important insights.
        """
        
        # Process each half
        left_bytes = pil_image_to_bytes(halves[0])
        right_bytes = pil_image_to_bytes(halves[1])
        
        left_analysis = analyze_image_with_gpt4_vision_custom(left_bytes, left_prompt, api_key)
        right_analysis = analyze_image_with_gpt4_vision_custom(right_bytes, right_prompt, api_key)
        
        # Now create a comprehensive analysis combining both halves
        summary_prompt = f"""You are creating a comprehensive analysis of a marketing mix model one-pager.
        Below are two analyses: one of the left half and one of the right half of the same document.
        
        LEFT HALF ANALYSIS:
        {left_analysis}
        
        RIGHT HALF ANALYSIS:
        {right_analysis}
        
        Combine these analyses into a single, well-structured report that covers:
        
        ### Model Performance Overview
        Summarize the model's statistical quality (R², NRMSE, etc.)
        
        ### Key Business Drivers
        Identify which channels are driving the most positive responses
        
        ### Investment Timing Impact
        Analyze the immediate vs. carryover effects
        
        ### ROI Analysis
        Compare ROI/ROAS across channels
        
        ### Optimization Recommendations
        Provide actionable suggestions based on the data
        
        Be specific with numbers, percentages, and insights. Maintain a structured format with clear headings.
        """
        
        final_analysis = summarize_responses(summary_prompt, api_key)
        return final_analysis
        
    except Exception as e:
        raise Exception(f"Error analyzing one-pager: {str(e)}")

def extract_predictor_percentages(image_bytes: bytes, api_key: str) -> dict:
    """
    Extracts predictor percentages from a Response Decomposition Waterfall chart.
    Returns a dictionary of predictor names and their percentage contributions.
    """
    try:
        # First, try to find the right section of the image that contains the waterfall chart
        # Split the image vertically since the chart is typically on the left half
        image = Image.open(io.BytesIO(image_bytes))
        halves = split_image_vertical(image)
        left_half_bytes = pil_image_to_bytes(halves[0])
        
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(left_half_bytes).decode("utf-8")
        
        prompt = """You are examining a marketing mix model one-pager, focusing on the Response Decomposition Waterfall by Predictor chart.
        
        This chart shows the contribution of each marketing channel to the total response.
        
        Please look at the Response Decomposition Waterfall by Predictor chart (if present) and extract:
        1. The exact name of each marketing channel/predictor (e.g., TV, Digital, Print, Radio)
        2. The exact percentage contribution of each channel (precise numbers as shown in the chart)
        
        Format your response as a JSON object where keys are predictor names (exactly as labeled in the chart) 
        and values are their percentage contributions as decimal numbers (not strings with % symbols).
        
        Example format:
        {
          "TV": 28.4,
          "Digital": 15.2,
          "Print": 5.6
        }
        
        IMPORTANT: Only include actual predictors from the image with their exact percentages as shown.
        Do NOT round numbers or include values not explicitly shown. Do NOT make up data or include
        channels not present in the chart. If you can't find the chart or the percentages, return an empty object.
        """
        
        # Create a completion with the json_object response format
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # If no data was found, try again with the full image
        if not result:
            base64_full_image = base64.b64encode(image_bytes).decode("utf-8")
            second_response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_full_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            result = json.loads(second_response.choices[0].message.content)
            
        return result
        
    except Exception as e:
        print(f"Error extracting predictor percentages: {str(e)}")
        return {}

def create_predictor_pie_chart(predictor_data: dict) -> Figure:
    """
    Creates a pie chart visualization of predictor percentages.
    Returns a matplotlib Figure object.
    """
    if not predictor_data:
        # Return an empty figure if no data
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No predictor data available", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
        
    # Sort data by value (descending)
    sorted_data = dict(sorted(predictor_data.items(), key=lambda item: item[1], reverse=True))
    
    # Prepare data for the pie chart
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    
    # Create the pie chart
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Custom colors with good contrast
    colors = plt.cm.tab20.colors
    
    # Define a custom autopct function to show the exact percentages from the data
    def make_autopct(values):
        def my_autopct(pct):
            # Find the closest value in our original data
            for i, v in enumerate(values):
                if abs(v/sum(values)*100 - pct) < 0.1:  # If within 0.1% tolerance
                    return f'{v:.1f}%'  # Show the exact value from our data
            return f'{pct:.1f}%'  # Fallback to calculated percentage
        return my_autopct
    
    # Create the pie chart with percentages shown
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=None,
        autopct=make_autopct(values),
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add a legend outside the pie with exact percentages
    legend_labels = [f"{label} ({value:.1f}%)" for label, value in sorted_data.items()]
    ax.legend(
        wedges,
        legend_labels,
        title="Media Channels",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    # Set the font size for percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    ax.set_title('Response Contribution by Media Channel', fontsize=16)
    
    # Adjust layout to make room for legend
    fig.tight_layout()
    
    return fig

def analyze_budget_allocation_with_gpt4_vision(image_bytes: bytes, api_key: str) -> str:
    """
    Analyzes a Budget Allocation Onepager for Model image with GPT-4 Vision.
    This is optimized specifically for budget allocation data.
    """
    try:
        # Create a focused prompt specifically for budget allocation data
        prompt = """You are an expert marketing analyst reviewing a 'Budget Allocation Onepager for Model' document.
        
        Analyze this image in detail and extract the following key information:
        
        1. Budget Allocation Tables:
           - Extract the complete 'Budget Allocation per Paid Media Variable per Month' table data
           - Create a comparison table showing channel performance across different channels:
           
           | Paid Media | abs.mean spend ($) | mean spend% | mean response% | mean ROAS | mROAS |
           |------------|-------------------|-------------|----------------|-----------|-------|
           | [Channel 1]| [Value]           | [Value]     | [Value]        | [Value]   | [Value] |
           | [Channel 2]| [Value]           | [Value]     | [Value]        | [Value]   | [Value] |
           ... and so on for all channels
        
        2. Budget Optimization Scenarios:
           Present three completely separate budget scenarios, with a clear heading and individual table for each:
           
           ### i) Maintain Current Spend Scenario
           
           Present a table of the current spending allocation:
           
           | Channel | Current Spend ($) | Current Spend (%) | Response (%) | ROAS | Recommendation |
           |---------|-------------------|-------------------|--------------|------|----------------|
           | Channel 1 | [Value]         | [Value]           | [Value]      | [Value] | [Brief note] |
           | Channel 2 | [Value]         | [Value]           | [Value]      | [Value] | [Brief note] |
           ... and so on for all channels
           
           After the table, include 2-3 sentences summarizing the current allocation performance.
           
           ### ii) 10% Increase in Spend Scenario
           
           Present a table showing the optimal allocation with a 10% budget increase:
           
           | Channel | Current ($) | Proposed ($) | Change (%) | Expected ROAS | Expected Response (%) |
           |---------|-------------|--------------|------------|---------------|------------------------|
           | Channel 1 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
           | Channel 2 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
           ... and so on for all channels
           
           After the table, add 2-3 sentences explaining which channels should receive increased investment and why.
           
           ### iii) 10% Decrease in Spend Scenario
           
           Present a table showing the optimal allocation with a 10% budget decrease:
           
           | Channel | Current ($) | Proposed ($) | Change (%) | Expected ROAS | Expected Response (%) |
           |---------|-------------|--------------|------------|---------------|------------------------|
           | Channel 1 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
           | Channel 2 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
           ... and so on for all channels
           
           After the table, add 2-3 sentences explaining which channels should have reduced investment and why.
        
        3. Optimization Insights:
           - Identify which channels have the highest and lowest ROAS
           - Analyze how budget allocations should change across the three scenarios
           - Determine which channels would benefit from increased/decreased investment
           - Include a focused recommendation table:
           
           | Channel | Optimum Scenario | Recommended Spend ($) | Expected ROI (%) | Recommendation |
           |---------|------------------|----------------------|------------------|----------------|
           | Channel 1 | [Scenario]     | [Value]              | [Value]          | [Increase/Decrease/Maintain] |
           | Channel 2 | [Scenario]     | [Value]              | [Value]          | [Increase/Decrease/Maintain] |
           ... and so on for all channels
        
        Be extremely specific with numbers and percentages. Create a structured analysis with clear, actionable insights.
        Ensure all tables align perfectly and contain accurate data extracted from the image.
        """
        
        # Send the full image for analysis
        analysis = analyze_image_with_gpt4_vision_custom(image_bytes, prompt, api_key)
        
        # Structure the output
        structured_output = f"""## Budget Allocation Analysis
        
        {analysis}
        
        """
        
        return structured_output
        
    except Exception as e:
        raise Exception(f"Error analyzing budget allocation: {str(e)}")

def comprehensive_analysis(one_pager_image: Image.Image, budget_image: Image.Image, api_key: str) -> tuple:
    """
    Performs a comprehensive analysis of both the One-pager for Model and 
    Budget Allocation Onepager images. 
    
    Processes each image appropriately (split vertically for one-pager,
    crop to 7/12 for budget allocation) and then generates a comprehensive
    report covering:
    
    1. Model Performance overview
    2. Key business drivers
    3. Investment Timing Impact
    4. ROI for each media channel
    5. Budget Allocation optimization scenarios
    6. Summary and recommendations
    
    Returns a tuple containing (final_report, predictor_data, pie_chart)
    """
    try:
        # Process and analyze the one-pager
        one_pager_bytes = pil_image_to_bytes(one_pager_image)
        one_pager_analysis = analyze_one_pager_with_gpt4_vision(one_pager_bytes, api_key)
        
        # Extract predictor percentages for visualization
        predictor_data = extract_predictor_percentages(one_pager_bytes, api_key)
        pie_chart = create_predictor_pie_chart(predictor_data)
        
        # Process and analyze the budget allocation image
        budget_bytes = pil_image_to_bytes(budget_image)
        budget_analysis = analyze_budget_allocation_with_gpt4_vision(budget_bytes, api_key)
        
        # Combine the analyses
        summary_prompt = f"""You are preparing a comprehensive marketing mix model report for executives.
        You have two sections of analysis below - one from the model performance one-pager and one from the budget allocation one-pager.
        
        MODEL PERFORMANCE ANALYSIS:
        {one_pager_analysis}
        
        BUDGET ALLOCATION ANALYSIS:
        {budget_analysis}
        
        Create a cohesive, executive-ready report that combines these analyses with the following structure:
        
        # Marketing Mix Model Comprehensive Analysis
        
        ## 1. Model Performance Overview
        Summarize the key model quality metrics and what they tell us about the reliability of the model.
        
        ## 2. Key Business Drivers
        Identify which marketing channels and other factors are driving business results.
        
        ## 3. Investment Timing Impact
        Explain the immediate vs. carryover effects of marketing investments and implications for campaign timing.
        
        ## 4. Channel ROI Analysis
        Compare ROI/ROAS across all channels with specific numbers.
        
        ## 5. Budget Optimization Scenarios
        Present each of the three budget scenarios in separate sections, with clear headings and individual tables:
        
        ### 5.1. Maintain Current Spend Scenario
        Present the current spending allocation and its performance.
        
        ### 5.2. 10% Increase in Spend Scenario
        Show how a 10% increase would be optimally allocated and what the expected performance improvements would be.
        
        ### 5.3. 10% Decrease in Spend Scenario
        Show how to handle a 10% budget reduction with minimal performance impact.
        
        After presenting all scenarios, include a final recommendation table showing the optimal allocation strategy.
        
        ## 6. Summary and Strategic Recommendations
        Provide 3-5 clear, actionable recommendations based on both analyses.
        
        Be executive-ready: concise but comprehensive, with specific numbers and actionable insights.
        Format any tables cleanly and ensure they add clarity to your insights.
        """
        
        final_report = summarize_responses(summary_prompt, api_key)
        return (final_report, predictor_data, pie_chart)
            
    except Exception as e:
        raise Exception(f"Error generating comprehensive analysis: {str(e)}")

def analyze_image_with_gpt4_vision(image_bytes: bytes, api_key: str) -> str:
    """
    General analyze function that detects the image type and uses the appropriate specialized analyzer.
    This is maintained for compatibility with the chat mode and any other parts of the code that expect this function.
    """
    # First, try to detect the image type and use specialized analyzers
    try:
        image_type = detect_image_type(image_bytes, api_key)
        
        if image_type == "one-pager":
            return analyze_one_pager_with_gpt4_vision(image_bytes, api_key)
        elif image_type == "budget-allocation":
            return analyze_budget_allocation_with_gpt4_vision(image_bytes, api_key)
    except Exception as e:
        # Log the error but continue with generic analysis
        print(f"Error in image type detection: {str(e)}")
    
    # If we get here, either image type detection failed or it's an unknown type
    # Use generic analysis as a fallback
    detailed_prompt = """You are an experienced marketing data scientist. The user has uploaded an image that appears to be from a marketing mix model.
    Please analyze this image in detail, focusing on any metrics, charts, graphs, or tables visible.
    Extract all relevant information about marketing performance, ROI, channel efficiency, or budget allocation that you can see.
    Provide a comprehensive analysis of what this image shows about the marketing mix model.
    
    If the image is titled "One-pager for Model", interpret the key insights from the charts provided and address:
    - Overall Model Performance (metrics like Adjusted R², NRMSE)
    - Channel Contributions and Efficiency 
    - Time Dynamics and Carryover Effects
    - Spend and Response Relationships
    - Actionable Recommendations
    
    If the image is titled "Budget Allocation Onepager", please extract:
    - Budget allocation data across different scenarios
    - Channel-specific metrics (spend, response, ROAS)
    - Optimization recommendations 
    - Present three completely separate budget scenarios, with a clear heading and individual table for each:
    
    ### i) Maintain Current Spend Scenario
           
    Present a table of the current spending allocation:
    
    | Channel | Current Spend ($) | Current Spend (%) | Response (%) | ROAS | Recommendation |
    |---------|-------------------|-------------------|--------------|------|----------------|
    | Channel 1 | [Value]         | [Value]           | [Value]      | [Value] | [Brief note] |
    
    After the table, include 2-3 sentences summarizing the current allocation performance.
    
    ### ii) 10% Increase in Spend Scenario
           
    Present a table showing the optimal allocation with a 10% budget increase:
    
    | Channel | Current ($) | Proposed ($) | Change (%) | Expected ROAS | Expected Response (%) |
    |---------|-------------|--------------|------------|---------------|------------------------|
    | Channel 1 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
    
    After the table, add 2-3 sentences explaining which channels should receive increased investment and why.
    
    ### iii) 10% Decrease in Spend Scenario
           
    Present a table showing the optimal allocation with a 10% budget decrease:
    
    | Channel | Current ($) | Proposed ($) | Change (%) | Expected ROAS | Expected Response (%) |
    |---------|-------------|--------------|------------|---------------|------------------------|
    | Channel 1 | [Value]   | [Value]      | [Value]    | [Value]       | [Value]               |
    
    After the table, add 2-3 sentences explaining which channels should have reduced investment and why.
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
                        {"type": "text", "text": detailed_prompt},
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

def main():
    # Initialize session state for API key if not exists
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load CSS styles
    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add sidebar with instructions
    with st.sidebar:
        st.title("Welcome to the Marketing Mix Model Analysis Tool")
        st.write("This application helps you analyze marketing mix model outputs using advanced AI. Here's how to use it:")
        
        st.markdown("**1. Select Mode:** Choose your preferred analysis mode:")
        st.markdown("   • **Comprehensive Analysis:** For detailed analysis of both model images")
        st.markdown("   • **Chat with Model:** For interactive Q&A about your model")
        
        st.markdown("**2. Upload Image(s):** Upload marketing mix model output image(s)")
        st.markdown("**3. Enter API Key:** Provide your OpenAI API key (securely masked)")
        st.markdown("**4. Get Results:** Receive detailed insights and recommendations")
        
        st.markdown("For optimal results, use high-resolution images showing clear charts and metrics.")
        
        st.subheader("Comprehensive Analysis Details")
        st.markdown("This mode requires two specific images:")
        st.markdown("• **One-pager for Model:** Contains performance metrics, ROI, and response charts")
        st.markdown("• **Budget Allocation Onepager:** Contains budget allocation scenarios")
        
        st.markdown("The output will include all six key sections: Model Performance, Business Drivers, Timing Impact, ROI Analysis, Budget Scenarios, and Recommendations.")
    
    st.title("📊 Marketing Mix Model Analysis")

    # Mode selection
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Comprehensive Analysis", "Chat with Model"],
        help="Choose between comprehensive analysis with two images or interactive Q&A"
    )

    if analysis_mode == "Comprehensive Analysis":
        # For comprehensive analysis, we need both types of images
        st.markdown("""
        <div class="info-box">
            <h3>Comprehensive Analysis Mode</h3>
            <p>Please upload both required model output images:</p>
            <ol>
                <li><b>"One-pager for Model"</b> - Contains performance metrics, ROI, and response charts</li>
                <li><b>"Budget Allocation Onepager for Model"</b> - Contains budget allocation scenarios</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload for One-pager
        uploaded_one_pager = st.file_uploader(
            "Upload One-pager for Model (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Select the One-pager for Model image",
            key="one_pager_uploader"
        )
        if uploaded_one_pager is not None:
            one_pager_bytes = uploaded_one_pager.read()
            one_pager_image = enhance_image(one_pager_bytes)
            st.image(one_pager_image, use_container_width=True, caption="One-pager for Model")
            
        # Upload for Budget Allocation
        uploaded_budget = st.file_uploader(
            "Upload Budget Allocation Onepager (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Select the Budget Allocation Onepager image",
            key="budget_uploader"
        )
        if uploaded_budget is not None:
            budget_bytes = uploaded_budget.read()
            budget_image = enhance_image(budget_bytes)
            st.image(budget_image, use_container_width=True, caption="Budget Allocation Onepager")
        
        # Store images in session state to preserve them between reruns
        if uploaded_one_pager is not None:
            st.session_state.one_pager_image = one_pager_image
            st.session_state.one_pager_bytes = one_pager_bytes
        if uploaded_budget is not None:
            st.session_state.budget_image = budget_image
            st.session_state.budget_bytes = budget_bytes
            
        # Setup for Chat with Model mode
        uploaded_file = None
        enhanced_image = None
        
    else:  # Chat with Model mode
        # Check if images from Comprehensive Analysis are available
        if 'one_pager_image' in st.session_state or 'budget_image' in st.session_state:
            st.info("Images from Comprehensive Analysis mode are available. You can ask questions about these images or upload a new image below.")
            
            # Let user select which image to use for chat
            image_selection = st.radio(
                "Select which image to chat with:",
                ["Upload a new image"] + 
                (["One-pager for Model"] if 'one_pager_image' in st.session_state else []) + 
                (["Budget Allocation Onepager"] if 'budget_image' in st.session_state else []),
                index=0
            )
            
            if image_selection == "One-pager for Model" and 'one_pager_image' in st.session_state:
                enhanced_image = st.session_state.one_pager_image
                st.image(enhanced_image, use_container_width=True, caption="One-pager for Model (from previous upload)")
                uploaded_file = True  # Just to indicate we have an image
            elif image_selection == "Budget Allocation Onepager" and 'budget_image' in st.session_state:
                enhanced_image = st.session_state.budget_image
                st.image(enhanced_image, use_container_width=True, caption="Budget Allocation Onepager (from previous upload)")
                uploaded_file = True  # Just to indicate we have an image
            elif image_selection == "Upload a new image":
                # Regular file uploader for new image
                uploaded_file = st.file_uploader(
                    "Upload Model Output Image (PNG, JPG, JPEG)",
                    type=["png", "jpg", "jpeg"],
                    help="Select an image containing your marketing mix model results"
                )
                if uploaded_file is not None:
                    image_bytes = uploaded_file.read()
                    enhanced_image = enhance_image(image_bytes)
                    st.image(enhanced_image, use_container_width=True, caption="Enhanced Image")
        else:
            # No previous images available, show standard file uploader
            uploaded_file = st.file_uploader(
                "Upload Model Output Image (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                help="Select an image containing your marketing mix model results"
            )
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                enhanced_image = enhance_image(image_bytes)
                st.image(enhanced_image, use_container_width=True, caption="Enhanced Image")

    # API Key input with masked input (always hidden with password type)
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your API key will not be stored and needs to be entered each time you use the application",
        key="api_key_input"
    )

    # Show warning if an image is uploaded but no API key is provided
    if not api_key and (
        (analysis_mode != "Comprehensive Analysis" and uploaded_file is not None) or 
        (analysis_mode == "Comprehensive Analysis" and (
            'uploaded_one_pager' in locals() and uploaded_one_pager is not None or 
            'uploaded_budget' in locals() and uploaded_budget is not None
        ))
    ):
        st.warning("Please enter your OpenAI API key to analyze the image(s).")
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

    # Analysis options based on selected mode
    if analysis_mode == "Comprehensive Analysis":
        # Comprehensive Analysis mode requires both images to be uploaded
        st.markdown("""
        <div class="info-box">
            <h4>Comprehensive Analysis will provide:</h4>
            <ol>
                <li>Model Performance overview (Actual vs Predicted Response)</li>
                <li>Key business drivers / consumption contributors</li>
                <li>Investment Timing Impact (Immediate vs Carryover)</li>
                <li>ROI for each media channel</li>
                <li>Budget Allocation optimization Scenarios (Maintain Current Spend, 10% Increase, 10% Decrease)</li>
                <li>Summary of the recommendations and insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Analyze button for comprehensive analysis
        comprehensive_ready = (
            'uploaded_one_pager' in locals() and uploaded_one_pager is not None and
            'uploaded_budget' in locals() and uploaded_budget is not None and
            api_key
        )
        
        analyze_button = st.button(
            "🔍 Run Comprehensive Analysis",
            help="Click to analyze both images and generate a comprehensive report",
            disabled=not comprehensive_ready
        )
        
        if analyze_button:
            with st.spinner("Analyzing marketing mix model images... This may take a minute or two."):
                try:
                    # Run the comprehensive analysis
                    report, predictor_data, pie_chart = comprehensive_analysis(
                        st.session_state.one_pager_image,
                        st.session_state.budget_image,
                        api_key
                    )
                    
                    # Show the report
                    st.markdown("## 📋 Comprehensive Analysis Report")
                    
                    # Show the text report
                    # Split the report into sections to insert the pie chart after Model Performance section
                    report_sections = report.split("## 2. Key Business Drivers")
                    
                    if len(report_sections) > 1:
                        # Display first section (Model Performance)
                        st.markdown(report_sections[0])
                        
                        # Display the pie chart after Model Performance section
                        if predictor_data:
                            st.markdown("### 📈 Response Contribution by Media Channel")
                            st.pyplot(pie_chart)
                            st.markdown("""
                            <div class="info-box">
                                <p>This visualization shows the percentage contribution of each media channel to the overall response, 
                                based on the Response Decomposition Waterfall by Predictor from the one-pager model.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display the rest of the report
                        st.markdown("## 2. Key Business Drivers" + report_sections[1])
                    else:
                        # Fallback if report doesn't have expected structure
                        st.markdown(report)
                        
                        # Display the pie chart if data is available
                        if predictor_data:
                            st.markdown("### 📈 Response Contribution by Media Channel")
                            st.pyplot(pie_chart)
                            st.markdown("""
                            <div class="info-box">
                                <p>This visualization shows the percentage contribution of each media channel to the overall response, 
                                based on the Response Decomposition Waterfall by Predictor from the one-pager model.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Option to download as DOCX
                    doc = Document()
                    add_formatted_text_to_doc(doc, str(report))
                    
                    # Save temporarily and offer for download
                    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                        doc.save(temp_file.name)
                        
                        with open(temp_file.name, 'rb') as file:
                            docx_bytes = file.read()
                            
                        st.download_button(
                            label="📄 Download Report as DOCX",
                            data=docx_bytes,
                            file_name="Marketing_Mix_Model_Analysis.docx",
                            help="Download the analysis as a Word document"
                        )
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.markdown("""
                    <div class="error-box">
                        <p>There was an error during analysis. Please ensure:</p>
                        <ul>
                            <li>Both images are clear and high resolution</li>
                            <li>Your OpenAI API key is valid and has sufficient credits</li>
                            <li>Your internet connection is stable</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:  # Chat with Model mode
        # Display chat interface
        st.markdown("""
        <div class="info-box">
            <h3>Chat with Model Mode</h3>
            <p>Ask specific questions about your marketing mix model image:</p>
            <ul>
                <li>Which channel has the highest ROI?</li>
                <li>What is the optimal budget allocation?</li>
                <li>How does TV performance compare to Digital?</li>
                <li>What's the R-squared value of this model?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_question = st.chat_input(
            "Ask a question about your marketing mix model image...",
            disabled=not (uploaded_file and api_key)
        )
        
        # Process chat input when submitted
        if user_question:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Determine which image to use (from selection or new upload)
            if analysis_mode == "Chat with Model":
                if 'image_selection' in locals() and image_selection != "Upload a new image":
                    # Use selected image from Comprehensive Analysis
                    if image_selection == "One-pager for Model":
                        image_bytes = st.session_state.one_pager_bytes
                    else:  # Budget Allocation Onepager
                        image_bytes = st.session_state.budget_bytes
                else:
                    # Use newly uploaded image
                    image_bytes = uploaded_file.read()
            
            # Process with AI
            with st.chat_message("assistant"):
                with st.spinner("Processing your question..."):
                    try:
                        # Get the AI's response
                        response = chatbot_with_image(image_bytes, user_question, api_key)
                        
                        # Display the response
                        st.markdown(response)
                        
                        # Add assistant message to chat
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    except Exception as e:
                        error_msg = f"Error processing your question: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        
        # Clear chat button
        if st.button("Clear Chat History", help="Click to clear the current conversation"):
            st.session_state.chat_history = []
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
