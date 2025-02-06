import os
import base64
from openai import OpenAI
import streamlit as st

def analyze_image_with_gpt4_vision(image_bytes: bytes, api_key: str) -> str:
    """
    Sends an image to GPT-4 Vision API for marketing mix model analysis
    """
    try:
        # Initialize OpenAI client with user provided API key
        client = OpenAI(api_key=api_key)
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        # Encode the image as Base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Marketing mix model analysis prompt
        analysis_prompt = """You are an experienced marketing data scientist. Below is a Robyn marketing mix model output containing charts and metrics:

Please analyze and interpret the following aspects:

Overall Model Performance and Quality
- Assess the reported R², NRMSE, and any other goodness-of-fit metrics
- Indicate whether the model appears robust, what the residual plot suggests, and any signs of overfitting or underfitting

Key Drivers and Contributions
- Explain which channels or predictors are contributing most to incremental sales/response (e.g., from the decomposition waterfall)
- Summarize how each channel's ROI compares and whether the results make intuitive sense

Immediate vs. Carryover Effects
- Evaluate the distribution of short-term (immediate) vs. long-term (carryover) responses across channels and what that implies for budgeting or planning

Recommendations
- Suggest how to optimize media spend based on ROI and diminishing returns (e.g., referencing the response curves)
- Identify any under- or over-invested channels
- Provide any next steps for improving the model (e.g., adding data sources, adjusting hyperparameters, validating assumptions)

Finally, please conclude with:
- A rating of whether this is a 'good' model overall by typical marketing analytics standards
- Specific, actionable recommendations on how to refine both the model and the media mix strategy"""

        # Make the API request
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Using the model that works
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
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

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Marketing Mix Model Analysis with GPT-4 Vision",
        page_icon="📊",
        layout="wide"
    )

    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Header section
    st.title("📊 Marketing Mix Model Analysis")

    st.markdown("""
    <div class="info-box">
        <p>Upload your marketing mix model output images (charts, metrics, etc.) for detailed analysis. 
        The AI will provide insights on model performance, key drivers, and optimization recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Image upload section
        uploaded_file = st.file_uploader(
            "Upload Model Output Image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Select an image containing your marketing mix model results"
        )

        # Display uploaded image
        if uploaded_file is not None:
            st.image(uploaded_file, use_container_width=True, caption="Marketing Mix Model Output")

    with col2:
        # API Key input section with masked input
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

        # Analysis button
        analyze_button = st.button(
            "📊 Analyze Model",
            help="Click to start the analysis",
            disabled=(uploaded_file is None or not api_key)
        )

    # Analysis section
    if uploaded_file is not None and analyze_button and api_key:
        with st.spinner("Analyzing marketing mix model..."):
            try:
                # Read image bytes
                image_bytes = uploaded_file.read()

                # Get analysis
                analysis_result = analyze_image_with_gpt4_vision(
                    image_bytes=image_bytes,
                    api_key=api_key
                )

                # Display results
                st.markdown("### 📋 Analysis Results")
                st.markdown(
                    f'<div class="analysis-result">{analysis_result}</div>',
                    unsafe_allow_html=True
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

    # Footer
    st.markdown("""
        <div class="footer">
            <p>Powered by GPT-4 Vision API • Upload limit: 10MB per image</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()