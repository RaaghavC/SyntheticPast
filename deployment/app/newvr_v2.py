import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os
import time
import io
import zipfile
import base64
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from typing import List, Dict, Optional, Union
from pathlib import Path

# Import additional LangChain classes, including SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage  # <-- New import for hidden messages
import openai


def extract_vector_store(zip_path="vectorstore.zip", extract_to="."):
    """Unpack the committed Chroma store on first run — cloud hosts start with no pdf_vectorstore/ dir."""
    if os.path.exists("pdf_vectorstore"):
        return
    if not os.path.exists(zip_path):
        print(f"Vector store zip '{zip_path}' not found; RAG chat will be unavailable.")
        return
    print(f"Extracting vector store from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Vector store extraction complete.")


extract_vector_store()

# ---------------------- Configuration and API Keys ---------------------- #
st.set_page_config(
    page_title="Gold Rush Explorer",
    page_icon="⛰️",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom right, #1A1A1A, #2D2D2D);
        color: #E5E5E5;
    }
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(to bottom right, #2A2000, #483800);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #B38B00;
    }
    .stMarkdown, p, h1, h2, h3 {
        color: #E5E5E5 !important;
    }
    .custom-card {
        background: linear-gradient(to bottom right, #2A2000, #403000);
        border: 1px solid #B38B00;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(179, 139, 0, 0.1);
        transition: all 0.3s ease !important;
    }
    .user-message {
        background: linear-gradient(to right, #B38B00, #806300);
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background: linear-gradient(to right, #2A2000, #403000);
        color: #FFD700;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #B38B00;
    }
    .stButton > button {
        background: linear-gradient(to right, #B38B00, #806300) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    .api-key-warning {
        background-color: #FF4500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Check if API keys are in session state, otherwise initialize with environment variables
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
if 'STABLE_DIFFUSION_API_KEY' not in st.session_state:
    st.session_state.STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY', '')
if 'IMGUR_API_KEY' not in st.session_state:
    st.session_state.IMGUR_API_KEY = os.getenv('IMGUR_API_KEY', '')

# Check if keys need to be input by user
if not st.session_state.OPENAI_API_KEY or not st.session_state.STABLE_DIFFUSION_API_KEY or not st.session_state.IMGUR_API_KEY:
    st.markdown('<div class="api-key-warning">API Keys Required</div>', unsafe_allow_html=True)
    
    with st.expander("Set API Keys", expanded=not st.session_state.OPENAI_API_KEY):
        st.write("Please enter your API keys to use this application:")
        st.session_state.OPENAI_API_KEY = st.text_input("OpenAI API Key", value=st.session_state.OPENAI_API_KEY, type="password")
        st.session_state.STABLE_DIFFUSION_API_KEY = st.text_input("Stable Diffusion API Key", value=st.session_state.STABLE_DIFFUSION_API_KEY, type="password")
        st.session_state.IMGUR_API_KEY = st.text_input("Imgur API Key", value=st.session_state.IMGUR_API_KEY, type="password")
        
        if st.button("Save API Keys"):
            if st.session_state.OPENAI_API_KEY and st.session_state.STABLE_DIFFUSION_API_KEY and st.session_state.IMGUR_API_KEY:
                st.success("API keys saved successfully!")
            else:
                st.error("All API keys are required.")

# Only proceed if all API keys are present
api_keys_set = bool(st.session_state.OPENAI_API_KEY and st.session_state.STABLE_DIFFUSION_API_KEY and st.session_state.IMGUR_API_KEY)

# Set API keys from session state instead of environment variables
if api_keys_set:
    OPENAI_API_KEY = st.session_state.OPENAI_API_KEY
    STABLE_DIFFUSION_API_KEY = st.session_state.STABLE_DIFFUSION_API_KEY
    IMGUR_API_KEY = st.session_state.IMGUR_API_KEY
    
    # Set the environment variable and OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
else:
    # Default values to prevent errors
    OPENAI_API_KEY = ""
    STABLE_DIFFUSION_API_KEY = ""
    IMGUR_API_KEY = ""

# Stable Diffusion API endpoint
STABLE_DIFFUSION_API_URL = "https://modelslab.com/api/v6/realtime/text2img"

# Pillow settings
ANTIALIAS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
Image.MAX_IMAGE_PIXELS = None

# Prompts for panorama generation
PANORAMIC_PROMPT = """A realistic photorealistic color photograph of San Francisco during the California Gold Rush around 1850, an ultra-wide panorama taken from a hillside overlooking the bay. In the CENTER, a dense frontier boom-town of weathered wooden buildings, warehouses and canvas tents along the shore, with a harbor crowded with dozens of wooden sailing ships at anchor, dirt streets, wooden wharves and people. The FAR-LEFT edge and the FAR-RIGHT edge BOTH show the same calm open bay water, distant misty hills and pale sky, so the two side edges match each other and the panorama wraps seamlessly into a continuous 360 degree view. Shot on a camera, documentary photography, sharp focus, fine natural detail, realistic daylight and soft shadows, true-to-life natural colors, ultra realistic, looks like a real photograph."""

NEGATIVE_PROMPT = """painting, oil painting, watercolor, gouache, illustration, drawing, pencil sketch, art, artwork, painterly, visible brush strokes, lithograph, engraving, etching, woodcut, sepia, monochrome, black and white, cartoon, anime, comic, 3d render, cgi, video game, concept art, matte painting, digital art, stylized, fantasy, skyscrapers, modern buildings, glass towers, bridges, cars, power lines, paved roads, blurry, deformed, watermark, text, signature, frame, border, mismatched edges, discontinuous sides, visible seam, hard seam, different left and right edges"""

# ---------------------- Additional Functions for Judging and Iterative Refinement ---------------------- #

def is_json(myjson: str) -> bool:
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

def ensure_seamless_edges(equi_image: Image.Image) -> Image.Image:
    """Ensure the edges of the equirectangular image blend seamlessly."""
    width, height = equi_image.size
    blend_width = width // 20  # 5% blend on each edge

    # Convert to numpy array
    img_array = np.array(equi_image)

    # Extract left and right edges
    left_edge = img_array[:, :blend_width, :]
    right_edge = img_array[:, -blend_width:, :]

    # Create gradient for blending
    gradient = np.linspace(0, 1, blend_width).reshape(1, -1, 1)
    gradient = np.repeat(gradient, height, axis=0)
    gradient = np.repeat(gradient, 3, axis=2)

    # Blend edges
    blend_left = left_edge * gradient + right_edge * (1 - gradient)
    blend_right = right_edge * gradient[:, ::-1, :] + left_edge * (1 - gradient[:, ::-1, :])

    # Apply blended edges
    img_array[:, :blend_width, :] = blend_left
    img_array[:, -blend_width:, :] = blend_right

    return Image.fromarray(img_array.astype(np.uint8))

def process_to_equirectangular(image: Image.Image) -> Image.Image:
    """Process panorama to ensure seamless edges."""
    if image.size != (2048, 1024):
        image = image.resize((2048, 1024), ANTIALIAS)
    seamless_equi = ensure_seamless_edges(image)
    img_array = np.array(seamless_equi)
    left_edge = img_array[:, :10, :]
    right_edge = img_array[:, -10:, :]
    seam_diff = np.mean(np.abs(left_edge - right_edge)) / 255.0
    print(f"Seam difference after processing: {seam_diff:.2%}")
    return seamless_equi

def download_image(url: str) -> Optional[Image.Image]:
    """Download and convert image from URL."""
    for _ in range(3):
        try:
            response = requests.get(url, allow_redirects=True, timeout=30)
            response.raise_for_status()
            image_data = response.content
            try:
                return Image.open(io.BytesIO(image_data))
            except Exception as e:
                print(f"Error opening image data: {str(e)}. Retrying...")
                time.sleep(5)
        except Exception as e:
            print(f"Error downloading image: {str(e)}. Retrying...")
            time.sleep(5)
    print("Failed to download image after 3 attempts.")
    return None

def analyze_image_with_gpt4v(image: Image.Image) -> str:
    """Analyze image using GPT-4V vision capabilities."""
    if not api_keys_set:
        return json.dumps({
            "description": "API keys are required to analyze images.",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this panoramic image objectively and in detail.\n"
                                "Focus on:\n"
                                "1. Geographical layout from hills to shoreline\n"
                                "2. Historical elements and accuracy\n"
                                "3. Atmospheric conditions\n"
                                "4. Activity and people\n"
                                "5. Scene transitions and continuity\n"
                                "6. Edge matching and seamless connection\n"
                                "List exactly what you see without interpretation."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        description = response.choices[0].message.content
        return json.dumps({
            "description": description,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"Error in GPT-4V analysis: {str(e)}")
        return json.dumps({
            "error": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })

def verify_historical_accuracy(image: Image.Image) -> bool:
    """Use GPT-4V to verify the image contains no modern elements before accepting it"""
    if not api_keys_set:
        return True  # Skip verification if API keys aren't set
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a historical accuracy validator for images of San Francisco during the 1851 California Gold Rush. Catch only CLEAR, unambiguous modern anachronisms - do not nitpick. 1851 San Francisco was a densely-built boom-town of mostly one- and two-story wooden buildings packed on steep hills above a ship-filled harbor; density, tightly packed or similar-looking buildings, a hilltop signal tower or lighthouse, and a colorful or hazy sky are all period-appropriate and must NOT be flagged."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "It should depict 1851 Gold Rush San Francisco: a dense town of one-to-two-story wooden buildings on hills above a bay, with a harbor of wooden sailing ships and wharves. Approve it if broadly plausible for 1851 AND it shows a bay/harbor with sailing ships. ONLY reject for a specific MODERN anachronism: skyscrapers or high-rises (more than about four stories), steel-and-glass towers, a large suspension bridge, cars, trucks, buses, paved asphalt roads, power lines, modern motor or container ships, steamships, ferries, motorboats or any engine-powered boat without sails, or aircraft. Every boat must be a wooden SAILING vessel; reject if any motorized/steam vessel is present even alongside sailing ships. Do NOT reject for dense or uniform buildings, a signal tower or lighthouse, color, or a general 'modern feeling'. Respond YES if acceptable, else 'NO: <specific modern anachronism>'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        print(f"Historical accuracy check result: {result}")
        return result.startswith("YES")
    except Exception as e:
        print(f"Error in historical verification: {str(e)}")
        return False

def evaluate_prompt_match(image_description: str) -> Dict:
    """Compare image against original prompt requirements."""
    if not api_keys_set:
        return create_default_evaluation()
    
    try:
        description_data = json.loads(image_description) if isinstance(image_description, str) else image_description
        description_text = description_data.get('description', '')
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Compare the image description against the requirements. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""Compare this image description against the original requirements.

Original Requirements:
{PANORAMIC_PROMPT}

Image Description:
{description_text}

Return a JSON response with:
{{
    "matched_elements": ["element1", "element2", ...],
    "missing_elements": ["element1", "element2", ...],
    "accuracy_analysis": {{
        "historical_accuracy": "description",
        "architectural_accuracy": "description",
        "clothing_accuracy": "description",
        "edge_matching": "description"
    }},
    "continuity_analysis": {{
        "scene_flow": "description",
        "lighting": "description",
        "perspective": "description",
        "edge_transition": "description"
    }},
    "suggestion": "improvement suggestion"
}}"""
                }
            ]
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return create_default_evaluation()

def create_default_evaluation() -> Dict:
    """Create default evaluation response for error cases."""
    return {
        "matched_elements": [],
        "missing_elements": ["Evaluation failed"],
        "accuracy_analysis": {
            "historical_accuracy": "Not evaluated",
            "architectural_accuracy": "Not evaluated",
            "clothing_accuracy": "Not evaluated",
            "edge_matching": "Not evaluated"
        },
        "continuity_analysis": {
            "scene_flow": "Not evaluated",
            "lighting": "Not evaluated",
            "perspective": "Not evaluated",
            "edge_transition": "Not evaluated"
        },
        "suggestion": "Retry image generation and analysis"
    }

def improve_panorama_prompt(current_prompt: str, evaluation: Dict) -> str:
    """Generate improved prompt based on the evaluation results."""
    if not api_keys_set:
        return current_prompt
    
    try:
        improvement_context = {
            "missing_elements": evaluation.get('missing_elements', []),
            "accuracy_issues": evaluation.get('accuracy_analysis', {}),
            "continuity_issues": evaluation.get('continuity_analysis', {}),
            "main_suggestion": evaluation.get('suggestion', '')
        }
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at improving image generation prompts. Return ONLY additional instructions to append to the original prompt."
                },
                {
                    "role": "user",
                    "content": f"""Based on this evaluation, provide additional prompt instructions to improve the image generation.

Current Issues:
{json.dumps(improvement_context, indent=2)}

Return ONLY the additional instructions focusing on:
1. Adding missing required elements
2. Improving historical and architectural accuracy
3. Enhancing scene continuity and lighting
4. Improving edge matching and transitions
5. Addressing specific issues noted in the suggestion
"""
                }
            ]
        )
        additional_instructions = response.choices[0].message.content.strip()
        improved_prompt = f"{current_prompt}\n\nAdditional refinements:\n{additional_instructions}"
        return improved_prompt
    except Exception as e:
        print(f"Error improving prompt: {str(e)}")
        return current_prompt

def remove_right_side(image: Image.Image, removal_fraction: float = 0.049) -> Image.Image:
    """Remove the rightmost portion of the image."""
    width, height = image.size
    new_width = int(width * (1 - removal_fraction))
    return image.crop((0, 0, new_width, height))

def make_seamless(image: Image.Image, blend_fraction: float = 0.22) -> Image.Image:
    """Make the 360 panorama wrap seamlessly: cross-fade the right edge into the left-edge content so
    the left and right columns match. The blended band lands at the panorama's edges = directly behind
    the viewer's initial position in the VR sphere, so the join is hidden."""
    try:
        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        h, w, _ = arr.shape
        b = max(8, int(w * blend_fraction))
        x = np.linspace(0, 1, b)
        alpha = (x * x * (3 - 2 * x)).reshape(1, b, 1)  # smoothstep
        left = arr[:, :b, :].copy()
        right = arr[:, w - b:, :].copy()
        arr[:, w - b:, :] = right * (1 - alpha) + left * alpha
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"make_seamless failed ({e}); returning original")
        return image

def iterative_panorama_generation(status=None) -> (Optional[Image.Image], Optional[Dict], Optional[float], str):
    """Iteratively generate, analyze, and refine the panorama until a target quality is met or iterations are exhausted."""
    if not api_keys_set:
        st.error("API keys are required to generate panoramas.")
        return None, None, None, PANORAMIC_PROMPT

    def _progress(msg, label=None):
        """Live step update shown to the user (no-op if no status panel was passed)."""
        if status is not None:
            try:
                status.write(msg)
                if label:
                    status.update(label=label)
            except Exception:
                pass

    max_iterations = 3
    best_match = None
    best_image = None
    best_analysis = None
    current_prompt = PANORAMIC_PROMPT

    for iteration in range(max_iterations):
        _progress(f"🖼️ **Pass {iteration + 1} of {max_iterations}** — generating a panorama with the image model…",
                  label=f"Pass {iteration + 1} of {max_iterations}: generating image…")
        image = None
        max_attempts = 10
        initial_delay = 5
        max_delay = 30
        current_delay = initial_delay
        attempt = 0

        while attempt < max_attempts:
            try:
                headers = {'Content-Type': 'application/json'}
                payload = {
                    "key": STABLE_DIFFUSION_API_KEY,
                    "prompt": current_prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "width": 1536,
                    "height": 768,
                    "safety_checker": False,
                    "samples": 1,
                    "guidance_scale": 5.0,
                    "num_inference_steps": 41
                }
                
                # Try adding these parameters if supported
                try:
                    payload["model_id"] = "juggernaut-xl"
                except:
                    pass
                    
                try:
                    payload["use_karras_sigmas"] = True
                    payload["tiling"] = True
                except:
                    pass
                    
                response = requests.post(STABLE_DIFFUSION_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

                if result.get('status') == 'processing':
                    fetch_url = result.get('fetch_result')
                    if not fetch_url:
                        raise ValueError("No fetch URL provided")
                    poll_attempts = 0
                    while poll_attempts < 6:
                        time.sleep(current_delay)
                        fetch_payload = {"key": STABLE_DIFFUSION_API_KEY}
                        fetch_response = requests.post(fetch_url, headers=headers, json=fetch_payload)
                        fetch_response.raise_for_status()
                        result = fetch_response.json()
                        if result.get('status') == 'success' and result.get('output'):
                            break
                        elif result.get('status') == 'error':
                            break
                        poll_attempts += 1
                        current_delay = min(current_delay * 1.5, max_delay)
                elif result.get('status') == 'success' and result.get('output'):
                    pass

                if result.get('output'):
                    image_url = result['output'][0]
                    image = download_image(image_url)
                    if image:
                        _progress("🔍 AI judge: checking the image for historical accuracy…",
                                  label="Checking historical accuracy (AI judge)…")
                        is_historically_accurate = verify_historical_accuracy(image)
                        if not is_historically_accurate:
                            print("Image contains modern elements, rejecting and trying again.")
                            _progress("↩️ Judge rejected it (modern elements spotted) — regenerating…")
                            image = None  # Reset image to force retry
                            attempt += 1
                            continue

                        _progress("✓ Passed the historical-accuracy check")
                        break
            except Exception as e:
                print(f"Error in image generation: {str(e)}")
                time.sleep(5)
            attempt += 1
            if attempt < max_attempts:
                time.sleep(current_delay)
                current_delay = min(current_delay * 1.5, max_delay)

        if image is None:
            continue

        _progress("📝 Analyzing the scene and scoring how well it matches the brief…",
                  label="Analyzing & scoring the panorama…")
        description = analyze_image_with_gpt4v(image)
        evaluation = evaluate_prompt_match(description)

        matched_elements = evaluation.get('matched_elements', [])
        missing_elements = evaluation.get('missing_elements', [])
        denominator = len(matched_elements) + len(missing_elements)
        match_quality = (len(matched_elements) / denominator) if denominator > 0 else 0

        if best_match is None or match_quality > best_match:
            best_match = match_quality
            best_image = image
            best_analysis = {
                'description': json.loads(description) if is_json(description) else description,
                'evaluation': evaluation
            }

        _progress(f"📊 Match score: {int(match_quality * 100)}%")
        if match_quality >= 0.85:
            _progress("🎯 Strong match — finishing up.")
            break

        if iteration < max_iterations - 1:
            _progress("✨ Refining the prompt to improve the next pass…",
                      label="Refining the prompt for the next pass…")
            current_prompt = improve_panorama_prompt(current_prompt, evaluation)

    if best_image and best_analysis:
        _progress("🧵 Stitching the seamless 360° wrap…", label="Stitching seamless 360° edges…")
        # Original seamless 360 wrap: blend the edges, THEN crop the right ~5% so the two wrap columns
        # become adjacent (identical) -> seam ~0. (Dropping remove_right_side is what re-introduced the ghost.)
        best_image = remove_right_side(process_to_equirectangular(best_image))
        return best_image, best_analysis, best_match, current_prompt
    else:
        return None, None, None, current_prompt

# ---------------------- End of Additional Functions ---------------------- #

# ---------------------- Existing Classes ---------------------- #

class ImageGenerator:
    def __init__(self):
        self.panoramic_prompt = PANORAMIC_PROMPT
        self.negative_prompt = NEGATIVE_PROMPT

    def generate_panorama(self):
        if not api_keys_set:
            st.error("API keys are required to generate panoramas.")
            return None, None
            
        headers = {'Content-Type': 'application/json'}
        payload = {
            "key": STABLE_DIFFUSION_API_KEY,
            "prompt": self.panoramic_prompt,
            "negative_prompt": self.negative_prompt,
            "width": 1536,
            "height": 768,
            "safety_checker": False,
            "samples": 1,
            "guidance_scale": 5.0,
            "num_inference_steps": 41
        }
        
        # Try adding these parameters if supported
        try:
            payload["model_id"] = "juggernaut-xl"
        except:
            pass
            
        try:
            payload["use_karras_sigmas"] = True
            payload["tiling"] = True
        except:
            pass
            
        try:
            response = requests.post(STABLE_DIFFUSION_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get('status') == 'success' and result.get('output'):
                image_url = result['output'][0]
                image = self.download_image(image_url)
                if image:
                    # Verify historical accuracy
                    is_historically_accurate = verify_historical_accuracy(image)
                    if not is_historically_accurate:
                        print("Generated image contains modern elements, retrying...")
                        return self.generate_panorama()  # Recursive retry
                        
                    return self.process_to_equirectangular(image), self.panoramic_prompt
            return None, None
        except Exception as e:
            print(f"Error generating panorama: {str(e)}")
            return None, None

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return None

    def process_to_equirectangular(self, image):
        if image.size != (2048, 1024):
            image = image.resize((2048, 1024), ANTIALIAS)
        return image

class ImgurClient:
    BASE_URL = "https://api.imgur.com/3"
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.headers = {'Authorization': f'Client-ID {client_id}'}
    def upload_image(self, image_path: Union[str, Path], title: Optional[str] = None, 
                     description: Optional[str] = None) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read())
        payload = {
            'image': image_data,
            'type': 'base64'
        }
        if title:
            payload['title'] = title
        if description:
            payload['description'] = description
        try:
            response = requests.post(f"{self.BASE_URL}/image", headers=self.headers, data=payload)
            response.raise_for_status()
            return response.json()['data']
        except requests.RequestException as e:
            print(f"Error uploading image: {e}")
            raise

class RAGChatbot:
    def __init__(self, persist_directory: str = "pdf_vectorstore", model_name: str = "gpt-4o"):
        if not api_keys_set:
            # Return a dummy chatbot that will prompt for API keys
            return
            
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7, 
            openai_api_key=OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        template = """You are the "Gold Rush Guide" in an interactive app about 1850s San Francisco and the California Gold Rush. You are warm, knowledgeable, and conversational.

Answer the visitor's question helpfully. Use the historical excerpts below when they are relevant, but you are NOT limited to them: draw freely on your own solid knowledge of the California Gold Rush (1848-1855), early San Francisco, mining, the ships, and daily life of the era. If the visitor asks about the panorama on screen, discuss it as a depiction of Gold Rush-era San Francisco harbor - a boom town of wooden buildings and canvas tents on bare hills, with tall wooden sailing ships crowding the bay and busy docks. Never refuse or say the answer "isn't in the context" just because the excerpts don't mention it; only say you're unsure if a question is genuinely unanswerable, and even then offer related context. Keep replies clear, accurate, and engaging.

Historical excerpts (background - may or may not be relevant):
{context}

Conversation so far:
{chat_history}

Visitor's question: {question}

Gold Rush Guide:"""
        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )
    def ask(self, question: str) -> str:
        if not api_keys_set:
            return "Please set your API keys in the expander at the top of the page to use the chat feature."

        try:
            # 1) RAG retrieval (unchanged) — pull relevant historical excerpts from the vector store
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join(d.page_content for d in docs)
            print(f"\nRetrieved {len(docs)} documents for: {question}")

            # 2) The EXACT panorama currently on screen: the generated one if present, else the default
            img_bytes = st.session_state.get("local_image") or _default_pano_bytes()

            # 3) Recent conversation for continuity
            history = ""
            for m in self.memory.chat_memory.messages[-6:]:
                who = "Visitor" if getattr(m, "type", "") == "human" else "Guide"
                history += f"{who}: {m.content}\n"

            system = (
                "You are the Gold Rush Guide, a warm, knowledgeable expert on the 1848-1855 California Gold "
                "Rush and early San Francisco. The image attached to the user's message is the EXACT panorama "
                "the visitor is looking at on screen right now. When they ask about 'the image', 'this panorama', "
                "or what they're seeing, describe THAT specific image truthfully - state only what is actually "
                "visible in it (buildings, ships, hills, water, people, activity, colors, mood). Do not invent "
                "details that aren't there. For general Gold Rush questions, use the historical excerpts when "
                "relevant plus your own knowledge. Never refuse for lack of context; only say you're unsure if a "
                "question is genuinely unanswerable. Keep replies accurate, clear, and engaging."
            )
            user_text = (
                f"Historical excerpts (background, may or may not be relevant):\n{context}\n\n"
                f"Conversation so far:\n{history}\n\n"
                f"Visitor's question: {question}"
            )
            content = [{"type": "text", "text": user_text}]
            if img_bytes:
                try:
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    im.thumbnail((1024, 1024))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    b64 = base64.b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })

            resp = openai.chat.completions.create(
                model="gpt-4o",
                max_tokens=600,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            )
            answer = resp.choices[0].message.content
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            return answer
        except Exception as e:
            return f"Error: {str(e)}"

@st.cache_resource
def get_chatbot():
    return RAGChatbot()

chatbot = get_chatbot()

# ---------------------- End of Existing Classes ---------------------- #

TOPIC_QUESTIONS = {
    "historical_facts": "Tell me more about the historical timeline and key events of the California Gold Rush.",
    "notable_locations": "What were the most important locations during the Gold Rush and why were they significant?",
    "mining_techniques": "Explain the different mining methods used during the Gold Rush and how they evolved."
}

def handle_topic_click(topic):
    if not api_keys_set:
        st.warning("Please set your API keys in the expander at the top of the page to use this feature.")
        return
        
    question = TOPIC_QUESTIONS[topic]
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        answer = chatbot.ask(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ---------------------- Streamlit App Layout ---------------------- #
st.markdown("""
    <div class="header-container">
        <h1 style="color: #FFD700;">⛰️ Gold Rush Explorer ⛏️</h1>
        <p style="color: #B38B00;">Journey back to the California Gold Rush era</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def _default_pano_data_uri():
    """Bundled default panorama shown in the 360 viewer before the user generates one."""
    try:
        with open("default_panorama.jpg", "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


@st.cache_data
def _default_pano_bytes():
    """Raw bytes of the default panorama, so the chatbot can SEE it when nothing has been generated yet."""
    try:
        with open("default_panorama.jpg", "rb") as f:
            return f.read()
    except Exception:
        return b""


if "messages" not in st.session_state:
    st.session_state.messages = []
if "image_url" not in st.session_state:
    st.session_state.image_url = _default_pano_data_uri()
    st.session_state.is_default_image = False
else:
    st.session_state.is_default_image = False
if "local_image" not in st.session_state:
    st.session_state.local_image = None
if "imgur_failed" not in st.session_state:
    st.session_state.imgur_failed = False

image_gen = ImageGenerator()
imgur_client = ImgurClient(IMGUR_API_KEY if api_keys_set else "")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # When the button is clicked, clear any previous panorama and generate a new one.
    if st.button("Generate New Panorama"):
        if not api_keys_set:
            st.warning("Please set your API keys in the expander at the top of the page to use this feature.")
        else:
            # Remove default image flag when generating new panorama
            st.session_state.is_default_image = False
            st.session_state.imgur_failed = False
            st.session_state.local_image = None
            with st.status("🎨 Generating your panorama… (this runs a few refinement passes, ~2–5 min)", expanded=True) as status:
                best_image, best_analysis, best_match, final_prompt = iterative_panorama_generation(status=status)
                if best_image:
                    status.update(label="📤 Saving & publishing the panorama…")
                    os.makedirs("static/generated", exist_ok=True)
                    save_path = f"static/generated/panorama_{int(time.time())}.jpg"
                    best_image.save(save_path)
                    # First save image in-memory for display in case Imgur fails
                    buffered = io.BytesIO()
                    best_image.save(buffered, format="JPEG")
                    img_data = buffered.getvalue()
                    st.session_state.local_image = img_data
                    
                    # Try uploading to Imgur but have a fallback
                    try:
                        result = imgur_client.upload_image(
                            save_path,
                            title=f"Gold Rush Panorama match {int(best_match * 100)}",
                            description=f"Generated panorama with final prompt:\n{final_prompt}\n\nAnalysis:\n{json.dumps(best_analysis, indent=2)}"
                        )
                        image_url = result['link']
                        st.session_state.image_url = image_url
                        st.session_state.imgur_failed = False
                        
                        if hasattr(chatbot, 'memory') and hasattr(chatbot.memory, 'chat_memory'):
                            chatbot.memory.chat_memory.add_message(
                                SystemMessage(
                                    content=f"Image Generation Details:\nFinal Prompt: {final_prompt}\nImage Evaluation: {json.dumps(best_analysis, indent=2)}"
                                )
                            )
                    except Exception as e:
                        st.warning(f"Error uploading to Imgur: {str(e)}")
                        st.info("Don't worry! The image was still generated and can be viewed below.")
                        st.session_state.imgur_failed = True
                    status.update(label="✅ Panorama ready — scroll down to explore it in 360°!", state="complete")
                else:
                    status.update(label="⚠️ Couldn't produce a clean panorama this time — please try again.", state="error")
                    st.error("Failed to generate panorama.")
    
    # Display the image - either from Imgur URL or directly from local storage
    if st.session_state.get("image_url"):
        if st.session_state.get("is_default_image", False):
            # Display the default historical panorama
            st.image(st.session_state.image_url, caption="San Francisco Harbor, 1851 (Default Historical Panorama)", use_container_width=True)
            st.info("This is a default historical panorama. Use the 'Generate New Panorama' button above to create a custom AI-generated panoramic view.")
        elif not st.session_state.get("imgur_failed"):
            # Use Imgur hosted image with VR viewer
            vr_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>WebXR 360° Panorama VR Viewer</title>
  <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
</head>
<body>
  <a-scene background="color: #000">
    <a-assets>
      <img id="generatedPano" crossorigin="anonymous" src="{image_url}" />
    </a-assets>
    <a-sky id="sky" src="#generatedPano" color="#FFF"
           animation__fade="property: material.color; type: color; from: #FFF; to: #000; dur: 300; startEvents: fade"
           animation__fadeback="property: material.color; type: color; from: #000; to: #FFF; dur: 300; startEvents: fadeback">
    </a-sky>
    <a-camera position="0 1.6 0">
      <a-cursor id="cursor" fuse="true" fuse-timeout="1000" material="color: white; shader: flat" geometry="primitive: ring; radiusInner: 0.005; radiusOuter: 0.01"></a-cursor>
    </a-camera>
  </a-scene>
</body>
</html>"""
            formatted_vr_html = vr_html.format(image_url=st.session_state.image_url)
            vr_viewer_url = f"data:text/html,{requests.utils.quote(formatted_vr_html)}"
            st.components.v1.iframe(vr_viewer_url, height=500, width=800, scrolling=False)
    
    elif st.session_state.get("local_image"):
        # Display the image directly in Streamlit when Imgur fails
        st.image(st.session_state.local_image, caption="Generated panorama", use_container_width=True)
        st.info("Note: The VR viewer is unavailable due to an image hosting issue. You can still view the panorama above.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 💭 Chat with the Gold Rush Guide")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(
                    f'<div class="{message["role"]}-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
    prompt = st.chat_input("Ask about the panorama...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-message">{prompt}</div>',
                unsafe_allow_html=True
            )
        try:
            answer = chatbot.ask(prompt)
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="assistant-message">{answer}</div>',
                    unsafe_allow_html=True
                )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #FFD700;">📖 Historical Facts</h3>
            <p>The California Gold Rush began on January 24, 1848, when gold was discovered 
            at Sutter's Mill. This historic event led to the migration of about 300,000 
            people to California.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Learn More About Historical Facts", key="historical_facts"):
        handle_topic_click("historical_facts")

with col2:
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #FFD700;">🗺️ Notable Locations</h3>
            <p>Explore key Gold Rush sites including Sutter's Mill, Coloma, and the American 
            River. Each location tells a unique story of fortune seekers and pioneers.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Learn More About Locations", key="notable_locations"):
        handle_topic_click("notable_locations")

with col3:
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #FFD700;">⛏️ Mining Techniques</h3>
            <p>Miners used various methods including panning, sluicing, and hydraulic mining. 
            Each technique evolved as the easy-to-find surface gold disappeared.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Learn More About Mining", key="mining_techniques"):
        handle_topic_click("mining_techniques")