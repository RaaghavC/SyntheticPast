import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os
import time
import io
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
</style>
""", unsafe_allow_html=True)

# API Keys (inserted directly here)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Stable Diffusion API endpoint
STABLE_DIFFUSION_API_URL = "https://modelslab.com/api/v6/realtime/text2img"

# Pillow settings
ANTIALIAS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
Image.MAX_IMAGE_PIXELS = None

# Prompts for panorama generation
PANORAMIC_PROMPT = """Create an ultra-wide panoramic photograph of San Francisco Bay during the California Gold Rush in 1852, spanning from the hills to the shoreline in a single continuous scene.

The panorama should flow naturally from left to right:
- Far Left: Begin with a elevated view from the hills, showing the entire bay stretching out below
- Left Side: Transition to the developing cityscape with wooden structures and warehouses along the shoreline
- Center: Focus on the main harbor area with the weathered 'Schooner Anthem' as the centerpiece
- Right Side: Show the bustling docks where cargo is being unloaded
- Far Right: End with a view of more ships arriving through the fog that matches seamlessly with the Far Left view

Throughout the panorama include:
- Natural lighting transitions and consistent fog conditions
- Multiple ships at various distances and angles
- Continuous flow of activity with miners, merchants, and laborers
- Period-accurate details like crates, barrels, rocker boxes, and luxury goods
- Historically accurate architectural elements and ship designs

Additional historical accuracy elements:
- Wooden sailing ships with accurate rigging and period-specific designs
- Early wooden structures of San Francisco lining the shoreline
- Authentic cargo including silk bales and cigar crates being offloaded
- Overcast weather with thick fog and choppy waters
- Accurate period clothing for miners, merchants, and laborers
- Rugged and challenging conditions characteristic of the Gold Rush era
- Somber and determined atmosphere reflecting the time period
- Authentic dock construction methods of the 1850s
- Historically accurate tools and equipment used for cargo handling

The scene should maintain consistent perspective and lighting while seamlessly blending these elements into a single cohesive panoramic view that captures the energy and scale of the Gold Rush era. Ensure the far left and far right edges match in terms of lighting, atmosphere, and terrain for a seamless 360-degree viewing experience."""
NEGATIVE_PROMPT = "modern elements, anachronistic details, steel ships, concrete buildings, modern clothing, clean pristine conditions, sunny clear skies, painting, illustration, artwork, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime, discontinuous edges, mismatched sides, seams between edges"

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

def evaluate_prompt_match(image_description: str) -> Dict:
    """Compare image against original prompt requirements."""
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

def iterative_panorama_generation() -> (Optional[Image.Image], Optional[Dict], Optional[float], str):
    """Iteratively generate, analyze, and refine the panorama until a target quality is met or iterations are exhausted."""
    max_iterations = 3
    best_match = None
    best_image = None
    best_analysis = None
    current_prompt = PANORAMIC_PROMPT

    for iteration in range(max_iterations):
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
                    "prompt": current_prompt + " ultra-wide panoramic view with consistent horizon line and seamless edges",
                    "negative_prompt": NEGATIVE_PROMPT,
                    "width": 2048,
                    "height": 1024,
                    "safety_checker": False,
                    "samples": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50
                }
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
                        image = process_to_equirectangular(image)
                        break
            except Exception as e:
                time.sleep(5)
            attempt += 1
            if attempt < max_attempts:
                time.sleep(current_delay)
                current_delay = min(current_delay * 1.5, max_delay)

        if image is None:
            continue

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

        if match_quality >= 0.85:
            break

        if iteration < max_iterations - 1:
            current_prompt = improve_panorama_prompt(current_prompt, evaluation)

    if best_image and best_analysis:
        best_image = remove_right_side(best_image)
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
        headers = {'Content-Type': 'application/json'}
        payload = {
            "key": STABLE_DIFFUSION_API_KEY,
            "prompt": self.panoramic_prompt,
            "negative_prompt": self.negative_prompt,
            "width": 2048,
            "height": 1024,
            "safety_checker": False,
            "samples": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
        try:
            response = requests.post(STABLE_DIFFUSION_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get('status') == 'success' and result.get('output'):
                image_url = result['output'][0]
                image = self.download_image(image_url)
                if image:
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
        template = """You are a helpful assistant that answers questions based on the provided context.
Use the following context to answer the question. If you're unsure or the answer isn't in the context,
say so directly rather than making assumptions.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer: """
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
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(question, k=3)
            print("\nRetrieved Documents:")
            for doc, score in docs_and_scores:
                print(f"Document: {doc.page_content}\nScore: {score}\n")
            response = self.chain({"question": question})
            return response["answer"]
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
    question = TOPIC_QUESTIONS[topic]
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        answer = chatbot.ask(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ---------------------- Streamlit App Layout ---------------------- #
st.markdown("""
    <div class="header-container">
        <h1 style="color: #FFD700;">⛰️ Gold Rush Explorer ⛏️</h1>
        <p style="color: #B38B00;">Journey back to the California Gold Rush era</p>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "image_url" not in st.session_state:
    st.session_state.image_url = None

image_gen = ImageGenerator()
IMGUR_API_KEY = os.getenv('IMGUR_API_KEY')
imgur_client = ImgurClient(IMGUR_API_KEY)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # When the button is clicked, clear any previous panorama and generate a new one.
    if st.button("Generate New Panorama"):
        st.session_state.pop("image_url", None)
        with st.spinner("Generating iterative panorama..."):
            best_image, best_analysis, best_match, final_prompt = iterative_panorama_generation()
            if best_image:
                os.makedirs("static/generated", exist_ok=True)
                save_path = f"static/generated/panorama_{int(time.time())}.jpg"
                best_image.save(save_path)
                try:
                    result = imgur_client.upload_image(
                        save_path,
                        title=f"Gold Rush Panorama match {int(best_match * 100)}",
                        description=f"Generated panorama with final prompt:\n{final_prompt}\n\nAnalysis:\n{json.dumps(best_analysis, indent=2)}"
                    )
                    image_url = result['link']
                    st.session_state.image_url = image_url
                    chatbot.memory.chat_memory.add_message(
                        SystemMessage(
                            content=f"Image Generation Details:\nFinal Prompt: {final_prompt}\nImage Evaluation: {json.dumps(best_analysis, indent=2)}"
                        )
                    )
                except Exception as e:
                    st.error(f"Error uploading to Imgur: {str(e)}")
            else:
                st.error("Failed to generate panorama.")
    
    # If an image URL exists, display the VR viewer with the image URL embedded.
    if st.session_state.get("image_url"):
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
    <a-entity id="nextButton" position="0 1.5 -3" geometry="primitive: plane; width: 0.8; height: 0.3" material="color: #333; opacity: 0.8" text="value: Next Panorama; align: center; color: #FFF; width: 4"></a-entity>
  </a-scene>
  <script>
    var panoramaImages = ['{image_url}'];
    var currentIndex = 0;
    var skyEl = document.querySelector('#sky');
    var nextButtonEl = document.querySelector('#nextButton');
    nextButtonEl.addEventListener('click', function () {{
      skyEl.emit('fade');
      setTimeout(function () {{
        currentIndex = (currentIndex + 1) % panoramaImages.length;
        skyEl.setAttribute('src', panoramaImages[currentIndex]);
        skyEl.emit('fadeback');
      }}, 300);
    }});
  </script>
</body>
</html>"""
        formatted_vr_html = vr_html.format(image_url=st.session_state.image_url)
        vr_viewer_url = f"data:text/html,{requests.utils.quote(formatted_vr_html)}"
        st.components.v1.iframe(vr_viewer_url, height=500, width=800, scrolling=False)
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
