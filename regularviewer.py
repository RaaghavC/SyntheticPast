import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os
import time
import io
import base64
import numpy as np
import zipfile
from dotenv import load_dotenv
from PIL import Image
from typing import List, Dict, Optional, Union
from pathlib import Path

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="Gold Rush Explorer",
    page_icon="⛰️",
    layout="wide"
)

# Import additional LangChain classes
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, Document
import openai

# Debugging info (hidden in expander)
with st.expander("Debug Info", expanded=False):
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
    
    if os.path.exists("vectorstore.zip"):
        st.write("Zip file exists!")
        st.write("Zip file size:", os.path.getsize("vectorstore.zip"))
    else:
        st.write("Zip file NOT found!")
    
    if os.path.exists("pdf_vectorstore"):
        st.write("Vector store directory exists!")
        st.write("Files in vector store:", os.listdir("pdf_vectorstore"))
    else:
        st.write("Vector store directory NOT found!")

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

# More robust API key handling
try:
    # Try to get API keys from Streamlit secrets
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if 'STABLE_DIFFUSION_API_KEY' not in st.session_state:
        st.session_state.STABLE_DIFFUSION_API_KEY = st.secrets.get("STABLE_DIFFUSION_API_KEY", "")
    if 'IMGUR_API_KEY' not in st.session_state:
        st.session_state.IMGUR_API_KEY = st.secrets.get("IMGUR_API_KEY", "")
except Exception as e:
    print(f"Error loading secrets: {str(e)}")
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    if 'STABLE_DIFFUSION_API_KEY' not in st.session_state:
        st.session_state.STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY', '')
    if 'IMGUR_API_KEY' not in st.session_state:
        st.session_state.IMGUR_API_KEY = os.getenv('IMGUR_API_KEY', '')

# API keys are set from session state
OPENAI_API_KEY = st.session_state.OPENAI_API_KEY
STABLE_DIFFUSION_API_KEY = st.session_state.STABLE_DIFFUSION_API_KEY
IMGUR_API_KEY = st.session_state.IMGUR_API_KEY

# Check if API keys are missing
if not OPENAI_API_KEY or not STABLE_DIFFUSION_API_KEY or not IMGUR_API_KEY:
    st.warning("Some API keys are missing. Please set them in Streamlit secrets or environment variables.")

# Set the environment variable and OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Flag to indicate API keys are set
api_keys_set = bool(OPENAI_API_KEY and STABLE_DIFFUSION_API_KEY and IMGUR_API_KEY)

# Stable Diffusion API endpoint
STABLE_DIFFUSION_API_URL = "https://modelslab.com/api/v6/realtime/text2img"

# Pillow settings
ANTIALIAS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
Image.MAX_IMAGE_PIXELS = None

# Function to create simple Gold Rush knowledge base
def create_simple_gold_rush_kb():
    """Create a simple in-memory vector store with Gold Rush facts."""
    from langchain.schema import Document
    
    print("Creating simple Gold Rush knowledge base...")
    
    # Gold Rush facts - comprehensive set for better responses
    facts = [
        "The California Gold Rush began on January 24, 1848, when gold was discovered by James W. Marshall at Sutter's Mill in Coloma, California.",
        "The news of gold brought approximately 300,000 people to California from the rest of the United States and abroad.",
        "The sudden influx of gold into the money supply reinvigorated the American economy, and the sudden population increase allowed California to go rapidly to statehood in 1850.",
        "The Gold Rush had severe effects on Native Californians and resulted in a precipitous population decline from disease, genocide and starvation.",
        "By the time it ended, California had gone from a thinly populated ex-Mexican territory to having one of its first two U.S. Senators, John C. Frémont, selected to be the first presidential nominee for the new Republican Party, in 1856.",
        "The discovery occurred at Sutter's Mill, a sawmill owned by 19th-century pioneer John Sutter, where gold was found, setting off the California Gold Rush.",
        "Mining methods during the California Gold Rush evolved over time. At first, miners used simple techniques like gold panning and sluice boxes to wash gold from gravel.",
        "Later, more sophisticated methods were developed, including hydraulic mining, dredging, and hard rock mining. Each of these methods had their own environmental impacts.",
        "The most basic method used by gold prospectors during the Gold Rush was 'placer mining'. In placer mining, loose gold is recovered from sand and gravel by using water and simple tools like gold pans.",
        "Hydraulic mining was a mining technique widely used during the California Gold Rush that used high-pressure jets of water to dislodge rock material and potential gold deposits.",
        "The pressurized water was directed through a nozzle, known as a 'monitor', which blasted water at the hillside, washing away tons of gravel and exposing potential gold deposits.",
        "This method was very effective for extracting gold from large areas quickly, but had devastating environmental consequences by destroying riparian habitats and causing flooding downstream.",
        "Notable locations during the California Gold Rush included Sutter's Mill, where gold was first discovered; Coloma, the nearby settlement; San Francisco, which grew from a small settlement into a major city.",
        "Sacramento became a major supply center; and the American River where mining camps sprang up all along the western foothills of the Sierra Nevada.",
        "San Francisco grew exponentially during the Gold Rush. The settlement of Yerba Buena had 900 residents in 1848, but by the end of 1849, San Francisco had over 20,000 residents.",
        "The city was the major port of entry for sea travelers to the goldfields, and its harbor was filled with abandoned ships whose crews had deserted for the gold mines.",
        "San Francisco became a center of shipping, banking, and finance for the western mining industry.",
        "The impact of the Gold Rush on California's indigenous population was devastating. The Native American population, estimated at 150,000 in 1845, was reduced to less than 30,000 by 1870.",
        "This dramatic decline was due to disease, dislocation, and outright violence against Native Americans. The California legislature passed laws that facilitated removing Native Americans from their lands, enslaving them, and murdering them.",
        "The Chinese came to California in large numbers during the Gold Rush, with 40,000 arriving from 1851-1860. They faced substantial discrimination but managed to thrive due to their organization and work ethic.",
        "Chinese immigrants worked claims that others had abandoned as unprofitable and managed to find gold by careful work. They also established businesses that served other miners, including laundries, restaurants, and stores.",
        "The journey to California during the Gold Rush could be made by three main routes: sailing around Cape Horn at the southern tip of South America, a journey of 18,000 nautical miles that could take six to eight months.",
        "Alternatively, sailing to Panama, crossing the Isthmus of Panama, and then sailing to California, which reduced the ocean voyage to 6,000 miles but introduced the risk of tropical disease.",
        "The third option was the overland route across the continental United States, which typically took four to six months and faced hazards of weather, terrain, and potential conflict with Native Americans.",
        "The California Gold Rush accelerated California's admission to the United States as the 31st state. California was admitted to the Union as a free state on September 9, 1850.",
        "This was part of the Compromise of 1850, just two years after the Treaty of Guadalupe Hidalgo transferred control of California from Mexico to the United States.",
        "A sluice box was a long, inclined wooden trough with cleats or 'riffles' on the bottom. Miners would shovel sediment into the upper end while water flowed through, washing the lighter materials away.",
        "The heavier gold would get trapped behind the riffles. Sluice boxes allowed miners to process more material than panning alone.",
        "Multiple sluice boxes were sometimes connected to create larger operations, increasing gold recovery rates.",
        "A rocker box (also called a cradle) was a mining device that improved upon simple panning. It consisted of a box on rockers with a sieve at the top, canvas underneath, and cleats at the bottom.",
        "A miner would shovel sediment into the sieve, pour water over it, and rock the cradle side to side. The motion would help separate gold from sediment, with the gold being caught in the cleats or on the canvas.",
        "A rocker allowed one or two miners to process more material than panning.",
        "The Long Tom was an extended sluice box, typically 10-15 feet long, with a screen at the head and riffles on the bottom. It required a steady flow of water and at least four workers to operate efficiently.",
        "Miners would shovel material onto the screen, where larger rocks would be separated out. The water would wash the finer material down the trough, with gold being caught by the riffles.",
        "The Long Tom was more efficient than rockers or simple sluices, allowing miners to process significantly more material.",
        "Drift mining involved digging horizontal tunnels (drifts) into hillsides to reach buried gold deposits, particularly ancient river channels that had been covered by lava flows or other material.",
        "These ancient riverbeds often contained concentrated gold deposits. Drift mining required more technical knowledge and investment than surface mining but could yield significant returns.",
        "Miners would follow the gold-bearing gravels, using timber supports to prevent cave-ins. This method was common in Nevada County and other areas where ancient river channels were accessible from hillsides.",
        "Hardrock mining targeted gold embedded in quartz veins deep underground. This method required substantial investment in equipment, infrastructure, and skilled labor.",
        "Miners dug deep shafts and tunnels, using explosives to break the ore, which was then hauled to the surface for processing.",
        "The gold-bearing quartz was crushed in stamp mills—large machines with heavy weights that pulverized the rock—and then processed with mercury to extract the gold.",
        "Hardrock mining dominated the later Gold Rush years and continued long after surface mining declined.",
        "Mercury (also called quicksilver) was widely used in gold mining because of its ability to amalgamate with gold.",
        "Miners would use mercury in sluice boxes, rocker boxes, and especially in processing ore from hardrock mining. The mercury would bond with gold particles, creating an amalgam that could be easily collected.",
        "This amalgam was then heated, vaporizing the mercury and leaving behind relatively pure gold. This process was efficient but extremely harmful to the environment and miners' health.",
        "Mercury contamination remains a lasting environmental legacy of the Gold Rush era.",
        "A stamp mill was a machine used to crush gold-bearing rock into smaller pieces for processing. It consisted of heavy metal stamps (weights) attached to rods that were lifted and dropped onto the ore by a camshaft.",
        "These were typically powered by a waterwheel or steam engine. The crushed material would then be processed with mercury to extract the gold.",
        "Stamp mills were a key technology for hardrock mining operations, allowing miners to process large quantities of ore.",
        "The distinctive pounding sound of stamp mills could be heard throughout mining districts during their operation."
    ]
    
    # Convert to documents
    documents = [Document(page_content=fact) for fact in facts]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Create in-memory vector store
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    
    print("Simple knowledge base created successfully")
    return vector_store

# Function to extract vector store from zip if needed
def extract_vector_store(zip_path="vectorstore.zip", extract_to="."):
    """Extract vector store from zip file if the directory doesn't exist."""
    try:
        if not os.path.exists("pdf_vectorstore"):
            print("Vector store directory not found. Checking for zip file...")
            
            if os.path.exists(zip_path):
                print(f"Found {zip_path}. Extracting...")
                with st.spinner(f"Extracting vector store from {zip_path}..."):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                print("Extraction complete!")
                return True
            else:
                print(f"Zip file {zip_path} not found.")
                return False
        else:
            print("Vector store directory already exists.")
            return True
    except Exception as e:
        print(f"Error extracting vector store: {str(e)}")
        return False

# Extract vector store at startup
vector_store_available = extract_vector_store()

# Function to verify vector store directory exists
def verify_vectorstore(directory_path="pdf_vectorstore"):
    """Verify that the vector store directory exists and contains necessary files."""
    try:
        if not os.path.exists(directory_path):
            print(f"Vector store directory not found: {directory_path}")
            return False
            
        # List contents of directory
        contents = os.listdir(directory_path)
        print(f"Vector store directory contents: {contents}")
        
        # Check for chroma.sqlite3 file
        if "chroma.sqlite3" not in contents:
            print("Missing chroma.sqlite3 in vector store directory")
            return False
            
        print(f"Vector store verified successfully at {directory_path}")
        return True
    except Exception as e:
        print(f"Error verifying vector store: {str(e)}")
        return False

# Check vector store after extraction
vectorstore_verified = verify_vectorstore()
if not vectorstore_verified:
    st.warning("Vector store not found or invalid. Using fallback knowledge base.")

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

# Enhanced negative prompt with stronger restrictions against modern elements
NEGATIVE_PROMPT = "modern elements, anachronistic details, steel ships, concrete buildings, modern clothing, clean pristine conditions, sunny clear skies, painting, illustration, artwork, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime, discontinuous edges, mismatched sides, seams between edges, skyscrapers, modern buildings, contemporary architecture, concrete structures, steel structures, glass windows, modern city, urban development, tall buildings, high-rise buildings, modern skyline, metal structures, modern infrastructure, highways, paved roads, vehicles, cars, trucks, modern boats, modern ships, electricity, power lines, telephone poles, street lamps, traffic lights, antennas, satellite dishes, modern technology, contemporary clothing, modern fashion, anachronistic elements, Golden Gate Bridge, Bay Bridge, suspension bridges, modern piers, asphalt, pavement, contemporary urban landscape, modernization, industrialization post-1852, metal framework, 20th century, 21st century, current era, Transamerica Pyramid, Salesforce Tower, Coit Tower, modern San Francisco skyline, Ferry Building clock tower, modern harbor facilities, container ships"

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
                    "content": "You are a historical accuracy validator specializing in the 1852 California Gold Rush period. You must identify if an image contains ANY modern elements (post-1852) whatsoever, especially tall buildings, modern infrastructure, or technology."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Does this image show a historically accurate San Francisco Bay during the Gold Rush (1852)? It should show a panoramic view of the bay with hills, a developing wooden cityscape, harbor area, docks, and ships - all historically accurate for 1852. Identify ANY modern elements such as skyscrapers, high-rise buildings, bridges, paved roads, or any anachronistic technology. Answer YES if it's historically accurate with NO modern elements, or NO with a list of the modern elements you see."
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

def iterative_panorama_generation() -> (Optional[Image.Image], Optional[Dict], Optional[float], str):
    """Iteratively generate, analyze, and refine the panorama until a target quality is met or iterations are exhausted."""
    if not api_keys_set:
        st.error("API keys are required to generate panoramas.")
        return None, None, None, PANORAMIC_PROMPT
        
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
                    "prompt": current_prompt + " This must be San Francisco Bay during 1852 Gold Rush with absolutely NO modern elements, NO skyscrapers, NO modern buildings, NO bridges, NO modern infrastructure, NO paved roads, NO cars, NO electricity - ONLY wooden structures, wooden ships, and natural terrain as would exist in 1852.",
                    "negative_prompt": NEGATIVE_PROMPT,
                    "width": 2048,
                    "height": 1024,
                    "safety_checker": False,
                    "samples": 1,
                    "guidance_scale": 12.0,  # Increased from 7.5 for stronger adherence to prompt
                    "num_inference_steps": 75  # Increased from 50 for more detail
                }
                
                # Try adding model_id if supported by the API
                try:
                    payload["model_id"] = "realistic-vision-v51"  # Specify a model that works well with historical scenes
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
                        # Verify historical accuracy before proceeding
                        is_historically_accurate = verify_historical_accuracy(image)
                        if not is_historically_accurate:
                            print("Image contains modern elements, rejecting and trying again.")
                            image = None  # Reset image to force retry
                            attempt += 1
                            continue
                            
                        image = process_to_equirectangular(image)
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
        # Save local image data for fallback display
        buffered = io.BytesIO()
        best_image.save(buffered, format="JPEG")
        st.session_state.local_image = buffered.getvalue()
        
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
        if not api_keys_set:
            st.error("API keys are required to generate panoramas.")
            return None, None
            
        headers = {'Content-Type': 'application/json'}
        payload = {
            "key": STABLE_DIFFUSION_API_KEY,
            "prompt": self.panoramic_prompt + " This must be San Francisco Bay during 1852 Gold Rush with absolutely NO modern elements, NO skyscrapers, NO modern buildings, NO bridges, NO modern infrastructure, NO paved roads, NO cars, NO electricity - ONLY wooden structures, wooden ships, and natural terrain as would exist in 1852.",
            "negative_prompt": self.negative_prompt,
            "width": 2048,
            "height": 1024,
            "safety_checker": False,
            "samples": 1,
            "guidance_scale": 12.0,  # Increased from 7.5
            "num_inference_steps": 75  # Increased from 50
        }
        
        # Try adding model_id if supported by the API
        try:
            payload["model_id"] = "realistic-vision-v51"
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
        try:
            print(f"Initializing RAGChatbot with vector store at: {persist_directory}")
            
            # Initialize the embeddings model
            print("Initializing HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            
            # Try loading the vector store
            try:
                print("Loading Chroma vector store...")
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
                # Test if it works
                print("Testing vector store with a sample query...")
                test_results = self.vector_store.similarity_search("gold rush", k=1)
                if not test_results or len(test_results) == 0:
                    print("Vector store returned no results, using fallback")
                    self.vector_store = create_simple_gold_rush_kb()
                else:
                    print(f"Vector store test successful. Found {len(test_results)} results.")
                    print(f"Sample document: {test_results[0].page_content[:100]}...")
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
                print("Using fallback knowledge base")
                self.vector_store = create_simple_gold_rush_kb()
            
            # Initialize the language model
            print(f"Initializing ChatOpenAI ({model_name})...")
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7, 
                openai_api_key=OPENAI_API_KEY
            )
            
            # Initialize the conversation memory
            print("Initializing conversation memory...")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create the prompt template for the chain
            print("Creating prompt template...")
            template = """You are a helpful Gold Rush expert that answers questions based on the provided context.
You are knowledgeable about Gold Rush history, mining techniques, notable locations, and key figures.
Use the following context to answer the question. If the answer isn't clearly in the context,
use your general knowledge about the Gold Rush era to provide a helpful response.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer: """
            self.prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template
            )
            
            # Create the conversational retrieval chain
            print("Creating conversational retrieval chain...")
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.prompt}
            )
            
            self.initialized = True
            print("RAGChatbot initialization successful!")
            
        except Exception as e:
            print(f"❌ Error initializing RAGChatbot: {str(e)}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            
    def ask(self, question: str) -> str:
        """Process a question and return an answer."""
        if not self.initialized:
            return "I apologize, but I'm having trouble accessing my knowledge base. Please try again later or contact the administrator."
            
        try:
            print(f"\nProcessing question: {question}")
            
            # Get relevant documents from vector store
            print("Retrieving documents...")
            try:
                docs_and_scores = self.vector_store.similarity_search_with_score(question, k=3)
                print(f"Retrieved {len(docs_and_scores)} documents")
                
                for i, (doc, score) in enumerate(docs_and_scores):
                    print(f"Document {i+1} (score: {score}):")
                    print(f"Content: {doc.page_content[:150]}...")
            except Exception as e:
                print(f"Error during document retrieval: {str(e)}")
                
            # Process through conversational retrieval chain
            print("Generating response...")
            response = self.chain({"question": question})
            print("Response generated successfully")
            
            return response["answer"]
        except Exception as e:
            print(f"❌ Error in RAGChatbot.ask: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a simpler response if the chain fails
            try:
                print("Attempting direct LLM response without retrieval...")
                response = self.llm.predict(f"""Question about the Gold Rush: {question}
                
                Please provide a helpful response about the California Gold Rush based on this question.
                Focus on historical accuracy and educational content about the 1848-1855 period.""")
                return response
            except Exception as e2:
                print(f"Error in fallback response: {str(e2)}")
                return f"I apologize, but I encountered an error while trying to access my knowledge base. Please try asking your question in a different way."

# Modified get_chatbot function with better error handling
@st.cache_resource(show_spinner=False)
def get_chatbot():
    """Get or create the RAG chatbot instance."""
    try:
        # Ensure API key is available
        if not OPENAI_API_KEY:
            print("No OpenAI API key available")
            return None
            
        # Create chatbot with expanded logging
        print("Creating RAG chatbot...")
        chatbot = RAGChatbot()
        
        # Verify initialization
        if not chatbot.initialized:
            print("Chatbot failed to initialize properly")
            return None
            
        print("Chatbot created successfully")
        return chatbot
    except Exception as e:
        print(f"Error in get_chatbot: {str(e)}")
        return None

# Initialize chatbot with better error handling
try:
    print("Initializing chatbot...")
    chatbot = get_chatbot()
    if chatbot is None:
        print("Using fallback chatbot due to initialization failure")
        # Create a simple fallback chatbot for error cases
        class FallbackChatbot:
            def ask(self, question: str) -> str:
                print(f"Using fallback chatbot to answer: {question}")
                try:
                    llm = ChatOpenAI(
                        model_name="gpt-4o",
                        temperature=0.7,
                        openai_api_key=OPENAI_API_KEY
                    )
                    response = llm.predict(f"""Question about the Gold Rush: {question}
                    
                    Please provide a helpful response about the California Gold Rush based on this question.
                    Focus on historical accuracy and educational content about the 1848-1855 period.""")
                    return response
                except Exception as e:
                    print(f"Error in fallback chatbot: {str(e)}")
                    return "I apologize, but I'm having trouble accessing my knowledge base. Please try again with a different question."
        chatbot = FallbackChatbot()
    else:
        print("Chatbot initialized successfully")
except Exception as e:
    print(f"Error during chatbot initialization: {str(e)}")
    class FallbackChatbot:
        def ask(self, question: str) -> str:
            try:
                llm = ChatOpenAI(
                    model_name="gpt-4o",
                    temperature=0.7,
                    openai_api_key=OPENAI_API_KEY
                )
                response = llm.predict(f"Based on your knowledge of the California Gold Rush (1848-1855), please answer this question: {question}")
                return response
            except:
                return "I encountered an unexpected error during initialization. Please try again later or contact the administrator."
    chatbot = FallbackChatbot()

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
    # Set a default panorama URL - this is a historic SF panorama from Wikimedia Commons that's free to use
    st.session_state.image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/San_Francisco_harbor%2C_1851.jpg/2560px-San_Francisco_harbor%2C_1851.jpg"
    st.session_state.is_default_image = True
else:
    st.session_state.is_default_image = False
if "local_image" not in st.session_state:
    st.session_state.local_image = None

image_gen = ImageGenerator()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # When the button is clicked, clear any previous panorama and generate a new one.
    if st.button("Generate New Panorama"):
        # Remove default image flag when generating new panorama
        st.session_state.is_default_image = False
        with st.spinner("Generating panorama..."):
            best_image, best_analysis, best_match, final_prompt = iterative_panorama_generation()
            if best_image:
                os.makedirs("static/generated", exist_ok=True)
                # Use a unique filename by appending the current timestamp.
                save_path = f"static/generated/panorama_{int(time.time())}.jpg"
                best_image.save(save_path)
                
                # Save image in-memory for display
                buffered = io.BytesIO()
                best_image.save(buffered, format="JPEG")
                img_data = buffered.getvalue()
                st.session_state.local_image = img_data
                
                # Add the image generation details as a hidden system message to the chatbot's memory.
                if hasattr(chatbot, 'memory') and hasattr(chatbot.memory, 'chat_memory'):
                    try:
                        chatbot.memory.chat_memory.add_message(
                            SystemMessage(
                                content=f"Image Generation Details:\nFinal Prompt: {final_prompt}\nImage Evaluation: {json.dumps(best_analysis, indent=2)}"
                            )
                        )
                    except Exception as e:
                        print(f"Error adding message to chatbot memory: {str(e)}")
            else:
                st.error("Failed to generate panorama.")
    
    # Display the panorama
    if st.session_state.get("image_url") or st.session_state.get("local_image"):
        if st.session_state.get("is_default_image", False):
            # Display the default historical panorama
            st.image(st.session_state.image_url, caption="San Francisco Harbor, 1851 (Default Historical Panorama)", use_container_width=True)
            st.info("This is a default historical panorama. Use the 'Generate New Panorama' button above to create a custom AI-generated panoramic view.")
        elif st.session_state.get("local_image"):
            # Display the image directly in Streamlit
            st.image(st.session_state.local_image, caption="Generated panorama", use_container_width=True)
        else:
            # Use the image URL if neither default nor local image is available
            st.image(st.session_state.image_url, caption="Generated panorama", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 💭 Chat with the Gold Rush Guide")
    
    # Container for chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(
                    f'<div class="{message["role"]}-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
    # Place the chat input after the chat messages so it's always at the bottom.
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

