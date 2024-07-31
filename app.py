import torch
import streamlit as st
from torchvision import transforms
from PIL import Image
from gtts import gTTS
from io import BytesIO
import base64
from res import enderes
from Flickr import FlickrDataset

data_location = "./archive"
model_path = 'attention_model_state.pth'  # Update with the path to your model


dataset = FlickrDataset(
    root_dir=data_location + "/Images",
    captions_file=data_location + "/captions.txt",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
)

embed_size = 300
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512

def load_model1(model_path, dataset, embed_size, attention_dim, encoder_dim, decoder_dim):
    vocab_size = len(dataset.vocab)
    
    model = enderes(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    )

    state_dict = torch.load(model_path)
    model_state_dict = state_dict['state_dict']
    
    model.load_state_dict(model_state_dict, strict=False)
    
   
    model.vocab = dataset.vocab

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    
    quantized_model.eval()
    return quantized_model


model = load_model1(model_path, dataset, embed_size, attention_dim, encoder_dim, decoder_dim)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def get_caps_from(features_tensors, model, dataset):
    model.eval()
    with torch.no_grad():
        features_tensors = features_tensors.to(next(model.parameters()).device)
        features = model.encoder(features_tensors)
        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        # Exclude the <end> token
        caps = [word for word in caps if word != '<end>' and word != '<UNK>']
        caption = ' '.join(caps)
    return caption, caps, alphas

def generate_caption(model, image_tensor, dataset):
    caption, caps, alphas = get_caps_from(image_tensor, model, dataset)
    return caption

#speech to text
def text_to_speech(text):
    tts = gTTS(text)
    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return audio


def audio_to_base64(audio):
    audio_bytes = audio.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return audio_base64


def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url({url});
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_url("https://www.datarobot.com/wp-content/uploads/2020/11/DataRobot_DataRobot_Vision_resource_card_BG_v.1.0.jpg")

st.markdown('<h1 style="font-family:sans-serif; color:#5c2f1b; font-size: 60px; font-weight: 900">Vision</h1>', unsafe_allow_html=True)

menu = ["Home", "Upload Image", "Take Photo", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.write('Vision - an attempt to ease the lives of visually impaired people')
elif choice == "Upload Image":
    st.write('Upload an image')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image_tensor = preprocess_image(image)
        caption = generate_caption(model, image_tensor, dataset)
        st.write('Generated Caption:')
        st.markdown(f'<h3 style="font-family:sans-serif; color:#5c2f1b;">{caption}</h3>', unsafe_allow_html=True)
        audio = text_to_speech(caption)
        audio_base64 = audio_to_base64(audio)
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
elif choice == "Take Photo":
    st.write('Take a photo ')
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image.', use_column_width=True)
        image_tensor = preprocess_image(image)
        caption = generate_caption(model, image_tensor, dataset)
        st.write('Generated Caption:')
        st.markdown(f'<h3 style="font-family:sans-serif; color:#5c2f1b;">{caption}</h3>', unsafe_allow_html=True)
        audio = text_to_speech(caption)
        audio_base64 = audio_to_base64(audio)
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
elif choice == "About":
    st.markdown('<h2 style="font-family:sans-serif; color:white;">About</h2>', unsafe_allow_html=True)
    about_text = """
    <ul style="font-family:sans-serif; font-size:100px; color:white;">
        <li>Vision is an attempt to ease the lives of visually impaired people.</li>
        <li>The original idea has been to make an interface such as some eyewear that could read the surroundings and notify the user.</li>
        <li>This can also be extended to CCTVs for security purposes.</li>
        <li>Any person's image at the doorstep can be captured and passed to a mobile app or some interface to notify the user.</li>
        <li>Note: This app is just an interface to give an idea of how the model works.</li>
    </ul>
    """
    st.markdown(about_text, unsafe_allow_html=True)
