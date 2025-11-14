from flask import Flask, request, Response
from flask.templating import render_template
from flask import request
from werkzeug.utils import secure_filename
from app import app
import torch
from PIL import Image
import torchvision.transforms as T
import os

def predict(model, img, tr, classes):
    img_tensor = tr(img)
    out = model(img_tensor.unsqueeze(0))
    pred, idx = torch.max(out, 1)
    return classes[idx]

def get_transforms():
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.ToTensor())
    return T.Compose(transform)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    res = None
    recommendation = None
    image_path = None
    if request.method == 'POST':
        classes = ['acanthosis-nigricans',
                'acne',
                'acne-scars',
                'alopecia-areata',
                'dry',
                'melasma',
                'oily',
                'vitiligo',
                'warts']
        f = request.files['file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(path)
        model = torch.load('./skin-model-pokemon.pt', map_location=torch.device('cpu'), weights_only=False)
        device = torch.device('cpu')
        model.to(device);
        img = Image.open(path).convert("RGB")
        tr = get_transforms()
        res = predict(model, img, tr, classes)

        # Recommendations and images for each skin type
        skin_data = {
            'acanthosis-nigricans': {
                'description': 'Acanthosis nigricans is a skin condition characterized by dark, thickened, velvety patches, often in body folds. It can be a sign of insulin resistance or other underlying conditions.',
                'recommendations': [
                    'Maintain a healthy weight through diet and exercise.',
                    'Use moisturizers to keep skin hydrated.',
                    'Consult a dermatologist for treatment options like retinoids or laser therapy.',
                    'Avoid tight clothing that may irritate the skin.'
                ],
                'image': 'images/acne.png'  # Using available images as placeholders
            },
            'acne': {
                'description': 'Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells, leading to pimples, blackheads, and whiteheads.',
                'recommendations': [
                    'Use gentle, oil-free cleansers twice daily.',
                    'Avoid picking or squeezing pimples to prevent scarring.',
                    'Consider topical treatments like benzoyl peroxide or salicylic acid.',
                    'Keep your pillowcases and phone screens clean to reduce bacteria.'
                ],
                'image': 'images/acne.png'
            },
            'acne-scars': {
                'description': 'Acne scars are marks left on the skin after acne lesions heal. They can be atrophic (depressed) or hypertrophic (raised) and vary in severity.',
                'recommendations': [
                    'Try laser therapy or chemical peels for scar reduction.',
                    'Use retinoids to promote skin cell turnover.',
                    'Protect skin from sun exposure with SPF 30+ sunscreen.',
                    'Consider microneedling or dermal fillers for deeper scars.'
                ],
                'image': 'images/acne.png'
            },
            'alopecia-areata': {
                'description': 'Alopecia areata is an autoimmune condition that causes hair loss in patches. It occurs when the immune system attacks hair follicles.',
                'recommendations': [
                    'Consult a specialist for treatments like corticosteroids or immunotherapy.',
                    'Use gentle hair care products to avoid further irritation.',
                    'Consider wigs or head coverings for confidence.',
                    'Join support groups for emotional support.'
                ],
                'image': 'images/Normal.png'
            },
            'dry': {
                'description': 'Dry skin occurs when the skin lacks moisture, leading to flakiness, itchiness, and a rough texture. It can be caused by environmental factors or underlying conditions.',
                'recommendations': [
                    'Use humidifiers to add moisture to the air.',
                    'Apply moisturizers immediately after bathing.',
                    'Avoid hot showers; opt for lukewarm water.',
                    'Wear protective clothing in cold or windy weather.'
                ],
                'image': 'images/dry.png'
            },
            'melasma': {
                'description': 'Melasma is a skin condition characterized by brown or gray-brown patches, usually on the face. It\'s often triggered by sun exposure and hormonal changes.',
                'recommendations': [
                    'Use broad-spectrum sunscreen daily with SPF 50+.',
                    'Avoid sun exposure, especially between 10 AM and 4 PM.',
                    'Consider topical creams like hydroquinone or kojic acid.',
                    'Wear hats and protective clothing outdoors.'
                ],
                'image': 'images/Normal.png'
            },
            'oily': {
                'description': 'Oily skin produces excess sebum, leading to a shiny appearance and potential for clogged pores. It can be influenced by genetics, hormones, or diet.',
                'recommendations': [
                    'Use oil-free, non-comedogenic skincare products.',
                    'Cleanse your face twice daily with a gentle cleanser.',
                    'Consider mattifying moisturizers to control shine.',
                    'Use blotting papers throughout the day to absorb excess oil.'
                ],
                'image': 'images/oily.png'
            },
            'vitiligo': {
                'description': 'Vitiligo is a condition that causes loss of skin color in patches due to the destruction of melanocytes. It can affect any part of the body.',
                'recommendations': [
                    'Protect depigmented skin from sun with high SPF sunscreen.',
                    'Use cosmetic camouflage makeup for coverage.',
                    'Consult for phototherapy or topical treatments.',
                    'Wear protective clothing and accessories.'
                ],
                'image': 'images/Normal.png'
            },
            'warts': {
                'description': 'Warts are small, non-cancerous growths caused by the human papillomavirus (HPV). They can appear on any part of the body.',
                'recommendations': [
                    'Avoid touching or picking at warts to prevent spreading.',
                    'Use over-the-counter treatments like salicylic acid.',
                    'See a doctor for cryotherapy or laser removal.',
                    'Keep the area clean and dry.'
                ],
                'image': 'images/acne.png'
            }
        }

        if res in skin_data:
            description = skin_data[res]['description']
            recommendations = skin_data[res]['recommendations']
            image_path = skin_data[res]['image']

    return render_template("index.html", res=res, description=description, recommendations=recommendations, image_path=image_path)
