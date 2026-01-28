#Code for testing the AI
#Takes image, resizes and preprocesses them before passing them to AI
#Converts image to usuable numPy array for AI, then pulls out AI predictions and prints result to screen

import os
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input # type: ignore
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

#List of classes
categories = [
    "Alcohol Inky Cap, Common Ink Cap (Coprinopsis atramentaria)",
    "Artist's Bracket, Artist's Conk, Bear Bread (Ganoderma applanatum)",
    "Aspen Bolete (Leccinum albostipitatum)",
    "Aspen Bracket (Phellinus tremulae)",
    "Bay Bolete (Imleria badia)",
    "Birch Bolete, Rough-Stemmed Bolete, Scaber Stalk (Leccinum scabrum)",
    "Birch Polypore, Birch Bracket, Razor Strop (Fomitopsis betulina)",
    "Bitter Oyster, Astringent Panus, Luminescent Panellus (Panellus stipticus)",
    "Blue Stain, Green Elfcup (Chlorociboria aeruginascens)",
    "Blue-green Stropharia (Stropharia aeruginosa)",
    "Blusher (Amanita rubescens)",
    "Blushing Bracket (Daedaleopsis confragosa)",
    "Boreal Oakmoss (Evernia mesomorpha)",
    "Brick Cap, Chestnut Mushroom, Cinnamon Cap (Hypholoma lateritium)",
    "Brown Roll-Rim (Paxillus involutus)",
    "Chaga (Inonotus obliquus)",
    "Chicken-of-the-woods (Laetiporus sulphureus)",
    "Clouded Agaric, Cloudy Clitocybe (Clitocybe nebularis)",
    "Common Puffball, Warted Puffball, Gem-studded Puffball (Lycoperdon perlatum)",
    "Common Stinkhorn (Phallus impudicus)",
    "Coral Spot (Nectria cinnabarina)",
    "Coral Tooth Fungus, Comb Coral Mushroom (Hericium coralloides)",
    "Crown Coral (Artomyces pyxidatus)",
    "Dark-Stalked Bolete, Orange Birch Bolete (Leccinum versipelle)",
    "Delicious Milk Cap, Saffron Milk Cap, Red Pine Mushroom (Lactarius deliciosus)",
    "Devil's or Gray Urn (Urnula craterium)",
    "Dryad's Saddle, Pheasant's Back (Cerioporus squamosus)",
    "Fairy Inkcap, Fairy Bonnet (Coprinellus disseminatus)",
    "False Blusher, Panther Cap (Amanita pantherina)",
    "False Chanterelle (Hygrophoropsis aurantiaca)",
    "False Death Cap (Amanita citrina)",
    "False Morel (Gyromitra esculenta)",
    "False Tinder Fungus, Hoof Fungus (Fomes fomentarius)",
    "False Turkey Tail (Stereum hirsutum)",
    "Fly Agaric, Fly Amanita (Amanita muscaria)",
    "Golden Chanterelle (Cantharellus cibarius)",
    "Golden Pholiota (Pholiota aurivella)",
    "Green Dog Lichen (Peltigera aphthosa)",
    "Hairy Turkey Tail (Trametes hirsuta)",
    "Hammered Shield Lichen (Parmelia sulcata)",
    "Hooded False Morel, Elfin Saddle (Gyromitra infula)",
    "Hooded Rosette Lichen (Physcia adscendens)",
    "Indian Oyster, Italian Oyster, Pheonix Mushroom (Pleurotus pulmonarius)",
    "Late Fall Oyster [North America] (Sarcomyxa serotina)",
    "Maritime sunburst lichen (Xanthoria parietina)",
    "Mealy Shadow Lichen (Phaeophyscia orbicularis)",
    "Mica Cap, Glistening Inky Cap (Coprinellus micaceus)",
    "Monk's-Hood Lichen (Hypogymnia physodes)",
    "Northern Honey Fungus (Armillaria borealis)",
    "Oakmoss (Evernia prunastri)",
    "Ochre bracket (Trametes ochracea)",
    "Oyster Mushroom (Pleurotus ostreatus)",
    "Parasol Mushroom (Macrolepiota procera)",
    "Pear-shaped Puffball (Apioperdon pyriforme)",
    "Penny Bun (Boletus edulis)",
    "Plums and Custard (Tricholomopsis rutilans)",
    "Powdered Sunshine Lichen (Vulpicida pinastri)",
    "Ragbag Lichen (Platismatia glauca)",
    "Ravenel's Red Stinkhorn (Mutinus ravenelii)",
    "Red-Belted Conk or Bracket (Fomitopsis pinicola)",
    "Red-Capped Scaber Stalk (Leccinum aurantiacum)",
    "Reindeer Cup Lichen (Cladonia rangiferina)",
    "Scaly Dog Pelt Lichen (Peltigera praetextata)",
    "Scarlet Elfcup (Sarcoscypha austriaca)",
    "Script Lichen, Secret Writing Lichen (Graphis scripta)",
    "Shaggy Ink Cap (Coprinus comatus)",
    "Shaggy Scalycap (Pholiota squarrosa)",
    "Sheathed Woodtuft (Kuehneromyces mutabilis)",
    "Silver Leaf (Chondrostereum purpureum)",
    "Slippery Jack (Suillus luteus)",
    "Smoky Polypore (Bjerkandera adusta)",
    "Snow Mushroom, Giants False Morel (Gyromitra gigas)",
    "Splash Cups (Crucibulum laeve)",
    "Splitgill (Schizophyllum commune)",
    "Star-tipped Cup Lichen (Cladonia stellaris)",
    "Sulphur Tuft, Clustered Woodlover (Hypholoma fasciculare)",
    "Summer Cep (Boletus reticulatus)",
    "Tamarack Jack (Suillus grevillei)",
    "Tar Spot (Rhytisma acerinum)",
    "Tiger's Eye (Coltricia perennis)",
    "Tree Lungwort (Lobaria pulmonaria)",
    "Tree Moss (Pseudevernia furfuracea)",
    "Trembling Merulius, Jelly Rot (Phlebia Tremellosa, Formerly 'Merulius tremellosus')",
    "Tricolor Maze Polypore (Daedaleopsis tricolor)",
    "True Iceland Lichen (Cetraria islandica)",
    "Trumpet Cup Lichen (Cladonia fimbriata)",
    "Turkey Tail (Trametes versicolor)",
    "Ugly Milk-Cap (Lactarius turpis)",
    "Velvet Foot or Stem or Shank, Wild Enoki (Flammulina velutipes)",
    "Violet Toothed Polypore (Trichaptum biforme)",
    "Weeping Bolete (Suillus granulatus)",
    "Willow Bracket, Fire Sponge, False Tinder Polypore (Phellinus igniarius)",
    "Witch's Butter (Tremella mesenterica)",
    "Witches Cauldron (Sarcosoma globosum)",
    "Wood Blewit (Lepista Nuda)",
    "Woolly Milkcap, Bearded Milkcap (Lactarius torminosus)",
    "Wrinkled Crust (Phlebia radiata)",
    "Wrinkled Thimble Morel (Verpa bohemica)",
    "Yellow Fairy Cups, Lemon Discos (Calycina citrina)",
    "Yellow Staghorn (Calocera viscosa)"
]

#Load the model
path_for_saved_model = r"/mnt/c/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/models/finalmodel.keras"
model = tf.keras.models.load_model(path_for_saved_model)

#Function to classify the image
def classify_image(imageFile, top_k=5):
    x = []

    #Load the image and resize it to input to mobileNetv2
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    #Convert image to numPy array
    x = image.img_to_array(img)

    #Add a batch dimension (i.e there is ONE image in this batch)
    x=np.expand_dims(x,axis=0)

    #Preprocesses the shape of the image for the AI
    x=preprocess_input(x)

    #Pass image through the model and predict
    #Take out highest value and convert to 1
    pred = model.predict(x)[0]

    # Get top-k indices sorted by confidence
    top_indices = pred.argsort()[-top_k:][::-1]

    # Map indices to class names and confidence
    results = [(categories[i], float(pred[i])) for i in top_indices]

    return results


#Run function with desired image
imagePath = r"/mnt/c/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/Mushroom Classification.v1i.folder/Amanita/011_DoZoYI2vj20_jpg.rf.9b4881c62cbeeccf613cbd3b25cec276.jpg"
top_predictions = classify_image(imagePath, top_k=5)

# Print results
print("Top 5 predictions:")
for cls, prob in top_predictions:
    print(f"{cls}: {prob*100:.2f}%")

#Create Cv2 GUI
img = cv2.imread(imagePath)
y0, dy = 50, 30
for i, (cls, prob) in enumerate(top_predictions):
    text = f"{cls}: {prob*100:.1f}%"
    y = y0 + i*dy
    img = cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

cv2.imshow("Predictions", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
