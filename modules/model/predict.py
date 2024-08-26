import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from tensorflow.image import resize




def get_class_names_from_directory(data_directory):
    """
    Fonction pour extraire les noms des classes à partir des noms des répertoires.

    Parameters:
        data_directory (str): Chemin vers le répertoire contenant les sous-répertoires de pays.

    Returns:
        list: Liste des noms de classes.
    """
    # Utiliser os.listdir pour obtenir la liste des sous-répertoires
    #class_names = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    class_names = ['Argentina', 'Austria', 'Chile', 'Germany', 'Israel', 'Italy', 'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Romania', 'Singapore', 'Sweden', 'Taiwan', 'Thailand']
    # Optionnel: Trier les noms de classes pour assurer un ordre cohérent
    class_names.sort()

    return class_names


# Remplacez 'mon_modele.h5' par le nom de votre fichier de modèle
model = load_model('../../models/20240823-170144.h5')

def prepare_image(img_path, target_size=(66, 153)):
    """
    Charge et prépare une image pour la prédiction.

    Parameters:
        img_path (str): Chemin de l'image.
        target_size (tuple): Taille à laquelle redimensionner l'image (default=(256, 256)).

    Returns:
        np.array: Tableau numpy de l'image préparée.
    """
    # Charger l'image avec la taille cible
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = resize(img_array, target_size)
    # Convertir l'image en tableau numpy
    # Ajouter une dimension supplémentaire (pour le batch)
    img_array = np.expand_dims(img_array, axis=0)
    # Normaliser les valeurs des pixels entre 0 et 1
    img_array = img_array / 255.0
    return img_array

# Chemin de l'image à prédire
#img_path = '../../data/compressed_dataset/Netherlands/canvas_1629477224.jpg'  # Remplacez par le chemin de votre image
prepared_image = prepare_image('../../data/574db65a1289dcd5e7f92b0674fac33a.jpg')

# Faire une prédiction
predictions = model.predict(prepared_image)


# Exemple d'utilisation
data_directory = '../../data/compressed_dataset/'  # Remplacez par le chemin de votre répertoire de données
class_names = get_class_names_from_directory(data_directory)
print("Noms des classes:", class_names)


# Trouver l'indice de la classe avec la plus haute probabilité
print(predictions,'\n\n')
predicted_class_index = np.argmax(predictions)
print(predicted_class_index)
predicted_country = class_names[predicted_class_index]

# Afficher le résultat
print(f"L'image est prédite comme appartenant au pays : {predicted_country}")

# Afficher l'image avec la prédiction
plt.imshow(image.load_img(img_path))
plt.title(f"Prédiction: {predicted_country}")
plt.axis('off')
plt.show()


def main():
        prepare_image(img_path='../../data/compressed_dataset/Netherlands/canvas_1629477224.jpg', )


if __name__ == "__main__":
    main()
