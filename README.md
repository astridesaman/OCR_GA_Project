# 🧠 OCR avec Réseau de Neurones et Algorithme Génétique

## 📌 Description du projet

Ce projet compare deux méthodes d’optimisation des poids d’un réseau de neurones pour la reconnaissance optique de caractères (OCR) :

1. **Backpropagation (Descente de gradient)**
2. **Algorithme génétique (Neuroévolution)**

L’objectif est d’analyser expérimentalement les différences en termes de :

- Précision (accuracy)
- Temps d’entraînement
- Vitesse de convergence
- Complexité computationnelle

Le dataset utilisé est **MNIST**.

---

## 📂 Dataset

Nous utilisons le dataset :

**MNIST (Modified National Institute of Standards and Technology database)**  
- 60 000 images d'entraînement  
- 10 000 images de test  
- Images 28x28 en niveaux de gris  
- 10 classes (chiffres 0–9)

Les images sont normalisées avant entraînement.

---

## 🏗️ Architecture du modèle

Nous utilisons un réseau de neurones multi-couches (MLP) simple :

- **Entrée** : 784 neurones (28×28)
- **Couche cachée** : 32 neurones (ReLU)
- **Sortie** : 10 neurones (classification)

Schéma simplifié :
