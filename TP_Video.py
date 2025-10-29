import cv2 as cv
import numpy as np
import time

# Changer chemin de la vidéo ici
cap = cv.VideoCapture("/home/lucas/Documents/Polytech/S7/Perception/Ball_Tracking/balle.mp4")

fps = 15 # Nombre d'images par seconde à traiter

epsilon = 11 # Tolérance pour la chrominance H
epsilon2 = 110 # Tolérance pour la saturation S
epsilon3 = 125 # Tolérance pour la luminosité V ; 125 permet d'exclure les zones très sombres et très claires
margin = 0.5 # Marge pour la reduction de la zone de recherche (0.5 = 50% de plus que le rayon détecté)
R=50 # Taille de la zone autour du point cliqué pour la reconnaissance de la couleur


def get_color_HSV(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        img, R = param
        # Conversion de l'image en HSV
        imgHSV = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
        global color_HSV
        global color_HSV_original
        global color_selected

        # Extraction d'une sous partie de l'image autour du point cliqué
        roi = imgHSV[y-R:y+R, x-R:x+R] # Region of interest (ROI)
        color_HSV = np.median(roi, axis=(0, 1)) # Mediane des H,S,V dans la ROI
        color_HSV_original = color_HSV.copy()   # Pour conserver la couleur originale de la balle

        color_selected = True
        print("Selected color (HSV): ", color_HSV)




def find_ball(img, epsilon, epsilon2, color, old_center=None, old_radius=None):
        
        # Reduction de l'image à traiter
        if old_center is not None and old_radius is not None:
            img_short = img[max(0,int(old_center[1]-old_radius*(1+margin))):min(img.shape[0],int(old_center[1]+old_radius*(1+margin))),
                            max(0,int(old_center[0]-old_radius*(1+margin))):min(img.shape[1],int(old_center[0]+old_radius*(1+margin)))]
            # Pour obtenir les coordonnées du cercle détecté (x,y) dans l'image originale
            offset = (max(0,int(old_center[0]-old_radius*(1+margin))),max(0 ,int(old_center[1]-old_radius*(1+margin))))
        else:
            img_short = img
            offset = (0,0)
        #cv.imshow('Image Reduite', img_short)

        # Reconaissance de la couleur en HSV
        imgHSV = cv.cvtColor(img_short.copy(), cv.COLOR_BGR2HSV)
        #cv.imshow('IMgHSV', imgHSV)
        imgresult = img_short.copy()

        # Beaucoup plus rapide avec inRange que la double boucle for
        # On definit les bornes inferieure et superieure comprises entre [0,0,0] et [179,255,255] pour la detection de la couleur
        lower_bound = np.clip(np.array([color[0]-epsilon, color[1]-epsilon2, color[2]-epsilon3]), [0,0,0], [179,255,255]).astype(np.uint8)
        upper_bound = np.clip(np.array([color[0]+epsilon, color[1]+epsilon2, color[2]+epsilon3]), [0,0,0], [179,255,255]).astype(np.uint8)
        #print("Lower bound: ", lower_bound)
        #print("Upper bound: ", upper_bound)
        imgresult = cv.inRange(imgHSV, lower_bound, upper_bound)

        """
        for i in range(len(imgHSV)):
            for j in range(len(imgHSV[i])):
                imgresult[i, j] = [0, 0, 0]
                if abs( ((int)(imgHSV[i, j][0]) - color[0])) < epsilon:
                    if abs( ((int)(imgHSV[i, j][1]) - color[1])) < epsilon2:
                        #img[i, j] = [imgresult[i,j,1], imgresult[i,j,1], imgresult[i,j,1]] # Passage en niveaux de gris selon la composante
                        imgresult[i, j] = [0, 0, 255]
        """
        
        #cv.imshow('Result', imgresult)

        # Opérations morphologiques pour réduire le bruit
        # On obtient un meilleur résultat sans aucune ouverture/fermeture
        # Néanmoins, la fonction de Hough doit alors prendre en compte plus de points (ceux dus au bruit) et est donc plus lente
        # Meilleur compromis : Fermeture seule
        """
        # Erosion et dilatation (Ouverture)
        kernel_ouverture = np.ones((3,3),np.uint8)
        iterations_ouverture = 1
        erosion1 = cv.erode(imgresult, kernel_ouverture, iterations_ouverture)
        #cv.imshow('Erosion', erosion1)
        dilation1 = cv.dilate(erosion1, kernel_ouverture, iterations_ouverture)
        #cv.imshow('Ouverture', dilation1)
        """
        # Dilatation et erosion (Fermeture)
        kernel_fermeture = np.ones((5,5),np.uint8)
        iterations_fermeture = 1
        dilation2 = cv.dilate(imgresult, kernel_fermeture, iterations_fermeture)
        #cv.imshow('Dilation', dilation2)
        erosion2 = cv.erode(dilation2, kernel_fermeture, iterations_fermeture)
        #cv.imshow('Fermeture', erosion2)

        # Gradient
        kernel_grad = np.ones((3,3),np.uint8) # Noyau du gradient
        gradient = cv.morphologyEx(erosion2, cv.MORPH_GRADIENT, kernel_grad)
        #cv.imshow('Gradient', gradient)

        # Transformée de Hough pour détecter les cercles
        rows = gradient.shape[0]
        circles = None
        circles = cv.HoughCircles(
            gradient,          # Image d'entrée (en noir et blanc)
            cv.HOUGH_GRADIENT, # Méthode de détection (toujours HOUGH_GRADIENT)
            1,                 # dp : Inverse du taux de résolution de l’accumulateur (1 = même résolution que l’image)
            rows / 8,          # minDist : Distance minimale entre les centres des cercles détectés
            param1=100,        # Seuil supérieur pour l’algorithme de Canny (détection des bords)
            param2=20,         # Seuil d’accumulation pour le centre du cercle (plus bas = plus sensible)
            minRadius=10,      # Rayon minimum des cercles à détecter
            maxRadius=500      # Rayon maximum des cercles à détecter
            )
        #print(circles)

        # Dessin du cercle détecté
        output = img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles)) # Arrondi les valeurs des coordonnées et du rayon
            i=circles[0, 0] # On garde le cercle avec le plus de votes
            center = (i[0], i[1]) # Centre du cercle dans l'image réduite
            center_original_image = (i[0]+offset[0], i[1]+offset[1]) # Centre du cercle dans l'image originale
            radius = i[2]
            # Dessiner le centre du cercle
            cv.circle(output, center_original_image, 1, (0, 255, 0), 3)
            # Dessiner le contour du cercle
            cv.circle(output, center_original_image, radius, (255, 0, 255), 3)
        else:
            center = None
            center_original_image = None
            radius = None

        cv.imshow("detected circles", output)

        # Mise à jour de la couleur en fonction du centre du cercle détecté
        if circles is not None:
            R = radius//10
            roi = imgHSV[center[1]-R:center[1]+R, center[0]-R:center[0]+R] # Define the region of interest (ROI)
            new_color = np.median(roi, axis=(0, 1))
            print("New color (HSV): ", new_color)
        else :
            new_color = color_HSV_original.copy() # Si on n'a pas detecté de cercle, on restaure la couleur de la balle originale
            print("No circle detected, back to original color")

        # Test prochaine image reduite
        """
        if circles is not None:
            next_img_short = img[max(0,int(center_original_image[1]-radius*(1+margin))):min(img.shape[0],int(center_original_image[1]+radius*(1+margin))),
                                 max(0,int(center_original_image[0]-radius*(1+margin))):min(img.shape[1],int(center_original_image[0]+radius*(1+margin)))]
            cv.imshow("Next Img Reduite",next_img_short)
        """

        return new_color, center_original_image, radius
        
fps_original = cap.get(cv.CAP_PROP_FPS) # fps de la vidéo originale 
color_HSV, center, radius = None, None, None
color_selected = False
center, radius = None, None
frame_duration_ms = int(1000 / fps)

# Version that skips frames to avoid slowdowns
while cap.isOpened():

    start_time = time.time()
    
    if color_selected == True:
        # Passe à la prochaine frame
        cap.set(cv.CAP_PROP_POS_FRAMES, int(cap.get(cv.CAP_PROP_POS_FRAMES) + fps_original/fps + time_skip*fps_original))
        # Le + time_skip*fps_original permet de rattraper le retard obtenu si le traitement de la frame est plus long que sa durée d'affichage

    # Si il n'y a pas de prochaine frame, on sort du while
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('Frame', frame)
    if color_selected == False:
        cv.setMouseCallback('Frame', get_color_HSV, param=(frame,R))
        while not color_selected : cv.waitKey(1) # Attends que l'utilisateur clique pour déclancher le callback
        cv.setMouseCallback('Frame', lambda *args : None) # Désactive le callback
        start_time = time.time()

    if color_HSV is not None:
        color_HSV, center, radius = find_ball(frame, epsilon, epsilon2, color_HSV, center, radius)

    time_skip = 0
    elapsed_ms = int((time.time() - start_time) * 1000) # Temps de calcul du traitement de la frame
    if elapsed_ms > frame_duration_ms:
        wait_time = 1
        time_skip = (elapsed_ms - frame_duration_ms)/1000 # Temps à rattraper en secondes
        #print("Skip time :", time_skip)
    else : 
        wait_time = max(1,frame_duration_ms - elapsed_ms) # Temps d'affichage d'une frame en theorie - temps de calcul

    if cv.waitKey(wait_time) & 0xFF == ord('q'): # Attends wait_time ms puis sort du while si 'q' pressé
        break


"""
# Version without time_skip that doesn't prevent video slowdonwns if fps too high (easier for testing)
while cap.isOpened():

    start_time = time.time()

    if color_selected == True:
        # Passe à la prochaine frame
        cap.set(cv.CAP_PROP_POS_FRAMES, int(cap.get(cv.CAP_PROP_POS_FRAMES) + fps_original/fps))

    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('Frame', frame)
    if color_selected == False:
        cv.setMouseCallback('Frame', get_color_HSV, param=(frame,R))
        while not color_selected : cv.waitKey(1) # Attends que l'utilisateur clique pour déclancher le callback
        cv.setMouseCallback('Frame', lambda *args : None) # Désactive le callback
        start_time = time.time()

    if color_HSV is not None:
        color_HSV, center, radius = find_ball(frame, epsilon, epsilon2, color_HSV, center, radius)

    elapsed_ms = int((time.time() - start_time) * 1000) # Temps de calcul du traitement de la frame
    wait_time = max(1,frame_duration_ms - elapsed_ms) # Temps d'affichage d'une frame en theorie - temps de calcul
    #wait_time = 0 # Pour faire du pas à pas
    if cv.waitKey(wait_time) & 0xFF == ord('q'): # Attends wait_time ms puis sort du while si 'q' pressé
        break
"""
cap.release() 
cv.destroyAllWindows()

