import numpy as np
import matplotlib.pylab as plt
import cv2

# Funzione per il Retinex a singola scala
def singleScaleRetinex(img, variance):
    # Calcola il logaritmo dell'immagine originale e sottrae il logaritmo dell'immagine sfocata utilizzando un filtro Gaussiano per migliorare il contrasto.
    return np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))

# Funzione per il Retinex a più scale
def multiScaleRetinex(img, variance_list):
    # Combina i risultati di Retinex calcolati su più scale di varianza.
    retinex = np.zeros_like(img) # Inizializza un array per accumulare i risultati.
    for variance in variance_list:
        # Aggiunge il risultato del Retinex a singola scala per ciascuna varianza.
        retinex += singleScaleRetinex(img, variance)
    return retinex / len(variance_list) # Media i risultati per evitare amplificazione eccessiva.

# Funzione per normalizzare i risultati di MSR e SSR
def normalize_retinex_output(img_retinex):
    # Normalizza i valori di output del Retinex per ogni canale.
    for i in range(img_retinex.shape[2]):  # Itera sui canali dell'immagine (R, G, B).
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        zero_count = count[unique.tolist().index(0)] if 0 in unique else 0
        # Determina i limiti inferiori e superiori per la normalizzazione.
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        # Applica la normalizzazione al canale corrente.
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255
    return np.uint8(img_retinex)

# Funzione per il MSR
def MSR(img, variance_list):
    img = np.float64(img) + 1.0 # Aggiunge 1 per evitare errori di logaritmo.
    img_retinex = multiScaleRetinex(img, variance_list) # Applica il Retinex multi-scala.
    return normalize_retinex_output(img_retinex) # Normalizza l'immagine risultante.

# Funzione per il SSR
def SSR(img, variance):
    img = np.float64(img) + 1.0 # Aggiunge 1 per evitare errori di logaritmo.
    img_retinex = singleScaleRetinex(img, variance) # Applica il Retinex a singola scala.
    return normalize_retinex_output(img_retinex) # Normalizza l'immagine risultante.


def MSRCR(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    '''
    Implementazione di Multi-Scale Retinex con Color Restoration (MSRCR).

    Args:
        img (numpy.ndarray): Immagine di input (array 3D per immagini a colori).
        sigma_scales (list): Valori delle varianze per le Gaussian Blur a diverse scale per MSR.
        alpha (float): Parametro di guadagno per il restauro del colore.
        beta (float): Parametro di bilanciamento per il restauro del colore.
        G (float): Fattore di guadagno globale per l'immagine finale.
        b (float): Fattore di offset per il risultato MSRCR.
        low_per (int): Percentuale inferiore per il bilanciamento del colore (non utilizzata qui).
        high_per (int): Percentuale superiore per il bilanciamento del colore (non utilizzata qui).

    Returns:
        numpy.ndarray: Immagine processata tramite MSRCR, normalizzata su scala 0-255.
    '''
    # Multi-scale retinex con color restoration

    # Converti l'immagine in formato float64 e aggiungi 1 per evitare problemi con il logaritmo
    img = img.astype(np.float64) + 1.0
    # Calcola l'immagine Multi-Scale Retinex (MSR)
    msr_img = MSR(img, variance_list) # Applica MSR sulle scale specificate
    # Calcolo del Color Restoration Function (CRF)
    # Utilizza una funzione logaritmica per enfatizzare i dettagli nei colori    
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    # Combina MSR e CRF per ottenere MSRCR
    msrcr = G * (msr_img * crf - b)
    # Normalizza il risultato per mappare i valori tra 0 e 255
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return msrcr


# Parametri
variance_list = [15, 80, 250]  # Varianze per il Retinex multi-scala.
variance = 300 # Varianza per il Retinex a singola scala.


# Lettura immagine
img = cv2.imread('img.jpg')


# Elaborazione
img_msr = MSR(img, variance_list)
img_ssr = SSR(img, variance)
img_msrcr = MSRCR(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1)


# Visualizzazione e salvataggio
# Directory di output
# out_dir = ''
# Salva le immagini nella directory specificata
# cv2.imwrite(out_dir + 'SSR.jpg', img_ssr)
# cv2.imwrite(out_dir + 'MSR.jpg', img_msr)
# cv2.imwrite(out_dir + 'MSRCR.jpg', img_msrcr)

# Visualizzazione delle immagini con Matplotlib
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.subplot(221), plt.title('original'), plt.imshow(img_rgb), plt.axis('off')

# ssr_rgb = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
# plt.subplot(222), plt.title('ssr'), plt.imshow(ssr_rgb), plt.axis('off')

# msr_rgb = cv2.cvtColor(img_msr, cv2.COLOR_BGR2RGB)
# plt.subplot(223), plt.title('msr'), plt.imshow(msr_rgb), plt.axis('off')

# msrcr_rgb = cv2.cvtColor(img_msr, cv2.COLOR_BGR2RGB)
# plt.subplot(224), plt.title('msrcr'), plt.imshow(msrcr_rgb), plt.axis('off')
# plt.show()


