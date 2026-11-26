# Súper-resolución de imágenes de gatos con SRGAN

Red generativa adversarial (SRGAN) en **TensorFlow/Keras** que duplica la resolución de imágenes de gatos (64×64 → 128×128), entrenada con un dataset personalizado.

Proyecto del curso **Redes Neuronales y Deep Learning** (Miniproyecto 3: Redes generativas) — Universidad Autónoma de Occidente, Cali, Colombia.



## Descripción

La súper-resolución toma una imagen de baja resolución (LR) y reconstruye una versión de mayor resolución (HR). Aquí se usa una SRGAN: un **generador** crea la imagen ampliada ×2 y un **discriminador** intenta distinguirla de una imagen real.

El pipeline genera cada par de entrenamiento a partir de la imagen HR: se toma un recorte aleatorio de 128×128 como objetivo y la entrada LR de 64×64 se obtiene reduciéndolo ×2 con interpolación bicúbica.

> **Resultado principal:** en este proyecto la SRGAN **no superó a la interpolación bicúbica** en PSNR ni en SSIM (ver [Resultados](#resultados)). El repositorio documenta el proceso completo y el análisis de las causas probables.

## Estructura del repositorio

```
.
├── Proyecto_Final_Redes.ipynb   # Notebook completo (datos, modelo, entrenamiento, validación, demo)
├── Informe_IEEE_SRGAN.docx      # Informe en formato IEEE
├── README.md
└── docs/
    ├── fig1_arquitectura.png
    ├── fig2_curvas.png
    ├── fig3_comparacion.jpg
    └── fig4_demo.png
```

## Dataset

- Imágenes de gatos: **69 para entrenamiento y 30 para validación**.
- Cada imagen existe en alta resolución (128×128) y en baja resolución (64×64).
- El dataset y los pesos entrenados no se incluyen en el repositorio. Estructura esperada en Google Drive:

```
MyDrive/
└── Dataset_Cats/
    ├── Train/
    │   ├── HR/   # 128×128
    │   └── LR/   # 64×64
    └── Val/
        ├── HR/
        └── LR/
```

El pipeline usa solo las imágenes HR: la LR se deriva de ellas para garantizar alineación exacta del par.

## Arquitectura


**Generador** (1,404,995 parámetros)
- Convolución 9×9 (64 filtros) + PReLU
- 16 bloques residuales (Conv 3×3 → BatchNorm → PReLU → Conv 3×3 → BatchNorm + conexión de salto)
- Convolución + BatchNorm con conexión de salto global
- Bloque de submuestreo (Conv + PixelShuffle ×2)
- Convolución 9×9 con activación `tanh`

**Discriminador** (38,247,361 parámetros)
- 8 bloques convolucionales (Conv → BatchNorm → LeakyReLU 0.2), de 64 a 512 filtros
- Dense(1024) → LeakyReLU → Dense(1, sigmoide)

**Pérdidas**

```
L_G = 1.0 · L_perceptual + 1e-4 · L_adversarial
```

- *Perceptual:* MSE entre características de VGG19 (`block3_conv3`, ImageNet, congelada).
- *Adversarial:* entropía cruzada binaria.
- *Discriminador:* entropía cruzada binaria con suavizado de etiquetas (real = 0.9).

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Entrada → salida | 64×64 → 128×128 (×2) |
| Optimizador | Adam (β₁ = 0.9) |
| Tasa de aprendizaje | G: 1e-4 · D: 5e-5 |
| Batch | 8 |
| Épocas | 100 (≈3 s por época en GPU de Colab) |
| Aumento de datos | Volteos horizontales y verticales sincronizados |
| Selección del modelo | Mejor PSNR en validación |

Se guardan checkpoints periódicos y logs de TensorBoard en Google Drive. El entrenamiento se puede reanudar si Colab se desconecta ajustando `START_EPOCH` (en este proyecto se reanudó en la época 48 desde el mejor checkpoint).


## Resultados

Evaluación en el conjunto de validación (64×64 → 128×128):

| Método | PSNR (dB) | SSIM |
|---|---|---|
| Bicúbica | **31.77** | **0.9385** |
| SRGAN | 22.80 | 0.5780 |

Dentro de la SRGAN, el PSNR varía entre 17.26 y 27.25 dB según la imagen (desviación estándar de 1.95 dB) y el SSIM entre 0.402 y 0.698.

<img width="1183" height="393" alt="image" src="https://github.com/user-attachments/assets/9fdaa80e-19c5-4ee3-a0f8-a45dc0f43571" />


Visualmente, la salida conserva la estructura del rostro pero presenta un **patrón regular de puntos (cuadrícula)** en las zonas lisas y no gana nitidez frente a la imagen ampliada.

### Análisis

Causas probables del resultado (no verificadas con experimentos adicionales):

1. **Dataset pequeño:** solo 69 imágenes de entrenamiento.
2. **La pérdida adversarial casi no influye:** la perceptual ronda 4,000–5,000 y el término adversarial aporta del orden de 10⁻³, por lo que en la práctica se entrenó casi solo con la pérdida perceptual.
3. **PSNR estancado** tras la época 48.
4. **Sin preentrenamiento con MSE** (como en el SRGAN original) ni aprendizaje de un residuo sobre la imagen bicúbica.

## Cómo ejecutarlo

1. Sube el dataset a Google Drive con la estructura indicada arriba.
2. Abre `Proyecto_Final_Redes.ipynb` en [Google Colab](https://colab.research.google.com/) y activa la GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU*).
3. Ejecuta las celdas en orden. Si es la primera vez, deja `START_EPOCH = 1` para entrenar desde cero.
4. Al final, la celda de **prueba de imagen propia** te permite subir una imagen, escalarla ×2 y descargar el resultado.

### Dependencias

Colab ya trae casi todo. Para ejecutarlo localmente necesitarías:

```
tensorflow>=2.x
numpy
matplotlib
Pillow
opencv-python
tensorboard
```

(El notebook monta Google Drive con `google.colab`, así que localmente habría que cambiar las rutas.)

## Limitaciones

- La SRGAN quedó por debajo de la bicúbica en PSNR y SSIM (≈ 9 dB menos).
- La salida presenta artefactos de patrón regular, visibles sobre todo en fondos lisos.
- La bicúbica es una referencia exigente en este montaje: la entrada LR se generó con esa misma interpolación, y los objetivos de entrenamiento resultan suaves porque las imágenes de 128×128 se amplían a 256×256 antes del recorte.
- No hay conjunto de prueba independiente: el de validación también se usa para elegir el mejor modelo.
- La demo con imágenes de 64×64 → 128×128 solo se evalúa visualmente, porque no existe imagen de referencia de 128×128.

## Mejoras propuestas

- Preentrenar el generador solo con MSE antes del entrenamiento adversarial.
- Hacer que la red aprenda la diferencia respecto a la imagen bicúbica (aprendizaje residual).
- Aumentar el peso de la pérdida adversarial o normalizar la pérdida perceptual.
- Ampliar el dataset y evaluar en un conjunto de prueba independiente.
- Revisar el origen de los artefactos de patrón regular.

## Referencias

1. I. Goodfellow *et al.*, "Generative adversarial nets," *NeurIPS*, 2014.
2. C. Ledig *et al.*, "Photo-realistic single image super-resolution using a generative adversarial network," *CVPR*, 2017.
3. K. He *et al.*, "Deep residual learning for image recognition," *CVPR*, 2016.
4. W. Shi *et al.*, "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network," *CVPR*, 2016.
5. K. Simonyan y A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *ICLR*, 2015.

## Autor

**Juan Camilo Henao Espinosa** — Universidad Autónoma de Occidente
