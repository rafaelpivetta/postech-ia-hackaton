import os
from PIL import Image,ImageDraw, ImageFont
import numpy as np
import cv2


def criar_pasta_para_facas(diretorio="detected_knives"):  
    caminho_absoluto = os.path.abspath(diretorio)  
    if not os.path.exists(caminho_absoluto):  
        os.makedirs(caminho_absoluto)  
        print(f"Pasta criada: {caminho_absoluto}")  
    else:  
        print(f"Pasta já existente: {caminho_absoluto}")  
    return caminho_absoluto  

# Função para rastrear ou criar ID para facas  
def rastrear_ou_criar_id(box, trackers, proximo_id):  
    for existing_id, tracker_info in trackers.items():  
        tracked_box, _ = tracker_info  
        iou = calcular_iou(box, tracked_box)  
        if iou > 0.5:  
            return existing_id, proximo_id, trackers  
    object_id = proximo_id  
    proximo_id += 1  
    return object_id, proximo_id, trackers  

# Função para calcular IOU (Intersection over Union)  
def calcular_iou(box1, box2):  
    x1, y1, w1, h1 = box1  
    x2, y2, w2, h2 = box2  
    xi1 = max(x1, x2)  
    yi1 = max(y1, y2)   
    xi2 = min(x1 + w1, x2 + w2)  
    yi2 = min(y1 + h1, y2 + h2)  
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)  
    box1_area = w1 * h1  
    box2_area = w2 * h2  
    iou = inter_area / float(box1_area + box2_area - inter_area)  
    return iou


import cv2
import numpy as np

def ProcessarWEBCAM(boxes, confidence_threshold, image_np):
    trackers = {}  # Dicionário para armazenar os rastreadores
    detections = []  # Lista para armazenar as detecções
    has_detections = False  # Variável para verificar se há detecções

    # Iterar sobre as caixas de detecção
    for box in boxes:
        # Obter a confiança (aqui assumimos que 'conf' é um tensor)
        confidence = box.conf.item()  # Pega o valor escalar da confiança

        # Verifique se a confiança está acima do limiar
        if confidence >= confidence_threshold:
            has_detections = True

            # Extrair as coordenadas xyxy (convertendo para inteiros)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # .cpu().numpy() para converter o tensor em array

            # Calcular o centro da caixa
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Inicializar o rastreador com o centro da caixa
            object_id = len(trackers) + 1  # Atribuir um ID único ao objeto
            trackers[object_id] = [center]  # Armazenar o ponto inicial no dicionário

            # Adicionar a detecção à lista de detecções
            detections.append({
                'confidence': float(confidence),
                'id': object_id,
                'box': [x1, y1, x2, y2]
            })

    return has_detections, detections, trackers

# Função para atualizar os rastreadores usando movimento óptico
def atualizar_rastreadores(image_np, trackers):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    new_trackers = {}

    for obj_id, points in trackers.items():
        if points:
            prev_point = np.float32(points[-1]).reshape(-1, 1, 2)
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(gray, gray, prev_point, None, **lk_params)

            if status[0] == 1:
                new_trackers[obj_id] = points + [tuple(new_point[0][0])]

    return new_trackers

# Parâmetros para o algoritmo de Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 

def Guardar_facas_detectadas(detections, knife_dir, image_np):  
    for detection in detections:  
        box = detection['box']  
        x1, y1, x2, y2 = map(int, box)  
        knife_image = image_np[y1:y2, x1:x2]  
        knife_pil = Image.fromarray(knife_image)  
        knife_pil.save(os.path.join(knife_dir, f"knife_{detection['id']}.jpg"))
        
def Desenhar(image, box, label):
    draw = ImageDraw.Draw(image)

    # Carregue a fonte padrão
    font = ImageFont.load_default()

    # Desenhe o retângulo
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)

    # Calcule o tamanho do texto usando a fonte
    #text_width, text_height = ImageFont.getsize(label, font=font)

    # Desenhe o retângulo de fundo para o texto
    # draw.rectangle(
    #     [box[0], box[1] - text_height, box[0] + text_width, box[1]],
    #     fill="red"
    # )

    # # Desenhe o texto
    # draw.text((box[0], box[1] - text_height), label, fill="white", font=font)

    return image