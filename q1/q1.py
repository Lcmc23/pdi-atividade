import cv2
import numpy as np

def detectar_formas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Intervalos de cores - vermelho e azul
    vermelho_baixo = np.array([0, 120, 70])
    vermelho_alto = np.array([10, 255, 255])
    azul_baixo = np.array([100, 150, 0])
    azul_alto = np.array([140, 255, 255])
    
    mascara_vermelho = cv2.inRange(hsv, vermelho_baixo, vermelho_alto)
    mascara_azul = cv2.inRange(hsv, azul_baixo, azul_alto)
    
    contornos_vermelho, _ = cv2.findContours(mascara_vermelho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_azul, _ = cv2.findContours(mascara_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contornos_vermelho, contornos_azul

def obter_maior_contorno(contornos):
    if contornos:
        return max(contornos, key=cv2.contourArea)
    return None

def desenhar_contornos(frame, contornos, cor):
    for cnt in contornos:
        cv2.drawContours(frame, [cnt], -1, cor, 2)

def obter_retangulo_delimitador(contorno):
    if contorno is not None:
        return cv2.boundingRect(contorno)
    return None

def verificar_colisao(ret1, ret2):
    if ret1 and ret2:
        x1, y1, largura1, altura1 = ret1
        x2, y2, largura2, altura2 = ret2
        return not (x1 + largura1 < x2 or x2 + largura2 < x1 or y1 + altura1 < y2 or y2 + altura2 < y1)
    return False

def ultrapassou_barreira(ret_maior, ret_menor):
    if ret_maior and ret_menor:
        x1, y1, largura1, altura1 = ret_maior
        x2, y2, largura2, altura2 = ret_menor
        return x1 > x2 + largura2 or x1 + largura1 < x2
    return False

video = cv2.VideoCapture("q1A.mp4")
colisao_detectada = False

while True:
    sucesso, frame = video.read()
    if not sucesso:
        break
    
    contornos_vermelho, contornos_azul = detectar_formas(frame)
    
    # Obter maior forma
    maior_vermelho = obter_maior_contorno(contornos_vermelho)
    maior_azul = obter_maior_contorno(contornos_azul)
    
    ret_vermelho = obter_retangulo_delimitador(maior_vermelho)
    ret_azul = obter_retangulo_delimitador(maior_azul)
    
    if ret_vermelho:
        x, y, largura, altura = ret_vermelho
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
    
    if ret_azul:
        x, y, largura, altura = ret_azul
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
    
    if verificar_colisao(ret_vermelho, ret_azul):
        colisao_detectada = True
        cv2.putText(frame, "COLISAO DETECTADA :(", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if colisao_detectada and ultrapassou_barreira(ret_vermelho, ret_azul):
        cv2.putText(frame, "ULTRAPASSOU BARREIRA :)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Feed", frame)
    
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break

video.release()
cv2.destroyAllWindows()